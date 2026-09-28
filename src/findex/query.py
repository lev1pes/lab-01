"""Рекурсивний спуск: OR < AND (зокрема неявний) < NOT < дужки."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from types import NotImplementedType

from findex.models import Index
from findex.tokenize import tokenize


class Query(ABC):
    @abstractmethod
    def evaluate(self, index: Index) -> set[int]: ...

    @abstractmethod
    def terms(self, negative: bool = False) -> set[str]: ...

    def __and__(self, other: object) -> And | NotImplementedType:
        return And(self, other) if isinstance(other, Query) else NotImplemented

    def __or__(self, other: object) -> Or | NotImplementedType:
        return Or(self, other) if isinstance(other, Query) else NotImplemented

    def __invert__(self) -> Not:
        return Not(self)


@dataclass(frozen=True, slots=True)
class Term(Query):
    value: str

    def evaluate(self, index: Index) -> set[int]:
        return {posting.doc_id for posting in index.get(self.value, ())}

    def terms(self, negative: bool = False) -> set[str]:
        return set() if negative else {self.value}


def adjacent(
    left: Sequence[int], right: Sequence[int], distance: int
) -> tuple[int, ...]:
    """Зберегти початки фрази, для яких існує наступна потрібна позиція."""
    result: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        target = left[i] + distance
        if target == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif target < right[j]:
            i += 1
        else:
            j += 1
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Phrase(Query):
    words: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.words:
            raise ValueError("Порожня фраза")

    def evaluate(self, index: Index) -> set[int]:
        if not index.has_positions:
            raise ValueError("Фразовий пошук потребує --positions")
        candidates = Term(self.words[0]).evaluate(index)
        for word in self.words[1:]:
            candidates &= Term(word).evaluate(index)
        result: set[int] = set()
        for doc in candidates:
            starts = index.positions(self.words[0], doc)
            for offset, word in enumerate(self.words[1:], 1):
                starts = adjacent(starts, index.positions(word, doc), offset)
                if not starts:
                    break
            if starts:
                result.add(doc)
        return result

    def terms(self, negative: bool = False) -> set[str]:
        return set() if negative else set(self.words)


@dataclass(frozen=True, slots=True)
class And(Query):
    left: Query
    right: Query

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def terms(self, negative: bool = False) -> set[str]:
        return self.left.terms(negative) | self.right.terms(negative)


@dataclass(frozen=True, slots=True)
class Or(Query):
    left: Query
    right: Query

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def terms(self, negative: bool = False) -> set[str]:
        return self.left.terms(negative) | self.right.terms(negative)


@dataclass(frozen=True, slots=True)
class Not(Query):
    child: Query

    def evaluate(self, index: Index) -> set[int]:
        return set(index.doc_meta) - self.child.evaluate(index)

    def terms(self, negative: bool = False) -> set[str]:
        return self.child.terms(not negative)


class Parser:
    def __init__(self, text: str) -> None:
        self.tokens = re.findall(r'"[^"]*"|[()]|[^\s()"]+', text)
        if text.count('"') % 2:
            raise ValueError("Незакрита фраза")
        if len(self.tokens) > 256:
            raise ValueError("Запит має понад 256 частин")
        self.position = 0

    def peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise ValueError("Неочікуваний кінець запиту")
        self.position += 1
        return token

    def or_expr(self) -> Query:
        node = self.and_expr()
        while self.peek() == "OR":
            self.take()
            node = Or(node, self.and_expr())
        return node

    def and_expr(self) -> Query:
        node = self.not_expr()
        while self.peek() not in (None, "OR", ")"):
            if self.peek() == "AND":
                self.take()
            node = And(node, self.not_expr())
        return node

    def not_expr(self) -> Query:
        if self.peek() == "NOT":
            self.take()
            return Not(self.not_expr())
        return self.atom()

    def atom(self) -> Query:
        token = self.take()
        if token == "(":
            node = self.or_expr()
            if self.take() != ")":
                raise ValueError("Потрібна закривальна дужка")
            return node
        if token in ("AND", "OR", ")"):
            raise ValueError(f"Неочікуваний оператор: {token}")
        words = tuple(tokenize(token.strip('"')))
        if token.startswith('"'):
            return Phrase(words)
        if len(words) != 1:
            raise ValueError("Операнд має утворювати один токен; фрази беріть у лапки")
        return Term(words[0])


def parse(text: str) -> Query:
    if len(text) > 4096:
        raise ValueError("Запит надто довгий (максимум 4096 символів)")
    parser = Parser(text)
    try:
        node = parser.or_expr()
        if parser.peek() is not None:
            raise ValueError("Зайва закривальна дужка")
        return node
    except RecursionError as error:
        raise ValueError("Запит має надто глибоку вкладеність") from error
