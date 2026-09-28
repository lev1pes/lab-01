"""Вікно вихідного тексту: найбільше різних термінів, потім входжень."""

import unicodedata
from collections import Counter
from collections.abc import Iterator

from findex.tokenize import tokenize


def matching_spans(text: str, terms: set[str]) -> Iterator[tuple[int, int, str]]:
    start = None
    for i in range(len(text) + 1):
        character = text[i] if i < len(text) else " "
        accepted = character.isalnum() or unicodedata.category(character).startswith(
            "M"
        )
        accepted = accepted or character in "'’ʼ‘"
        if accepted and start is None:
            start = i
        if not accepted and start is not None:
            begin, end = start, i
            while begin < end and text[begin] in "'’ʼ‘":
                begin += 1
            while end > begin and text[end - 1] in "'’ʼ‘":
                end -= 1
            words = tuple(tokenize(text[begin:end]))
            if len(words) == 1 and words[0] in terms:
                yield begin, end, words[0]
            start = None


def snippet(text: str, terms: set[str], radius: int = 80) -> str:
    if radius < 0:
        raise ValueError("Радіус має бути невід'ємним")
    spans = list(matching_spans(text, terms))
    if not spans:
        return text[: 2 * radius].replace("\n", " ") + (
            "…" if len(text) > 2 * radius else ""
        )
    # Ковзне вікно: кожне входження додається й видаляється не більше разу.
    left = right = 0
    counts: Counter[str] = Counter()
    best = (-1, -1)
    chosen = spans[0]
    for match in spans:
        low, high = match[0] - radius, match[0] + radius
        while right < len(spans) and spans[right][0] <= high:
            counts[spans[right][2]] += 1
            right += 1
        while left < right and spans[left][0] < low:
            word = spans[left][2]
            counts[word] -= 1
            if not counts[word]:
                del counts[word]
            left += 1
        value = (len(counts), right - left)
        if value > best:
            best, chosen = value, match
    low = max(0, chosen[0] - radius)
    high = min(len(text), chosen[0] + radius)
    high = max(high, chosen[1])
    pieces = ["…" if low else ""]
    cursor = low
    for begin, end, _ in spans:
        if low <= begin and end <= high:
            pieces.extend((text[cursor:begin], "[", text[begin:end], "]"))
            cursor = end
    pieces.extend((text[cursor:high], "…" if high < len(text) else ""))
    return "".join(pieces).replace("\n", " ")
