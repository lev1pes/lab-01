import inspect

import pytest

from findex.tokenize import tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PyThOn PYTHON Straße STRASSE", ["python", "python", "strasse", "strasse"]),
        ("Привіт, СВІТ! Їжак", ["привіт", "світ", "їжак"]),
        ("café cafe\u0301", ["café", "café"]),
        ("Hello... (world)! 😀", ["hello", "world"]),
        ("", []),
        ("don't п’ять пʼять п'ять", ["don't", "п'ять", "п'ять", "п'ять"]),
        ("'word' ‘word’", ["word", "word"]),
        ("well-known пів–Європи a—b", ["well", "known", "пів", "європи", "a", "b"]),
        (
            "Python3 2026 3.12 snake_case",
            ["python3", "2026", "3", "12", "snake", "case"],
        ),
        ("  \n\t___---!!!", []),
    ],
)
def test_tokenize(text, expected):
    assert list(tokenize(text)) == expected


def test_generator_is_single_pass():
    assert inspect.isgeneratorfunction(tokenize)
    tokens = tokenize("One two")
    assert iter(tokens) is tokens
    assert next(tokens) == "one"
    assert list(tokens) == ["two"]
    assert list(tokens) == []
