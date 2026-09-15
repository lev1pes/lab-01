"""NFC + casefold; апострофи всередині зберігаються, дефіси розділяють слова."""

import re
import unicodedata
from collections.abc import Iterator

APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "‘": "'"})
WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*")


def tokenize(text: str) -> Iterator[str]:
    """Видавати токени з літер і цифр Unicode, не створюючи список токенів."""
    text = unicodedata.normalize("NFC", text)
    text = unicodedata.normalize("NFC", text.casefold()).translate(APOSTROPHES)
    for match in WORD.finditer(text):
        yield match.group()
