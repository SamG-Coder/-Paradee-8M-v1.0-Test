"""Conservative English targets: preserve words; never silently drop digits.

The same normalized string is sent to TTS and used as the target. This avoids
training on '$12' while the audio says 'twelve dollars'. For numbers, write the
spoken form explicitly; unsupported input is rejected, not silently corrupted.
"""
from __future__ import annotations
import hashlib
import re
import unicodedata
from collections.abc import Iterable

# CTC blank=0; no language model or word dictionary is used by the recognizer.
SYMBOLS = " abcdefghijklmnopqrstuvwxyz'"
ALPHABET = ["<blank>"] + list(SYMBOLS)
TO_ID = {c: i for i, c in enumerate(ALPHABET)}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.replace("’", "'").replace("‘", "'"))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    if re.search(r"\d|[$€£%&@+=]", text):
        raise ValueError("Write numbers, symbols and abbreviations in spoken words before training: " + text[:100])
    text = re.sub(r"[\-‐‑–—/]", " ", text)
    text = re.sub(r'''[.,!?;:"()\[\]{}…“”«»]''', " ", text)
    text = re.sub(r"\s+", " ", text).strip(" '")
    unknown = set(text) - set(SYMBOLS)
    if unknown:
        raise ValueError(f"Unsupported target characters: {sorted(unknown)}")
    if not text:
        raise ValueError("Empty text after normalization")
    return text


def encode(text: str) -> list[int]:
    return [TO_ID[c] for c in normalize(text)]


def decode(ids: Iterable[int], collapse: bool = True) -> str:
    result, previous = [], None
    for value in ids:
        i = int(value)
        if not 0 <= i < len(ALPHABET):
            raise ValueError(f"Invalid token {i}")
        if i != 0 and (not collapse or i != previous):
            result.append(ALPHABET[i])
        previous = i
    return "".join(result).strip()


def text_id(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:20]


def split_for(text: str, seed: int = 17) -> str:
    # Assignment is independent of synthesis voice/speed and input file order.
    digest = hashlib.sha256(f"{seed}:{normalize(text)}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "test" if bucket < 10 else "validation" if bucket < 20 else "train"


def sentences(text: str, max_words: int = 18) -> list[str]:
    if max_words < 1:
        raise ValueError("max_words must be positive")
    result = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if not sentence.strip():
            continue
        words = normalize(sentence).split()
        result.extend(" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words))
    return list(dict.fromkeys(result))
