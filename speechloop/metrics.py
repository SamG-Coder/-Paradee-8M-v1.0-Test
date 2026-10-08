from __future__ import annotations
from collections.abc import Sequence


def distance(reference: Sequence, hypothesis: Sequence) -> int:
    prev = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        row = [i]
        for j, b in enumerate(hypothesis, 1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j - 1] + (a != b)))
        prev = row
    return prev[-1]


def score(pairs: list[tuple[str, str]]) -> dict:
    if not pairs:
        raise ValueError("No evaluation examples")
    words = sum(len(ref.split()) for ref, _ in pairs)
    chars = sum(len(ref) for ref, _ in pairs)
    we = sum(distance(ref.split(), hyp.split()) for ref, hyp in pairs)
    ce = sum(distance(ref, hyp) for ref, hyp in pairs)
    return {"samples": len(pairs), "word_errors": we, "reference_words": words,
            "wer": we / max(1, words), "char_errors": ce, "reference_chars": chars,
            "cer": ce / max(1, chars), "exact_match": sum(a == b for a, b in pairs) / len(pairs)}
