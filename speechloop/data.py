from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import os
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .audio import features, read_audio, write_audio, augment_audio, augment_features, SAMPLE_RATE
from .text import normalize, encode, text_id, split_for


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def atomic_json(path: Path, value: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def load_manifest(path: Path) -> list[dict]:
    root = Path(path).resolve().parent
    rows = read_jsonl(path)
    seen_ids, text_splits, audio_splits = set(), {}, {}
    for row in rows:
        for key in ("id", "text", "audio", "split"):
            if key not in row:
                raise ValueError(f"Manifest row missing {key}")
        row["text"] = normalize(row["text"])
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if row["id"] in seen_ids:
            raise ValueError("Duplicate sample ID in manifest")
        seen_ids.add(row["id"])
        tid = text_id(row["text"])
        if tid in text_splits and text_splits[tid] != row["split"]:
            raise ValueError("Data leakage: identical normalized text occurs in different splits")
        text_splits[tid] = row["split"]
        p = Path(row["audio"])
        p = p.resolve() if p.is_absolute() else (root / p).resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        if str(p) in audio_splits and audio_splits[str(p)] != row["split"]:
            raise ValueError("Data leakage: identical audio path occurs in different splits")
        audio_splits[str(p)] = row["split"]
        row["audio"] = str(p)
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def synthesize_corpus(texts: list[str], root: Path, engine, variants: int = 2,
                      seed: int = 17, forced_split: str | None = None, round_id: int = 0) -> Path:
    if variants < 1:
        raise ValueError("variants must be positive")
    if forced_split not in {None, "train", "validation", "test"}:
        raise ValueError("Invalid forced split")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    rows = read_jsonl(manifest) if manifest.exists() else []
    known = {r["id"] for r in rows}
    assigned = {text_id(r["text"]): r["split"] for r in rows}
    for text in texts:
        tid = text_id(text)
        if tid in assigned and assigned[tid] != (forced_split or split_for(text, seed)):
            raise ValueError("Data leakage: attempted to move existing text to another split")
    fingerprint = json.dumps(engine.fingerprint, sort_keys=True)
    unique = list(dict.fromkeys(normalize(t) for t in texts))
    try:
        for i, text in enumerate(unique):
            split = forced_split or split_for(text, seed)
            for variant in range(variants):
                key = f"{fingerprint}:{seed}:{round_id}:{text}:{variant}"
                digest = hashlib.sha256(key.encode()).hexdigest()
                sid = digest[:24]
                relative = f"audio/{sid}.wav"
                if sid in known and (root / relative).is_file():
                    continue
                if sid in known:
                    rows = [r for r in rows if r["id"] != sid]
                    known.remove(sid)
                rng = random.Random(int(digest[:16], 16))
                speed = 1.0 if variant == 0 and round_id == 0 else rng.uniform(0.85, 1.15)
                audio, phonemes = engine.synthesize(text, speed=speed)
                duration = len(audio) / SAMPLE_RATE
                if not 0.05 <= duration <= 30:
                    raise ValueError(f"Unexpected duration {duration:.2f}s for {text!r}; use shorter sentences")
                write_audio(root / relative, audio)
                rows.append({"id": sid, "text_id": text_id(text), "text": text, "audio": relative,
                             "split": split, "duration_seconds": duration, "sample_rate": SAMPLE_RATE,
                             "engine": engine.name, "speed": speed, "phoneme_ids": phonemes,
                             "round": round_id, "engine_fingerprint": engine.fingerprint})
                known.add(sid)
            if (i + 1) % 25 == 0:
                atomic_jsonl(manifest, rows)
                print(f"Synthesized {i + 1}/{len(unique)} texts ({len(rows)} audio rows)", flush=True)
    finally:
        # A failed engine download or synthesis must not erase completed samples.
        if rows:
            atomic_jsonl(manifest, rows)
    atomic_json(root / "dataset_info.json", {"engine": engine.fingerprint, "seed": seed,
                "sample_rate": SAMPLE_RATE, "phoneme_vocab": engine.phoneme_vocab,
                "splits": {s: sum(r["split"] == s for r in rows) for s in ["train", "validation", "test"]}})
    return manifest


class FeatureCache:
    """Bounded cache: a large book corpus must not occupy unbounded RAM."""
    def __init__(self, max_entries: int = 512):
        self.max_entries = max_entries
        self.cache: OrderedDict = OrderedDict()

    def get(self, row: dict) -> torch.Tensor:
        key = row["audio"]
        if key not in self.cache:
            self.cache[key] = features(read_audio(Path(key)))
            if len(self.cache) > self.max_entries:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]

    def batch(self, rows: list[dict], rng: np.random.Generator | None = None) -> tuple:
        xs = []
        for row in rows:
            # Occasionally recompute from audio to exercise real microphone-like changes.
            if rng is not None and rng.random() < 0.25:
                x = features(augment_audio(read_audio(Path(row["audio"])), rng))
            else:
                x = self.get(row)
            xs.append(augment_features(x, rng) if rng is not None else x)
        lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
        ys = [encode(r["text"]) for r in rows]
        targets = torch.tensor([v for y in ys for v in y], dtype=torch.long)
        target_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
        # CTC needs an extra frame for each consecutive repeated token.
        minima = torch.tensor([len(y) + sum(a == b for a, b in zip(y, y[1:])) for y in ys])
        if torch.any((lengths + 1) // 2 < minima):
            bad = [r["text"] for r, l, m in zip(rows, (lengths + 1) // 2, minima) if l < m]
            raise ValueError(f"Not enough audio frames for CTC targets (possibly bad transcripts): {bad}")
        return pad_sequence(xs, batch_first=True), lengths, targets, target_lengths


def assert_disjoint(train: list[dict], evaluation: list[dict]) -> None:
    overlap = {text_id(r["text"]) for r in train} & {text_id(r["text"]) for r in evaluation}
    if overlap:
        raise ValueError(f"Evaluation leaks {len(overlap)} training texts")
