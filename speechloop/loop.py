"""Fresh synthesis plus training-only hard-example replay; never trains on test text."""
from __future__ import annotations
from pathlib import Path
import json
from .curriculum import generated_texts
from .data import read_jsonl, load_manifest, synthesize_corpus, atomic_json
from .text import split_for, normalize
from .train import train_model
from .model import ModelConfig


def run_loop(engine, root: Path, texts: list[str], rounds: int = 3, fresh_per_round: int = 100,
             steps_per_round: int = 300, batch_size: int = 8, seed: int = 17,
             device: str = "cpu", threads: int = 2, hidden: int = 128,
             phone_weight: float = 0.0) -> dict:
    if min(rounds, fresh_per_round, steps_per_round) < 1:
        raise ValueError("rounds, fresh_per_round and steps_per_round must be positive")
    if phone_weight and not engine.phoneme_vocab:
        raise ValueError("Auxiliary phoneme training requires Paradee targets")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "loop_state.json"
    manifest = root / "corpus" / "manifest.jsonl"
    texts = list(dict.fromkeys(normalize(t) for t in texts))
    initial_ids = sorted(texts)
    import hashlib
    signature = {"seed": seed, "engine": engine.fingerprint,
                 "initial_text_sha256": hashlib.sha256(json.dumps(initial_ids).encode()).hexdigest(),
                 "fresh_per_round": fresh_per_round, "steps_per_round": steps_per_round,
                 "batch_size": batch_size, "hidden": hidden, "phone_weight": phone_weight}
    state = {"completed_rounds": 0, "rounds": [], "signature": signature}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["signature"] != signature:
            raise ValueError("Loop configuration changed; use a new directory to avoid corrupting the experiment")
    # Idempotent: also finishes an interrupted initial synthesis.
    manifest = synthesize_corpus(texts, root / "corpus", engine, variants=1, seed=seed)
    previous = Path(state["rounds"][-1]["checkpoint"]) if state["rounds"] else None
    # rounds is the desired TOTAL, not an endless/unbounded download/training job.
    for round_index in range(state["completed_rounds"], rounds):
        if round_index > 0:
            existing = read_jsonl(manifest)
            known_texts = {r["text"] for r in existing}
            # Persist the plan BEFORE synthesis so an interrupted round does not
            # choose a different set after seeing its own partially written rows.
            plan_path = root / f"round_{round_index:03d}" / "synthesis_plan.json"
            if plan_path.exists():
                planned = json.loads(plan_path.read_text(encoding="utf-8"))["texts"]
            else:
                candidates = generated_texts(fresh_per_round * 5, seed + round_index * 1009)
                fresh = [t for t in candidates if split_for(t, seed) == "train" and t not in known_texts][:fresh_per_round]
                if len(fresh) < fresh_per_round:
                    raise ValueError("Not enough fresh training text; expand the generator vocabulary or start a new corpus")
                difficulty_path = previous.parent / "difficulty.json"
                difficulty = json.loads(difficulty_path.read_text()) if difficulty_path.exists() else {}
                ranked = sorted((r for r in existing if r["split"] == "train"),
                                key=lambda r: difficulty.get(r["id"], 0), reverse=True)
                hard = list(dict.fromkeys(r["text"] for r in ranked))[:max(1, fresh_per_round // 4)]
                planned = fresh + hard
                atomic_json(plan_path, {"texts": planned, "round": round_index})
            manifest = synthesize_corpus(planned, root / "corpus", engine, variants=1,
                                        seed=seed, forced_split="train", round_id=round_index)
        rows = load_manifest(manifest)
        train = [r for r in rows if r["split"] == "train"]
        validation = [r for r in rows if r["split"] == "validation"]
        if not validation or not any(r["split"] == "test" for r in rows):
            raise ValueError("Initial corpus needs nonempty validation and test splits; use at least 60 diverse texts")
        config = ModelConfig(hidden=hidden, phoneme_classes=(max(engine.phoneme_vocab.values()) + 1)
                             if phone_weight and engine.phoneme_vocab else 0)
        run = root / f"round_{round_index:03d}"
        current = run / "last.pt"
        resume = current if current.exists() else previous
        already = 0
        if current.exists():
            import torch
            total = int(torch.load(current, weights_only=True, map_location="cpu")["step"])
            prev_steps = 0
            if previous:
                prev_steps = int(torch.load(previous, weights_only=True, map_location="cpu")["step"])
            already = total - prev_steps
        remaining = max(0, steps_per_round - already)
        if remaining:
            _, report = train_model(train, validation, run, steps=remaining, batch_size=batch_size,
                                    seed=seed, config=config, resume=resume, device_name=device,
                                    threads=threads, phone_weight=phone_weight)
        else:
            report = json.loads((run / "training_report.json").read_text())
        previous = current.resolve()
        state["completed_rounds"] = round_index + 1
        state["rounds"].append({"round": round_index, "train_examples": len(train),
                                "validation_examples": len(validation), "checkpoint": str(previous),
                                "validation_cer": report["final_validation"]["cer"],
                                "validation_wer": report["final_validation"]["wer"]})
        atomic_json(state_path, state)
    return state
