from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import os
import platform
import random
import time
import numpy as np
import torch
from torch import nn
from .audio import SAMPLE_RATE
from .curriculum import sampling_weights, update_difficulty
from .data import FeatureCache, atomic_json, assert_disjoint
from .metrics import distance, score
from .model import TinyRecognizer, ModelConfig
from .text import decode, ALPHABET, text_id


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch installation has no available CUDA device")
    if name not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda or auto")
    return torch.device(name)


def atomic_torch_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def load_model(path: Path, device: str = "cpu") -> tuple[TinyRecognizer, dict]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("alphabet") != ALPHABET or state.get("sample_rate") != SAMPLE_RATE:
        raise ValueError("Checkpoint uses an incompatible alphabet or sample rate")
    model = TinyRecognizer(ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    return model.to(choose_device(device)), state


@torch.inference_mode()
def evaluate(model: TinyRecognizer, rows: list[dict], cache: FeatureCache,
             device: torch.device, batch_size: int = 8) -> dict:
    if not rows:
        raise ValueError("No rows in requested evaluation split")
    model.eval()
    pairs, details = [], []
    start = time.perf_counter()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        x, lengths, _, _ = cache.batch(batch)
        logits, out_lengths, _ = model(x.to(device), lengths)
        paths = logits.argmax(-1).cpu()
        for row, path, length in zip(batch, paths, out_lengths):
            predicted = decode(path[:int(length)])
            pairs.append((row["text"], predicted))
            details.append({"id": row["id"], "reference": row["text"], "prediction": predicted})
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = score(pairs)
    result.update(seconds=time.perf_counter() - start, predictions=details,
                  engines=sorted({r.get("engine", "human_or_unspecified") for r in rows}))
    return result


def _phone_loss(logits: torch.Tensor | None, rows: list[dict], lengths: torch.Tensor,
                criterion: nn.CTCLoss, device: torch.device) -> torch.Tensor | None:
    if logits is None:
        return None
    indices = [i for i, row in enumerate(rows) if row.get("phoneme_ids")]
    if not indices:
        return None
    targets = [rows[i]["phoneme_ids"] for i in indices]
    for i, phones in zip(indices, targets):
        minimum = len(phones) + sum(a == b for a, b in zip(phones, phones[1:]))
        if minimum > int(lengths[i]):
            raise ValueError("Not enough audio frames for auxiliary phoneme CTC")
        if min(phones) <= 0 or max(phones) >= logits.size(-1):
            raise ValueError("Phoneme ID is blank or outside the saved teacher vocabulary")
    y = torch.tensor([p for ps in targets for p in ps], dtype=torch.long, device=device)
    yl = torch.tensor([len(ps) for ps in targets], dtype=torch.long)
    return criterion(logits[indices].log_softmax(-1).transpose(0, 1), y, lengths[indices], yl)


def train_model(train_rows: list[dict], validation_rows: list[dict], run_dir: Path,
                steps: int = 500, batch_size: int = 8, learning_rate: float = 0.002,
                device_name: str = "cpu", threads: int = 2, seed: int = 17,
                config: ModelConfig | None = None, augment: bool = True,
                resume: Path | None = None, evaluate_every: int = 100,
                phone_weight: float = 0.2) -> tuple[TinyRecognizer, dict]:
    if min(steps, batch_size, threads, evaluate_every) < 1 or learning_rate <= 0 or phone_weight < 0:
        raise ValueError("Invalid training hyperparameters")
    if not train_rows or any(r["split"] != "train" for r in train_rows):
        raise ValueError("Training must contain only nonempty TRAIN rows")
    if any(r["split"] != "validation" for r in validation_rows):
        raise ValueError("Model selection accepts only VALIDATION rows, never TEST rows")
    assert_disjoint(train_rows, validation_rows)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    device = choose_device(device_name)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "last.pt").exists() and resume is None:
        raise FileExistsError("Run directory already has a checkpoint; pass --resume or use another --run")
    model = TinyRecognizer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    start_step, best_cer, difficulty = 0, float("inf"), {}
    training_text_ids = {text_id(r["text"]) for r in train_rows}
    if resume is not None:
        model, state = load_model(resume, str(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        if "optimizer" not in state:
            raise ValueError("Resume needs last.pt or best.pt, not the inference-only recognizer.pt")
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        training_text_ids.update(state.get("training_text_ids", []))
        if training_text_ids & {text_id(r["text"]) for r in validation_rows}:
            raise ValueError("Evaluation leaks text previously learned by the resumed checkpoint")
        start_step = int(state["step"])
        best_cer = float(state.get("best_cer", float("inf")))
        difficulty = state.get("difficulty", {})
        torch.set_rng_state(state["torch_rng"].cpu())
        rng.bit_generator.state = state["numpy_rng"]
        random.setstate(state["python_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    if model.config.phoneme_classes and not any(r.get("phoneme_ids") for r in train_rows):
        raise ValueError("Phoneme head enabled, but no training row has phoneme labels")
    cache = FeatureCache()
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)
    baseline = evaluate(model, validation_rows, cache, device, batch_size) if validation_rows else None
    initial_train = evaluate(model, train_rows[:min(32, len(train_rows))], cache, device, batch_size)
    history = []
    started = time.perf_counter()
    losses = []
    journal = run_dir / "metrics.jsonl"

    def save(step: int, best: bool = False):
        state = {"format_version": 1, "model": model.state_dict(), "model_config": asdict(model.config),
                 "optimizer": optimizer.state_dict(), "step": step, "best_cer": best_cer,
                 "alphabet": ALPHABET, "sample_rate": SAMPLE_RATE, "difficulty": difficulty,
                 "training_text_ids": sorted(training_text_ids),
                 "torch_rng": torch.get_rng_state(), "numpy_rng": rng.bit_generator.state,
                 "python_rng": random.getstate(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        atomic_torch_save(run_dir / "last.pt", state)
        if best:
            atomic_torch_save(run_dir / "best.pt", state)

    step = completed_step = start_step
    try:
        for step in range(start_step + 1, start_step + steps + 1):
            model.train()
            weights = sampling_weights(train_rows, difficulty)
            selected = rng.choice(len(train_rows), size=batch_size, replace=True, p=weights)
            batch = [train_rows[int(i)] for i in selected]
            x, lengths, targets, target_lengths = cache.batch(batch, rng if augment else None)
            optimizer.zero_grad(set_to_none=True)
            logits, out_lengths, phones = model(x.to(device), lengths)
            char_loss = criterion(logits.float().log_softmax(-1).transpose(0, 1),
                                  targets.to(device), out_lengths, target_lengths)
            phone_loss = _phone_loss(phones, batch, out_lengths, criterion, device) if phone_weight else None
            loss = char_loss + phone_weight * phone_loss if phone_loss is not None else char_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite CTC loss; inspect transcript/audio alignment. No silent zeroing.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, error_if_nonfinite=True)
            optimizer.step()
            completed_step = step
            losses.append(float(loss.detach()))
            # Only TRAIN predictions influence the sampling distribution.
            for row, path, length in zip(batch, logits.detach().argmax(-1).cpu(), out_lengths):
                pred = decode(path[:int(length)])
                error = distance(row["text"], pred) / max(1, len(row["text"]))
                update_difficulty(difficulty, row["id"], error)
            if step % evaluate_every == 0 or step == start_step + steps:
                validation = evaluate(model, validation_rows, cache, device, batch_size) if validation_rows else None
                entry = {"step": step, "mean_train_loss": sum(losses) / len(losses),
                         "elapsed_seconds": time.perf_counter() - started,
                         "validation": {k: v for k, v in validation.items() if k != "predictions"} if validation else None}
                losses.clear()
                history.append(entry)
                with journal.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, allow_nan=False) + "\n")
                improved = validation is not None and validation["cer"] < best_cer
                if improved:
                    best_cer = validation["cer"]
                save(step, best=improved)
                suffix = f" val CER={validation['cer']:.3f} WER={validation['wer']:.3f}" if validation else " (no validation)"
                print(f"step={step} loss={entry['mean_train_loss']:.4f}{suffix}", flush=True)
    except KeyboardInterrupt:
        save(completed_step)
        print("Interrupted; saved last.pt. Resume to continue.", flush=True)
        raise
    final_train = evaluate(model, train_rows, cache, device, batch_size)
    final_val = evaluate(model, validation_rows, cache, device, batch_size) if validation_rows else None
    report = {"device": str(device), "torch_version": str(torch.__version__), "python": platform.python_version(),
              "threads": threads, "seed": seed, "steps_this_run": step - start_step, "total_steps": step,
              "seconds": time.perf_counter() - started, "model": model.describe(),
              "initial_train_subset": initial_train, "initial_validation": baseline,
              "final_train": final_train, "final_validation": final_val, "history": history,
              "important": "Synthetic train/validation results do not establish real human speech accuracy. TTS is frozen; recognizer weights are new."}
    atomic_json(run_dir / "training_report.json", report)
    atomic_json(run_dir / "difficulty.json", difficulty)
    # Inference weights exclude the optimizer and RNG state; this is the deployable size.
    atomic_torch_save(run_dir / "recognizer.pt", {"model": model.state_dict(), "model_config": asdict(model.config),
                                               "alphabet": ALPHABET, "sample_rate": SAMPLE_RATE,
                                               "training_text_ids": sorted(training_text_ids)})
    return model, report
