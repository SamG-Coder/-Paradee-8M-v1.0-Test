from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request
import torch
from .audio import read_audio, features, write_audio
from .curriculum import generated_texts
from .data import atomic_json, atomic_jsonl, load_manifest, synthesize_corpus, FeatureCache
from .model import ModelConfig, TinyRecognizer
from .text import sentences, normalize, text_id, split_for, decode
from .train import train_model, load_model, evaluate, choose_device
from .tts import make_engine


def add_engine(p):
    p.add_argument("--engine", choices=["paradee", "espeak"], default="paradee")
    p.add_argument("--model-dir", type=Path, help="Local Paradee ONNX + config; its phonemizer also needs cached assets")
    p.add_argument("--voice", default="en-us", help="eSpeak test voice only; Paradee has one fixed voice")
    p.add_argument("--threads", type=int, default=2)


def add_text(p):
    p.add_argument("--text-file", type=Path, action="append", default=[])
    p.add_argument("--generate", type=int, default=120)
    p.add_argument("--seed", type=int, default=17)


def collect_text(args) -> list[str]:
    result = generated_texts(args.generate, args.seed) if args.generate else []
    for path in args.text_file:
        result.extend(sentences(path.read_text(encoding="utf-8-sig")))
    if not result:
        raise ValueError("Provide --generate or --text-file")
    return list(dict.fromkeys(result))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SpeechLoopLab: text -> frozen Paradee -> trainable tiny speech recognizer")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("info", help="Describe the randomly initialized recognizer")
    q.add_argument("--hidden", type=int, default=128)
    q = sub.add_parser("speak", help="Check the actual TTS backend and write audio")
    add_engine(q)
    q.add_argument("text")
    q.add_argument("--output", type=Path, default=Path("hello.wav"))
    q = sub.add_parser("generate", help="Cache labelled speech; the backend is never silently substituted")
    add_engine(q); add_text(q)
    q.add_argument("--output", type=Path, default=Path("runs/corpus"))
    q.add_argument("--variants", type=int, default=2)
    q = sub.add_parser("train", help="Train on train rows; choose checkpoints on validation only")
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--run", type=Path, default=Path("runs/train"))
    q.add_argument("--steps", type=int, default=500)
    q.add_argument("--batch-size", type=int, default=8)
    q.add_argument("--lr", type=float, default=0.002)
    q.add_argument("--threads", type=int, default=2)
    q.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    q.add_argument("--seed", type=int, default=17)
    q.add_argument("--hidden", type=int, default=128)
    q.add_argument("--layers", type=int, default=2)
    q.add_argument("--resume", type=Path)
    q.add_argument("--no-augment", action="store_true")
    q.add_argument("--eval-every", type=int, default=100)
    q.add_argument("--phoneme-weight", type=float, default=0.0)
    q = sub.add_parser("evaluate", help="Explicit held-out evaluation, not adaptive curriculum input")
    q.add_argument("--checkpoint", type=Path, required=True)
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--split", choices=["train", "validation", "test"], default="test")
    q.add_argument("--output", type=Path, default=Path("evaluation.json"))
    q.add_argument("--allow-text-overlap", action="store_true", help="Explicit same-text diagnostic only, not held-out-text accuracy")
    q.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    q.add_argument("--threads", type=int, default=2)
    q = sub.add_parser("transcribe", help="Greedy character CTC decoding; no dictionary or transcript lookup")
    q.add_argument("audio", type=Path)
    q.add_argument("--checkpoint", type=Path, required=True)
    q.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    q.add_argument("--threads", type=int, default=2)
    q = sub.add_parser("loop", help="Run a bounded adaptive synthetic-data experiment with resume")
    add_engine(q); add_text(q)
    q.add_argument("--output", type=Path, default=Path("runs/loop"))
    q.add_argument("--rounds", type=int, default=3)
    q.add_argument("--fresh", type=int, default=100)
    q.add_argument("--steps", type=int, default=300, help="Optimizer steps per round")
    q.add_argument("--batch-size", type=int, default=8)
    q.add_argument("--hidden", type=int, default=128)
    q.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    q.add_argument("--phoneme-weight", type=float, default=0.0)
    q = sub.add_parser("fetch-text", help="Download an explicitly licensed UTF-8 plain-text source; never auto-crawls")
    q.add_argument("url")
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--license", required=True, help="Your record of the source's actual license; not verified by this tool")
    q.add_argument("--rights-confirmed", action="store_true", required=True)
    q = sub.add_parser("import-human", help="Import real recordings from CSV with audio,text,split columns")
    q.add_argument("csv", type=Path)
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("test-paradee", help="Real integration test: downloads/loads Paradee and synthesizes speech")
    add_engine(q)
    q.add_argument("--output", type=Path, default=Path("runs/paradee_test"))
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "info":
            print(json.dumps(TinyRecognizer(ModelConfig(hidden=args.hidden)).describe(), indent=2))
        elif args.command in {"speak", "generate", "test-paradee", "loop"}:
            if args.command == "test-paradee" and args.engine != "paradee":
                raise ValueError("test-paradee must use the real Paradee engine")
            engine = make_engine(args.engine, args.threads, args.model_dir, args.voice)
            if args.command == "speak":
                # Keep targets and speech consistent even for standalone sanity checks.
                audio, _ = engine.synthesize(normalize(args.text))
                write_audio(args.output, audio)
                print(args.output)
            elif args.command == "generate":
                manifest = synthesize_corpus(collect_text(args), args.output, engine, args.variants, args.seed)
                print(manifest)
            elif args.command == "test-paradee":
                import time
                import numpy as np
                args.output.mkdir(parents=True, exist_ok=True)
                start = time.perf_counter()
                audio, phones = engine.synthesize("the small bird can see the blue river")
                elapsed = time.perf_counter() - start
                if not len(phones) or not np.isfinite(audio).all() or np.sqrt(np.mean(audio ** 2)) < 1e-5:
                    raise RuntimeError("Paradee integration produced invalid speech")
                write_audio(args.output / "paradee.wav", audio)
                report = {"status": "passed", "actual_engine": engine.fingerprint, "seconds": elapsed,
                          "audio_seconds": len(audio) / 16000, "phoneme_targets": len(phones),
                          "caution": "This checks execution and signal validity, not perceptual speech quality."}
                atomic_json(args.output / "result.json", report)
                print(json.dumps(report, indent=2))
            else:
                from .loop import run_loop
                result = run_loop(engine, args.output, collect_text(args), args.rounds, args.fresh,
                                  args.steps, args.batch_size, args.seed, args.device, args.threads,
                                  args.hidden, args.phoneme_weight)
                print(json.dumps(result, indent=2))
        elif args.command == "train":
            rows = load_manifest(args.manifest)
            phones = [p for r in rows if r["split"] == "train" for p in r.get("phoneme_ids", [])]
            if args.phoneme_weight and not phones:
                raise ValueError("Auxiliary phoneme training requires a corpus made with Paradee")
            # Use the entire teacher vocabulary, not just IDs seen in this tiny batch.
            info_path = args.manifest.parent / "dataset_info.json"
            vocab = json.loads(info_path.read_text()).get("phoneme_vocab", {}) if info_path.exists() else {}
            classes = max(max(vocab.values(), default=0), max(phones, default=0)) + 1 if args.phoneme_weight else 0
            config = ModelConfig(hidden=args.hidden, layers=args.layers, phoneme_classes=classes)
            _, report = train_model([r for r in rows if r["split"] == "train"],
                                    [r for r in rows if r["split"] == "validation"], args.run,
                                    args.steps, args.batch_size, args.lr, args.device, args.threads,
                                    args.seed, config, not args.no_augment, args.resume,
                                    args.eval_every, args.phoneme_weight)
            print(json.dumps({"model": report["model"], "total_steps": report["total_steps"],
                              "report": str(args.run / "training_report.json")}, indent=2))
        elif args.command == "evaluate":
            torch.set_num_threads(args.threads)
            model, state = load_model(args.checkpoint, args.device)
            rows = [r for r in load_manifest(args.manifest) if r["split"] == args.split]
            overlap = set(state.get("training_text_ids", [])) & {text_id(r["text"]) for r in rows}
            if overlap and args.split != "train" and not args.allow_text_overlap:
                raise ValueError("Evaluation contains training text; use new text or --allow-text-overlap for an explicitly labelled diagnostic")
            result = evaluate(model, rows, FeatureCache(), choose_device(args.device))
            result["training_text_overlap"] = len(overlap)
            result["overlap_checked_against_checkpoint"] = "training_text_ids" in state
            result["split"] = args.split
            result["checkpoint"] = str(args.checkpoint)
            atomic_json(args.output, result)
            print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, indent=2))
        elif args.command == "transcribe":
            torch.set_num_threads(args.threads)
            model, _ = load_model(args.checkpoint, args.device)
            model.eval()
            x = features(read_audio(args.audio))
            with torch.inference_mode():
                logits, lengths, _ = model(x[None].to(choose_device(args.device)), torch.tensor([len(x)]))
            print(decode(logits[0, :int(lengths[0])].argmax(-1).cpu()))
        elif args.command == "fetch-text":
            if urllib.parse.urlsplit(args.url).scheme != "https":
                raise ValueError("Use an HTTPS URL for a plain-text source")
            request = urllib.request.Request(args.url, headers={"User-Agent": "SpeechLoopLab/0.1 research"})
            with urllib.request.urlopen(request, timeout=30) as response:
                if not response.headers.get("Content-Type", "").startswith("text/plain"):
                    raise ValueError("Only text/plain is accepted; do not supply a webpage or PDF")
                payload = response.read(2_000_001)
            if len(payload) > 2_000_000:
                raise ValueError("Text source exceeds the two-megabyte limit")
            content = payload.decode("utf-8-sig")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding="utf-8")
            atomic_json(args.output.with_suffix(args.output.suffix + ".source.json"),
                        {"url": args.url, "declared_license": args.license, "rights_confirmed_by_user": True,
                         "sha256": hashlib.sha256(payload).hexdigest()})
            print(args.output)
        elif args.command == "import-human":
            import csv
            rows = []
            with args.csv.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if not {"audio", "text"} <= set(reader.fieldnames or []):
                    raise ValueError("CSV needs audio and text column headers; split is optional")
                for item in reader:
                    text = normalize(item["text"])
                    audio = (args.csv.resolve().parent / item["audio"]).resolve()
                    split = item.get("split") or "test"
                    rows.append({"id": hashlib.sha256((str(audio) + text).encode()).hexdigest()[:24],
                                 "text": text, "text_id": text_id(text), "audio": str(audio), "split": split,
                                 "engine": "human", "phoneme_ids": []})
            atomic_jsonl(args.output, rows)
            load_manifest(args.output)  # validate paths, transcripts and cross-split overlap
            print(args.output)
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
