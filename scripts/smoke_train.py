"""Actual speech optimization, not a mock or an ASR quality benchmark.

Run from the repository root: python scripts/smoke_train.py --engine espeak
Selecting Paradee requires its real dependencies; no automatic fallback exists.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speechloop.data import synthesize_corpus, load_manifest, FeatureCache, atomic_json
from speechloop.model import ModelConfig
from speechloop.train import train_model, load_model, evaluate
from speechloop.tts import make_engine
from speechloop.audio import write_audio


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', choices=['paradee', 'espeak'], default='paradee')
    p.add_argument('--output', type=Path, default=Path('runs/smoke'))
    p.add_argument('--steps', type=int, default=350)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--model-dir', type=Path)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    engine = make_engine(args.engine, threads=args.threads, model_dir=args.model_dir)
    words = ['red', 'blue', 'green', 'cat', 'dog', 'bird', 'left', 'right']
    manifest = synthesize_corpus(words, args.output / 'corpus', engine, variants=1, forced_split='train')
    rows = load_manifest(manifest)
    model, report = train_model(rows, [], args.output / 'train', steps=args.steps, batch_size=8,
                               threads=args.threads, seed=17, config=ModelConfig(hidden=64, layers=1, conv_channels=48, dropout=0),
                               augment=False, evaluate_every=100, learning_rate=0.003, phone_weight=0)
    changed = []
    for i, text in enumerate(words):
        audio, _ = engine.synthesize(text, speed=1.12)
        path = (args.output / 'changed_speed' / f'{i}.wav').resolve()
        write_audio(path, audio)
        changed.append({'id': f'speed-{i}', 'audio': str(path), 'text': text, 'split': 'train', 'engine': engine.name})
    speed_result = evaluate(model, changed, FeatureCache(), torch.device('cpu'))
    restored, state = load_model(args.output / 'train' / 'recognizer.pt')
    reloaded = evaluate(restored, rows, FeatureCache(), torch.device('cpu'))
    if reloaded['predictions'] != report['final_train']['predictions']:
        raise AssertionError('Predictions changed after checkpoint reload')
    _, resumed = train_model(rows, [], args.output / 'resume', steps=2, batch_size=8,
                             threads=args.threads, resume=args.output / 'train' / 'last.pt',
                             augment=False, evaluate_every=2, learning_rate=0.003, phone_weight=0)
    train_pass = report['final_train']['cer'] < 0.15 and report['final_train']['cer'] < report['initial_train_subset']['cer']
    result = {'status': 'passed' if train_pass else 'failed', 'engine': engine.fingerprint,
              'train_examples': len(rows), 'steps': args.steps, 'seconds': time.perf_counter()-started,
              'initial_train_cer': report['initial_train_subset']['cer'],
              'final_train_cer': report['final_train']['cer'], 'final_train_wer': report['final_train']['wer'],
              'final_train_exact_match': report['final_train']['exact_match'],
              'same_words_changed_speed': speed_result, 'train_predictions': reloaded['predictions'],
              'checkpoint_reload_identical': True, 'resume_total_steps': resumed['total_steps'],
              'parameters': report['model']['parameters'],
              'inference_checkpoint_bytes': (args.output / 'train' / 'recognizer.pt').stat().st_size,
              'scope': 'Eight-word training-set overfit and speed diagnostic. NOT held-out-text or human-speech accuracy.',
              'paradee_executed': args.engine == 'paradee', 'python': report['python'],
              'torch': report['torch_version'], 'threads': args.threads}
    atomic_json(args.output / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if train_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
