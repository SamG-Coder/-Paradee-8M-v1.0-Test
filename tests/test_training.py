"""Optimization/integration checks, separate from speech quality claims."""
from pathlib import Path
import shutil
import numpy as np
import pytest
import torch
from speechloop.audio import write_audio
from speechloop.cli import main
from speechloop.curriculum import generated_texts
from speechloop.data import load_manifest, synthesize_corpus, read_jsonl, atomic_jsonl
from speechloop.loop import run_loop
from speechloop.model import ModelConfig, TinyRecognizer
from speechloop.train import train_model, load_model, _phone_loss
from speechloop.tts import EspeakEngine, ParadeeEngine


def toy_rows(tmp_path):
    rows = []
    for i, word in enumerate(['red', 'cat']):
        p = tmp_path / f'{word}.wav'
        write_audio(p, (np.sin(np.arange(8000) * (.06 + .01 * i)) * .2).astype(np.float32))
        rows.append({'id': word, 'audio': str(p), 'text': word, 'split': 'train'})
    return rows


def test_auxiliary_phoneme_head_receives_gradients():
    torch.set_num_threads(1)
    model = TinyRecognizer(ModelConfig(hidden=16, layers=1, conv_channels=16, phoneme_classes=8))
    _, lengths, phones = model(torch.randn(2, 80, 40), torch.tensor([80, 61]))
    loss = _phone_loss(phones, [{'phoneme_ids': [1, 2, 2, 3]}, {'phoneme_ids': [4, 5]}],
                       lengths, torch.nn.CTCLoss(), torch.device('cpu'))
    loss.backward()
    assert model.phoneme_head.weight.grad.abs().sum() > 0
    assert model.conv[0].weight.grad.abs().sum() > 0


def test_resume_matches_uninterrupted_training(tmp_path):
    rows = toy_rows(tmp_path)
    options = dict(batch_size=2, threads=1, seed=17, augment=True, evaluate_every=3,
                   config=ModelConfig(hidden=16, layers=2, conv_channels=16, dropout=.1))
    full, _ = train_model(rows, [], tmp_path / 'full', steps=6, **options)
    train_model(rows, [], tmp_path / 'part', steps=3, **options)
    resumed, report = train_model(rows, [], tmp_path / 'continued', steps=3,
                                resume=tmp_path / 'part' / 'last.pt', **options)
    assert report['total_steps'] == 6
    for key, weight in full.state_dict().items():
        assert torch.equal(weight, resumed.state_dict()[key]), key


def test_evaluation_checks_checkpoint_training_text(tmp_path):
    rows = toy_rows(tmp_path)
    train_model(rows, [], tmp_path / 'run', steps=1, batch_size=2, threads=1,
                config=ModelConfig(hidden=16, layers=1, conv_channels=16))
    atomic_jsonl(tmp_path / 'external.jsonl', [dict(r, split='test') for r in rows])
    args = ['evaluate', '--checkpoint', str(tmp_path / 'run' / 'recognizer.pt'),
            '--manifest', str(tmp_path / 'external.jsonl'), '--output', str(tmp_path / 'eval.json')]
    assert main(args) == 1
    assert main(args + ['--allow-text-overlap']) == 0
    with pytest.raises(ValueError, match='previously learned'):
        train_model([dict(rows[1], text='dog')], [dict(rows[0], split='validation')],
                    tmp_path / 'invalid', steps=1, resume=tmp_path / 'run' / 'last.pt')
    with pytest.raises(ValueError, match='inference-only'):
        train_model(rows, [], tmp_path / 'inference_resume', steps=1,
                    resume=tmp_path / 'run' / 'recognizer.pt')


def test_paradee_threads_validation():
    with pytest.raises(ValueError, match='threads'):
        ParadeeEngine(threads=0)


@pytest.mark.integration
@pytest.mark.skipif(not (shutil.which('espeak-ng') or shutil.which('espeak')), reason='eSpeak executable absent')
def test_real_speech_loop_resume_and_split_isolation(tmp_path):
    engine = EspeakEngine()
    options = dict(engine=engine, root=tmp_path, texts=generated_texts(60),
                   fresh_per_round=10, steps_per_round=2, hidden=16, batch_size=2, threads=1)
    first = run_loop(rounds=1, **options)
    before = load_manifest(tmp_path / 'corpus' / 'manifest.jsonl')
    held_out = {r['id'] for r in before if r['split'] != 'train'}
    state = run_loop(rounds=2, **options)
    after = load_manifest(tmp_path / 'corpus' / 'manifest.jsonl')
    assert state['completed_rounds'] == 2
    assert {r['id'] for r in after if r['split'] != 'train'} == held_out
    assert len(after) > len(before)
    _, checkpoint = load_model(tmp_path / 'round_001' / 'last.pt')
    assert checkpoint['step'] == 4
    assert run_loop(rounds=2, **options) == state
    assert len(read_jsonl(tmp_path / 'corpus' / 'manifest.jsonl')) == len(after)
    with pytest.raises(ValueError, match='configuration changed'):
        run_loop(rounds=3, **dict(options, fresh_per_round=11))
    with pytest.raises(ValueError, match='Data leakage'):
        synthesize_corpus([before[0]['text']], tmp_path / 'corpus', engine,
                          forced_split='test' if before[0]['split'] == 'train' else 'train')
