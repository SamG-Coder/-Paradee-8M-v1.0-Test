from pathlib import Path
import json
import numpy as np
import pytest
import torch
from speechloop.text import normalize, encode, decode, split_for, text_id
from speechloop.audio import canonical_audio, features, write_audio, read_audio, augment_audio
from speechloop.metrics import score, distance
from speechloop.model import ModelConfig, TinyRecognizer
from speechloop.data import FeatureCache, load_manifest, atomic_jsonl, assert_disjoint
from speechloop.curriculum import generated_texts, sampling_weights
from speechloop.train import train_model, load_model, evaluate
from speechloop.tts import ParadeeEngine

@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)
    torch.manual_seed(17)


def test_text_normalization():
    assert normalize("Hello, WORLD! It's fine.") == "hello world it's fine"
    assert normalize("café—door") == "cafe door"
    assert decode(encode("hello"), collapse=False) == "hello"


@pytest.mark.parametrize("text", ["", "   ", "price $12", "a & b", "hello🙂", "at 12:00"])
def test_unsupported_text_is_not_silently_mislabelled(text):
    with pytest.raises(ValueError):
        normalize(text)


def test_ctc_repeated_letters_and_blanks():
    l = encode("l")[0]
    assert decode([l, l, 0, l]) == "ll"
    assert decode([0, 0]) == ""


def test_metrics():
    assert distance("kitten", "sitting") == 3
    s = score([("one two", "one"), ("red", "red blue")])
    assert s["wer"] == 2 / 3
    assert s["exact_match"] == 0


def test_splits_are_text_level():
    assert split_for("Hello!") == split_for("hello")
    assert text_id("THE CAT.") == text_id("the cat")
    assert {split_for(t) for t in generated_texts(200)} == {"train", "validation", "test"}


def test_curriculum_has_uniform_floor():
    weights = sampling_weights([{"id": "a"}, {"id": "b"}], {"a": 100000})
    assert sum(weights) == pytest.approx(1)
    assert min(weights) >= 0.15


def test_audio_resampling_and_features(tmp_path):
    audio = np.sin(np.arange(24000, dtype=np.float32) * 0.08) * 0.2
    x = canonical_audio(audio, 24000)
    assert len(x) == 16000
    write_audio(tmp_path / "sample.wav", x)
    y = features(read_audio(tmp_path / "sample.wav"))
    assert y.shape == (101, 40)
    assert torch.isfinite(y).all()
    assert np.isfinite(augment_audio(x, np.random.default_rng(17))).all()


@pytest.mark.parametrize("audio", [np.array([]), np.array([np.nan]), np.zeros(1000)])
def test_invalid_audio_rejected(audio):
    with pytest.raises(ValueError):
        canonical_audio(audio, 16000)


def test_padded_batch_matches_single():
    model = TinyRecognizer(ModelConfig(hidden=16, layers=1, conv_channels=16, dropout=0)).eval()
    a, b = torch.randn(1, 37, 40), torch.randn(1, 60, 40)
    batch = torch.cat([torch.nn.functional.pad(a, (0, 0, 0, 23)), b])
    with torch.inference_mode():
        one, lengths, _ = model(a, torch.tensor([37]))
        two, _, _ = model(batch, torch.tensor([37, 60]))
    assert lengths.tolist() == [19]
    assert torch.allclose(one[0], two[0, :19], atol=1e-5)


def test_ctc_gradients_and_auxiliary_head():
    model = TinyRecognizer(ModelConfig(hidden=16, layers=1, conv_channels=16, phoneme_classes=100))
    logits, lengths, phones = model(torch.randn(2, 80, 40), torch.tensor([80, 61]))
    assert phones.shape == (2, 40, 100)
    target = torch.tensor(encode("red") + encode("cat"))
    loss = torch.nn.CTCLoss()(logits.log_softmax(-1).transpose(0, 1), target, lengths, torch.tensor([3, 3]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.conv[0].weight.grad.abs().sum() > 0


def test_manifest_rejects_leakage(tmp_path):
    write_audio(tmp_path / "a.wav", np.sin(np.arange(16000) * 0.08).astype(np.float32) * .2)
    rows = [{"id": "1", "audio": "a.wav", "text": "the cat", "split": "train"},
            {"id": "2", "audio": "a.wav", "text": "THE CAT!", "split": "test"}]
    atomic_jsonl(tmp_path / "manifest.jsonl", rows)
    with pytest.raises(ValueError, match="leak"):
        load_manifest(tmp_path / "manifest.jsonl")


def test_training_rejects_test_examples(tmp_path):
    with pytest.raises(ValueError, match="TRAIN"):
        train_model([{"split": "test", "text": "red"}], [], tmp_path, steps=1)


def test_normalized_overlap_is_rejected():
    with pytest.raises(ValueError, match="leaks"):
        assert_disjoint([{"text": "The cat!"}], [{"text": "the cat"}])


def test_paradee_adapter_contract(monkeypatch):
    # Contract-only test, deliberately NOT labelled as a real Paradee integration.
    import types, sys
    class Stub:
        vocab = {"a": 1, "b": 2, " ": 3}
        def __init__(self, **kwargs):
            assert kwargs["threads"] == 1
        def phonemize(self, text):
            return "ab ba"
        def generate_from_phonemes(self, phonemes, speed=1):
            assert phonemes == "ab ba"
            return (np.sin(np.arange(24000) * .1) * .2).astype(np.float32)
    monkeypatch.setitem(sys.modules, "paradee", types.SimpleNamespace(Paradee=Stub))
    engine = ParadeeEngine()
    audio, phones = engine.synthesize("some words")
    assert len(audio) == 16000 and phones == [1, 2, 3, 2, 1]


def test_ctc_infeasible_targets_raise(tmp_path):
    p = tmp_path / "short.wav"
    write_audio(p, np.sin(np.arange(500) * .1).astype(np.float32) * .2)
    with pytest.raises(ValueError, match="Not enough"):
        FeatureCache().batch([{"audio": str(p), "text": "this sentence is far too long"}])
