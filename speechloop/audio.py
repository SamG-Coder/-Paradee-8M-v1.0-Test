"""16 kHz, 40-bin log-mel features without torchaudio or a downloaded frontend."""
from __future__ import annotations
from functools import lru_cache
from math import gcd
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
N_MELS = 40


def canonical_audio(audio: np.ndarray, rate: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or len(x) == 0 or rate <= 0 or not np.isfinite(x).all():
        raise ValueError("Expected finite, non-empty audio and a positive sample rate")
    if rate != SAMPLE_RATE:
        divisor = gcd(rate, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
    if np.max(np.abs(x)) < 1e-7:
        raise ValueError("Silent audio cannot train recognition")
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    return x


def read_audio(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    return canonical_audio(x, sr)


def write_audio(path: Path, audio: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, rate, subtype="PCM_16")


@lru_cache(maxsize=1)
def mel_filter() -> torch.Tensor:
    low, high = 2595 * np.log10(1 + 50 / 700), 2595 * np.log10(1 + 7600 / 700)
    hz = 700 * (10 ** (np.linspace(low, high, N_MELS + 2) / 2595) - 1)
    bins = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    filters = np.zeros((N_MELS, len(bins)), dtype=np.float32)
    for m in range(N_MELS):
        filters[m] = np.maximum(0, np.minimum((bins - hz[m]) / (hz[m + 1] - hz[m]),
                                              (hz[m + 2] - bins) / (hz[m + 2] - hz[m + 1])))
    return torch.from_numpy(filters)


def features(audio: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(audio, dtype=np.float32).copy())
    if x.ndim != 1 or len(x) < N_FFT or not torch.isfinite(x).all():
        raise ValueError("Use canonical_audio before feature extraction")
    spec = torch.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                      window=torch.hann_window(N_FFT), center=True, return_complex=True)
    mel = torch.log(torch.clamp(mel_filter() @ spec.abs().square(), min=1e-8)).T
    # Per-utterance statistics do not inspect other splits; evaluation is identical.
    return ((mel - mel.mean(0, keepdim=True)) / mel.std(0, keepdim=True).clamp_min(0.1)).contiguous()


def augment_audio(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Preserves the transcript: gain, noise, modest echoes; no word cropping."""
    x = audio.copy() * float(rng.uniform(0.75, 1.2))
    if rng.random() < 0.8:
        snr = float(rng.uniform(18, 35))
        rms = np.sqrt(np.mean(x * x) + 1e-12)
        x += rng.normal(0, rms / 10 ** (snr / 20), x.shape).astype(np.float32)
    if rng.random() < 0.3:
        delay = int(rng.uniform(0.01, 0.04) * SAMPLE_RATE)
        if len(x) > delay:
            x[delay:] += float(rng.uniform(0.04, 0.12)) * x[:-delay].copy()
    return np.clip(x, -1, 1).astype(np.float32)


def augment_features(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    x = x.clone()
    if rng.random() < 0.8:
        width = int(rng.integers(1, 5))
        start = int(rng.integers(0, x.shape[1] - width + 1))
        x[:, start:start + width] = 0
    if rng.random() < 0.5 and len(x) > 20:
        width = int(rng.integers(1, min(7, len(x) // 10) + 1))
        start = int(rng.integers(0, len(x) - width + 1))
        x[start:start + width] = 0
    return x
