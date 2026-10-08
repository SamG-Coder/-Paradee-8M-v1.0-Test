"""Speech engines. Selecting Paradee NEVER silently falls back to another voice."""
from __future__ import annotations
import io
import json
import shutil
import subprocess
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import numpy as np
import soundfile as sf
from .audio import canonical_audio


def _version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


class ParadeeEngine:
    name = "paradee"

    def __init__(self, threads: int = 1, model_dir: Path | None = None):
        if threads < 1:
            raise ValueError("threads must be positive")
        try:
            from paradee import Paradee
        except ImportError as e:
            raise RuntimeError("Paradee is not installed. Run SETUP_WINDOWS.bat or pip install -e '.[paradee]'.") from e
        kwargs = {"threads": threads}
        # Prefer the model already committed at the repository root. A missing
        # config is downloaded by Paradee itself from its pinned v1.0 release.
        if model_dir is None and Path("paradee_int8.onnx").is_file():
            model_dir = Path.cwd()
        if model_dir is not None:
            model_dir = Path(model_dir).resolve()
            model = model_dir / "paradee_int8.onnx"
            config = model_dir / "config.json"
            if not model.is_file():
                raise FileNotFoundError(f"Missing local Paradee model: {model}")
            kwargs["model_path"] = str(model)
            if config.is_file():
                kwargs["config_path"] = str(config)
        self.model = Paradee(**kwargs)
        self.phoneme_vocab = dict(self.model.vocab)
        self.fingerprint = {"engine": self.name, "package_version": _version("paradee"),
                            "model": "sahilmahendrakar/Paradee-8M-v1.0", "revision": "v1.0",
                            "model_dir": str(model_dir) if model_dir else None}
        if model_dir:
            import hashlib
            self.fingerprint["model_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()

    def synthesize(self, text: str, speed: float = 1.0) -> tuple[np.ndarray, list[int]]:
        if not 0.6 <= speed <= 1.5:
            raise ValueError("Use speeds from 0.6 to 1.5")
        phonemes = self.model.phonemize(text)
        if len(phonemes) > 510:
            raise ValueError("Input exceeds Paradee's phoneme limit; shorten the sentence")
        unknown = set(phonemes) - set(self.phoneme_vocab)
        if unknown:
            raise ValueError(f"Paradee returned unsupported phonemes: {sorted(unknown)}")
        # Pad/blank ID 0 is never a CTC target. All remaining teacher IDs are kept.
        ids = [self.phoneme_vocab[p] for p in phonemes if self.phoneme_vocab[p] != 0]
        if not ids:
            raise ValueError("Paradee produced no phoneme targets")
        audio = self.model.generate_from_phonemes(phonemes, speed=speed)
        return canonical_audio(audio, 24000), ids


class EspeakEngine:
    """Explicit local smoke-test engine; not a substitute for Paradee evaluation."""
    name = "espeak"
    phoneme_vocab: dict = {}

    def __init__(self, voice: str = "en-us", **_):
        self.exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.exe:
            raise RuntimeError("eSpeak is not installed. Use --engine paradee, or install eSpeak NG for local smoke tests.")
        self.voice = voice
        v = subprocess.run([self.exe, "--version"], capture_output=True, text=True, timeout=15)
        self.fingerprint = {"engine": self.name, "voice": voice, "version": v.stdout.strip()}

    def synthesize(self, text: str, speed: float = 1.0) -> tuple[np.ndarray, list[int]]:
        if not 0.6 <= speed <= 1.5:
            raise ValueError("Use speeds from 0.6 to 1.5")
        p = subprocess.run([self.exe, "--stdout", "-v", self.voice, "-s", str(round(160 * speed)),
                            "--stdin"], input=text.encode("utf-8"), capture_output=True, timeout=60, check=True)
        audio, sr = sf.read(io.BytesIO(p.stdout), dtype="float32")
        return canonical_audio(audio, sr), []


def make_engine(name: str, threads: int = 1, model_dir: Path | None = None, voice: str = "en-us"):
    if name == "paradee":
        return ParadeeEngine(threads=threads, model_dir=model_dir)
    if name == "espeak":
        return EspeakEngine(voice=voice)
    raise ValueError(f"Unknown engine: {name}")
