# SpeechLoopLab — Paradee reverse-speech experiment

A runnable base for **text → synthetic voice → learned speech-to-text → error → more training**.
Uses the existing `paradee_int8.onnx` in this repository as a frozen CPU speech generator. A new, small PyTorch recognizer learns from its waveforms and the exact text supplied to the generator. Difficult **training** sentences are replayed; fresh sentences are added between rounds.

**This is not a pretrained speech recognizer and does not reverse Paradee's weights.** The ONNX model stays unchanged. An optional auxiliary head learns Paradee's phoneme IDs alongside characters, sharing the recognizer's audio encoder. It does not share the TTS model's trained layers.

## Start on Windows

Install Python 3.11 and Git, open a terminal in the repository, then:

```bat
SETUP_WINDOWS.bat
.venv\Scripts\python -m speechloop test-paradee
.venv\Scripts\python scripts\smoke_train.py --engine paradee --steps 800 --output runs\paradee_smoke
RUN_LOOP_WINDOWS.bat
```

`test-paradee` writes an actual generated WAV and a signal-validity report. `smoke_train.py` checks whether the recognizer can learn eight spoken training words and reload its checkpoint. The main loop is a separate, deliberately small experiment, not a promise of useful general transcription after three rounds.

The setup script installs CPU PyTorch, the pinned Paradee package, and the English phonemizer assets. **Internet is needed for initial setup.** The ONNX file is only the TTS weights; Python, PyTorch, ONNX Runtime and phonemizer assets make the complete installation much larger than 9 MB. No eSpeak executable is required for the Paradee path; the upstream package uses its own dependencies. Windows setup was not executed in the Linux development environment.

## Linux / macOS

```sh
python3.11 -m venv .venv
# Linux:
. .venv/bin/activate
python -m pip install --upgrade pip
# Optional on Linux: install CPU-only PyTorch first to avoid CUDA packages.
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,paradee]'
python -m spacy download en_core_web_sm
python -m speechloop test-paradee
python -m pytest -q
```

On macOS, omit the CPU-index line and let the ordinary dependency install choose PyTorch. The project defaults to CPU on all platforms.

## The loop

```sh
python -m speechloop loop --engine paradee --generate 120 --rounds 3 --fresh 100 --steps 300 --output runs/loop
```

1. Generate short original sentences, or read supplied plain-text files.
2. Normalize text once; send **that same string** to TTS and use it as the target.
3. Assign text to train/validation/test by a stable hash. Every speed variant stays in that text's split.
4. Generate and cache audio, then train a log-mel → convolution → bidirectional GRU → character-CTC recognizer.
5. Increase sampling of difficult training examples, retaining a uniform-sampling component. Between rounds, synthesize fresh training text and different-speed versions of difficult training text.
6. Select best checkpoints using validation only. Test examples are never used for curriculum selection or optimizer updates.

Rerun the identical command to continue an interrupted run. `--rounds` is the **desired total number of rounds**; use `--rounds 5` to extend a completed three-round run. Keep the same output directory and experiment settings. Each round has a persisted synthesis plan so a partial synthesis retry uses the same sentences. The built-in grammar is intentionally limited: for broader language coverage, add your own text and expand `speechloop/curriculum.py` rather than repeatedly training on a thousand similar phrases.

### Use books or other text

```sh
python -m speechloop loop --engine paradee --generate 0 --text-file examples/sentences.txt --text-file my_book.txt --rounds 3 --output runs/books
```

Only use material you are entitled to process. The project does not autonomously crawl websites. An explicit plain-text downloader records source URL, declared license and checksum:

```sh
python -m speechloop fetch-text "https://YOUR-SOURCE.example/book.txt" --output sources/book.txt --license "public domain or actual license" --rights-confirmed
```

That example URL is a placeholder. The downloader accepts HTTPS `text/plain`, UTF-8, and at most 2 MB per request. It records your rights declaration; it does not verify it. Clean book headers/footers and spell numbers/symbols as spoken words. Unsupported characters or numeric targets are rejected rather than silently producing incorrect labels. Repeated text is deduplicated; sentence chunks are capped at 18 words, with an additional phoneme-length check before synthesis.

### Reuse Paradee's phoneme targets

```sh
python -m speechloop loop --engine paradee --phoneme-weight 0.2 --output runs/phoneme_loop
```

This adds an auxiliary CTC loss over the teacher's phoneme vocabulary. The encoder learns to predict both written characters and the actual phonemes used for synthesis. This is an experiment in additional supervision, **not** weight reuse, invertible TTS, or a differentiable round-trip through the quantized ONNX model.

## Separate generation, training and evaluation

```sh
python -m speechloop generate --engine paradee --generate 200 --variants 2 --output runs/corpus
python -m speechloop train --manifest runs/corpus/manifest.jsonl --run runs/train --steps 1000
python -m speechloop evaluate --checkpoint runs/train/best.pt --manifest runs/corpus/manifest.jsonl --split test --output runs/test_result.json
python -m speechloop transcribe recording.wav --checkpoint runs/train/recognizer.pt
```

Use `last.pt` to resume optimization; `--steps` on the `train` command means **additional steps**. `best.pt` includes the optimizer and is saved only on validation improvement. `recognizer.pt` is the final inference-only checkpoint; it excludes optimizer/RNG state. For multi-round runs, best checkpoints are per-round; a later round does not necessarily improve the global best. Compare validation scores before selecting one. No language model or known-transcript lookup is used by the greedy decoder.

`--device cuda` or `--device auto` can train the recognizer on a supported GPU when an appropriate PyTorch build is installed. Paradee generation remains CPU. CUDA training was not tested here. The bidirectional encoder processes whole clips; it is not a low-latency streaming recognizer.

### Real human speech

Create a CSV with paths relative to the CSV file:

```csv
audio,text,split
recordings/alice.wav,the small bird is near the river,test
recordings/bob.wav,please open the green door,test
```

```sh
python -m speechloop import-human human.csv --output runs/human/manifest.jsonl
python -m speechloop evaluate --checkpoint runs/train/recognizer.pt --manifest runs/human/manifest.jsonl --split test --output runs/human_result.json
```

Use speakers and text not used for training. Checkpoints record training-text hashes, so external evaluation containing the same text is rejected unless `--allow-text-overlap` explicitly requests a **same-text diagnostic**. This guards exact normalized text, not paraphrases, speaker identity, or copied audio under a different path. Speaker-disjoint evaluation is your responsibility.

## What was actually tested

See [the test report](docs/TEST_RESULTS.md) and machine-readable [speech smoke results](reports/espeak_smoke.json).

* **27 passing tests**, including CTC gradients, phoneme-head gradients, padding invariance, manifests, split isolation, checkpoint lineage, exact CPU resume equivalence and an actual two-round eSpeak-backed loop.
* **Eight-word eSpeak speech-training smoke test:** seven words exact; 3.23% training character error after 350 steps. Different-speed versions were substantially worse. This establishes optimization/checkpoint plumbing, not useful ASR quality.
* **Paradee was not executed in the local development environment.** Its dependencies were absent and outbound DNS failed. The adapter contract is tested; actual Paradee inference/learning remains an integration test to run after setup. There is no silent fallback to eSpeak.

The GitHub Actions workflow runs core tests and an actual Paradee integration job, preserving generated samples/reports as artifacts. Its results must be checked in Actions; adding a workflow does not mean it passed.

## Size

`python -m speechloop info` reports the actual configured recognizer size. Defaults: **496,957 parameters**, **1,987,828 FP32 parameter bytes** before checkpoint metadata. The eight-word smoke test uses a smaller **57,261-parameter** model. These sizes exclude frozen Paradee, feature-processing code and all runtime dependencies. Quantized recognizer export is not implemented in this base.

## Important boundaries

Synthetic labels are known input text, **not a guarantee that the TTS actually pronounced every word correctly**. Systematic TTS errors become label noise. Learning a single voice, memorizing eight words, or reducing training loss does not demonstrate general speech recognition. Use held-out sentences, unfamiliar voices, microphones, accents, noise and human recordings before making quality claims. Keep the teacher frozen initially to avoid both networks developing an artificial communication shortcut.

The synthetic generator is the base teacher here. Reusing trainable Paradee internal features or training both directions jointly is a later experiment, not hidden functionality in this starter.

## Upstream and license

Project code: Apache-2.0. The model already committed by the repository owner is Paradee by Sahil Mahendrakar; its upstream license is Apache-2.0. See [THIRD_PARTY.md](THIRD_PARTY.md).

* [Paradee source, pinned integration version](https://github.com/sahilmahendrakar/paradee/tree/9c8b4d7504cbee7e64de2d0341bb690f0b0ab708)
* [Paradee model card and weights](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.0)
* [Upstream inference API](https://github.com/sahilmahendrakar/paradee/blob/9c8b4d7504cbee7e64de2d0341bb690f0b0ab708/paradee/tts.py)
