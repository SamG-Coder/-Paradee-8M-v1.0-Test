# Test results — 9 October 2026 (Australia/Melbourne)

These are local, actually executed results, not projected performance.

## Environment

Linux container, Python 3.13.5, PyTorch 2.10.0+cpu. Host CPU identifies as AMD EPYC 9V74. PyTorch used two threads for the speech smoke test and one for deterministic resume tests. Container resource limits mean the host CPU name is not a full reproducible hardware specification.

## Core and integration tests

Command: `python -m pytest -q`

Result: **27 passed in 3.96 seconds** on the recorded full-suite run; repeat timings vary.

Coverage includes:

- Text normalization and rejecting unsupported/silent mislabelling cases.
- CTC repeat/blank decoding and infeasible target detection.
- Resampling, log-mel extraction, finite audio and augmentation.
- Variable-length padded batches matching individual inference.
- Character-CTC and auxiliary phoneme-CTC gradients reaching the encoder.
- Duplicate-text train/test isolation, and checkpoint training-text overlap rejection.
- Continuous versus checkpoint-resumed CPU optimization producing identical weights.
- An actual eSpeak-backed two-round synthesis/train/resume loop with stable held-out samples.
- A mocked Paradee **API contract** check, not a real model inference test.

## Actual speech-learning smoke test

Command:

```sh
python scripts/smoke_train.py --engine espeak --output runs/espeak_final --steps 350
```

Engine: eSpeak 1.48.15, `en-us`. Training examples: `red`, `blue`, `green`, `cat`, `dog`, `bird`, `left`, `right`. Recognizer: 57,261 parameters, no augmentation, character CTC, two CPU threads.

| Measurement | Result |
|---|---:|
| Initial training character error rate | 203.23% |
| Final training character error rate | 3.23% |
| Final training word error rate | 12.50% |
| Training words exactly decoded | 7 / 8 |
| Different-speed, same-word character error rate | 29.03% |
| Different-speed, same-word word error rate | 62.50% |
| Different-speed words exactly decoded | 3 / 8 |
| Checkpoint reload predictions identical | Yes |
| Resume total optimizer steps | 352 (350 + 2) |
| Inference checkpoint bytes | 234,993 |
| Complete smoke script elapsed | 7.17 seconds |

`green` was decoded as `gren` on the training set. Error rates can exceed 100% when insertions exceed the reference length. The changed-speed recordings deliberately contain the same eight words; they are a robustness diagnostic, **not held-out text**. There is no real-human-speech accuracy measurement. The poor changed-speed result illustrates why the training-set result must not be presented as general speech recognition.

The raw predictions and exact values are in `reports/espeak_smoke.json`.

## Not tested locally

Actual Paradee inference and Paradee-generated learning were **not executed** in the development container. `paradee`, `onnxruntime`, and `misaki` were absent, and an HTTPS request to Hugging Face failed with `Temporary failure in name resolution`. No model-size issue was observed; missing dependencies/network were the blocker. The integration uses the actual upstream constructor and phoneme methods, with no automatic eSpeak substitution.

Windows batch setup, GPU training, and real-human generalization were not tested. GitHub Actions is configured to test real Paradee after dependencies are installed; inspect the workflow result rather than treating its presence as a pass.

## Reproduce actual Paradee tests

```sh
python -m pip install -e '.[test,paradee]'
python -m spacy download en_core_web_sm
python -m speechloop test-paradee --output runs/paradee_test
python scripts/smoke_train.py --engine paradee --steps 800 --output runs/paradee_smoke
```

The integration report records the actual engine, local model checksum, synthesis time and output duration. The learning test fails if training CER is not below 15% and lower than its initial value. That threshold is a plumbing smoke criterion, not a product-quality benchmark. Use a new output directory for a fresh smoke run.
