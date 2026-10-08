# Third-party components

The pre-existing `paradee_int8.onnx` was supplied by this repository's owner and has not been modified by SpeechLoopLab.

Paradee author: Sahil Mahendrakar.
Upstream model: https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.0
Upstream source: https://github.com/sahilmahendrakar/paradee
Upstream declares Apache-2.0; see LICENSE for the license text.
Paradee is distilled from hexgrad/Kokoro-82M; consult the upstream model card for lineage and notices.

The integration dependency is pinned to upstream source commit `9c8b4d7504cbee7e64de2d0341bb690f0b0ab708`; upstream inference fetches model assets from the `v1.0` model revision unless local files are supplied.

Python packages and eSpeak/eSpeak NG retain their respective licenses. This repository does not vendor those packages, executables or their datasets. eSpeak is an optional external test executable invoked as a subprocess, not the default speech engine. The included example text is original project material, not a copied book. Source-text rights remain separate from the code/model licenses.
