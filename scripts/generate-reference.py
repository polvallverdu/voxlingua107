"""Generate parity fixtures from the original SpeechBrain (PyTorch) model.

Writes 16 kHz mono PCM16 WAV fixtures to test/fixtures/ and the reference
log-probabilities the PyTorch model produces for each of them to
test/fixtures/reference.json. The Bun tests then check that the ONNX model
reproduces these numbers.

Usage:
    uv run --python 3.12 --with speechbrain==1.0.3 --with torch==2.6.0 \
        --with torchaudio==2.6.0 --with soundfile --with numpy --with requests --with "huggingface_hub<0.26" \
        python scripts/generate-reference.py
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier

UPSTREAM = "speechbrain/lang-id-voxlingua107-ecapa"
UPSTREAM_REVISION = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"
SR = 16000

root = Path(__file__).resolve().parent.parent
fixtures = root / "test" / "fixtures"
fixtures.mkdir(parents=True, exist_ok=True)

clf = EncoderClassifier.from_hparams(
    source=UPSTREAM,
    revision=UPSTREAM_REVISION,
    savedir=str(root / ".cache" / "speechbrain"),
)
clf.eval()

# --- Build fixtures -------------------------------------------------------
# The upstream model card's example clip (Thai, 44.1 kHz). Resample it once
# here with torchaudio (what SpeechBrain's classify_file uses) so the JS side
# and the PyTorch side see the exact same 16 kHz samples.
th, th_sr = sf.read(fixtures / "udhr_th.wav", dtype="float32")
th = torchaudio.functional.resample(torch.from_numpy(th), th_sr, SR).numpy()

rng = np.random.default_rng(1234)
clips = {
    "th_full.wav": th,
    "th_3s.wav": th[: 3 * SR],
    "th_1s.wav": th[SR : 2 * SR],
    "noise_4s.wav": (rng.standard_normal(4 * SR) * 0.05).astype(np.float32),
}

reference = {
    "upstream": UPSTREAM,
    "upstreamRevision": UPSTREAM_REVISION,
    "speechbrain": __import__("speechbrain").__version__,
    "torch": torch.__version__,
    "labels": [clf.hparams.label_encoder.ind2lab[i] for i in range(107)],
    "clips": {},
}

for name, audio in clips.items():
    sf.write(fixtures / name, audio, SR, subtype="PCM_16")
    # Read back so the reference is computed on the PCM16-quantized samples,
    # exactly what the tests will load.
    wav, _ = sf.read(fixtures / name, dtype="float32")
    with torch.no_grad():
        out_prob, score, index, label = clf.classify_batch(torch.from_numpy(wav)[None])
    reference["clips"][name] = {
        "label": label[0],
        "score": float(score[0]),
        "logprobs": [float(x) for x in out_prob[0]],
    }
    print(f"{name:14s} {label[0]:22s} p={float(score[0].exp()):.4f}")

(fixtures / "reference.json").write_text(json.dumps(reference, indent=1) + "\n")
print(f"wrote {fixtures / 'reference.json'}")
