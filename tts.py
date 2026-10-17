"""Text -> KITT-style speech: Piper TTS followed by a small numpy effect chain."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "voices", "en_US-ryan-high.onnx")

# Effect parameters; defaults can be overridden by voice.json (written by the tuning panel)
DEFAULTS = {
    "speed": 0.90,       # <1 = deeper and slower
    "ring_mix": 0.04,    # ring modulation amount (metallic)
    "ring_hz": 90.0,
    "comb_ms": 4.0,      # short comb delay for a "voice box" resonance
    "comb_gain": 0.12,
    "band_lo": 120.0,    # band-pass edges (Hz)
    "band_hi": 7000.0,
    "drive": 1.1,        # soft-clip drive
}
PARAMS_FILE = os.path.join(HERE, "voice.json")
PARAMS = dict(DEFAULTS)
try:
    with open(PARAMS_FILE) as f:
        PARAMS.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
except (OSError, ValueError):
    pass


def save_params():
    with open(PARAMS_FILE, "w") as f:
        json.dump(PARAMS, f, indent=2)


_voice = None


def _load():
    global _voice
    if _voice is None:
        from piper import PiperVoice
        _voice = PiperVoice.load(MODEL)
    return _voice


def _bandpass(x, sr, lo, hi):
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    mask = np.clip((freqs - lo) / 50.0, 0, 1) * np.clip((hi - freqs) / 500.0, 0, 1)
    return np.fft.irfft(spec * mask, len(x))


def kitt_effect(x, sr, p=None):
    p = p or PARAMS
    n = int(len(x) / p["speed"])
    x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    t = np.arange(len(x)) / sr
    x = x * (1 - p["ring_mix"] + p["ring_mix"] * np.sin(2 * np.pi * p["ring_hz"] * t))
    d = max(1, int(sr * p["comb_ms"] / 1000))
    y = x.copy()
    y[d:] += p["comb_gain"] * x[:-d]
    y = _bandpass(y, sr, p["band_lo"], p["band_hi"])
    y = np.tanh(p["drive"] * y / (np.max(np.abs(y)) + 1e-9))
    return (0.9 * y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


_raw_cache = (None, None, None)   # (text, samples, sample_rate) of the last Piper output


def synthesize(text):
    """Return (float32 samples in [-1, 1], sample_rate). Piper output of the
    last text is cached so effect changes can be re-applied instantly."""
    global _raw_cache
    if _raw_cache[0] != text:
        chunks = list(_load().synthesize(text))
        x = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float64)
        _raw_cache = (text, x, chunks[0].sample_rate)
    _, x, sr = _raw_cache
    return kitt_effect(x, sr), sr
