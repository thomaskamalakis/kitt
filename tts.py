"""Text -> KITT-style speech: Piper TTS followed by a small numpy effect chain."""
import json
import os
import socket
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(HERE, "voices")
HF_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang_short}/{lang}/{name}/{quality}/{lang}-{name}-{quality}.onnx"

# (label, lang, name, quality) - voices are downloaded on first use
VOICES = [
    ("Ryan (US, high)", "en_US", "ryan", "high"),
    ("Ryan (US, medium)", "en_US", "ryan", "medium"),
    ("Joe (US)", "en_US", "joe", "medium"),
    ("HFC male (US)", "en_US", "hfc_male", "medium"),
    ("John (US)", "en_US", "john", "medium"),
    ("Bryce (US)", "en_US", "bryce", "medium"),
    ("Kusal (US)", "en_US", "kusal", "medium"),
    ("Norman (US)", "en_US", "norman", "medium"),
    ("Lessac (US, high)", "en_US", "lessac", "high"),
    ("Danny (US, low)", "en_US", "danny", "low"),
    ("Alan (UK)", "en_GB", "alan", "medium"),
    ("Northern English male (UK)", "en_GB", "northern_english_male", "medium"),
]

# Effect parameters; defaults can be overridden by voice.json (written by the tuning panel)
DEFAULTS = {
    "speed": 0.90,       # <1 = deeper and slower
    "length": 1.0,       # Piper length_scale: >1 = slower speech, same pitch
    "ring_mix": 0.04,    # ring modulation amount (metallic)
    "ring_hz": 90.0,
    "comb_ms": 4.0,      # short comb delay for a "voice box" resonance
    "comb_gain": 0.12,
    "presence": 0.0,     # dB boost of the upper mids (above ~1 kHz) for a nasal, "speaker" tone
    "band_lo": 120.0,    # band-pass edges (Hz)
    "band_hi": 7000.0,
    "drive": 1.1,        # soft-clip drive
}
PARAMS_FILE = os.path.join(HERE, "voice.json")
PARAMS = dict(DEFAULTS)
voice_index = 0   # index into VOICES, saved in voice.json
try:
    with open(PARAMS_FILE) as f:
        _saved = json.load(f)
    PARAMS.update({k: v for k, v in _saved.items() if k in DEFAULTS})
    voice_index = min(max(int(_saved.get("voice", 0)), 0), len(VOICES) - 1)
except (OSError, ValueError):
    pass


def save_params():
    with open(PARAMS_FILE, "w") as f:
        json.dump({**PARAMS, "voice": voice_index}, f, indent=2)


_voice = None


def _model_path(index=None):
    _, lang, name, quality = VOICES[voice_index if index is None else index]
    return os.path.join(VOICE_DIR, f"{lang}-{name}-{quality}.onnx")


def needs_download():
    return not os.path.exists(_model_path())


socket.setdefaulttimeout(30)   # fail instead of hanging on a stalled download
progress = None   # optional callback(percent:int), called while a voice downloads


def _download(index):
    _, lang, name, quality = VOICES[index]
    url = HF_URL.format(lang_short=lang.split("_")[0], lang=lang, name=name, quality=quality)
    os.makedirs(VOICE_DIR, exist_ok=True)
    for suffix in ("", ".json"):
        dest = _model_path(index) + suffix
        tmp = dest + ".part"
        def hook(blocks, size, total):
            if progress and total > 0:
                progress(min(100, int(blocks * size * 100 / total)))
        urllib.request.urlretrieve(url + suffix, tmp, hook)
        os.replace(tmp, dest)


def set_voice(index):
    global voice_index, _voice, _raw_cache
    voice_index, _voice, _raw_cache = index, None, (None, None, None)


def _load():
    global _voice
    if _voice is None:
        if needs_download():
            _download(voice_index)
        from piper import PiperVoice
        _voice = PiperVoice.load(_model_path())
    return _voice


def _bandpass(x, sr, lo, hi):
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    mask = np.clip((freqs - lo) / 50.0, 0, 1) * np.clip((hi - freqs) / 500.0, 0, 1)
    return np.fft.irfft(spec * mask, len(x))


def _presence(x, sr, db):
    if not db:
        return x
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    shelf = np.clip((freqs - 500.0) / 700.0, 0, 1)
    gain = 1 + (10 ** (db / 20) - 1) * shelf
    return np.fft.irfft(np.fft.rfft(x) * gain, len(x))


def kitt_effect(x, sr, p=None):
    p = p or PARAMS
    n = int(len(x) / p["speed"])
    x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    t = np.arange(len(x)) / sr
    x = x * (1 - p["ring_mix"] + p["ring_mix"] * np.sin(2 * np.pi * p["ring_hz"] * t))
    d = max(1, int(sr * p["comb_ms"] / 1000))
    y = x.copy()
    y[d:] += p["comb_gain"] * x[:-d]
    y = _presence(y, sr, p["presence"])
    y = _bandpass(y, sr, p["band_lo"], p["band_hi"])
    y = np.tanh(p["drive"] * y / (np.max(np.abs(y)) + 1e-9))
    return (0.9 * y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


_raw_cache = (None, None, None)   # (text, samples, sample_rate) of the last Piper output


def synthesize(text):
    """Return (float32 samples in [-1, 1], sample_rate). Piper output of the
    last text is cached so effect changes can be re-applied instantly."""
    global _raw_cache
    key = (text, PARAMS["length"])
    if _raw_cache[0] != key:
        from piper import SynthesisConfig
        cfg = SynthesisConfig(length_scale=PARAMS["length"])
        chunks = list(_load().synthesize(text, cfg))
        x = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float64)
        _raw_cache = (key, x, chunks[0].sample_rate)
    _, x, sr = _raw_cache
    return kitt_effect(x, sr), sr
