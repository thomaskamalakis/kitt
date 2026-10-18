"""Speech-to-text: record from the default microphone until the speaker pauses,
then transcribe (English) locally with faster-whisper."""
import shutil
import subprocess

import numpy as np

RATE = 16000
CHUNK = RATE // 10          # 100 ms
MODEL = "base.en"
SILENCE_END = 1.0           # seconds of quiet after speech that end the phrase
WAIT_SPEECH = 8.0           # give up if nobody speaks for this long
MAX_PHRASE = 20.0

_model = None


def available():
    return shutil.which("pw-record") or shutil.which("arecord")


def _recorder():
    if shutil.which("pw-record"):
        cmd = ["pw-record", "--rate", str(RATE), "--channels", "1", "--format", "s16", "-"]
    else:
        cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def record(stop=lambda: False, on_level=None):
    """Return float32 mono samples of one spoken phrase, or None if nobody spoke."""
    proc = _recorder()
    chunks, quiet, spoke, heard_for = [], 0.0, False, 0.0
    ambient = []
    threshold = 0.02
    try:
        while not stop():
            data = proc.stdout.read(CHUNK * 2)
            if len(data) < CHUNK * 2:
                break
            x = np.frombuffer(data, np.int16).astype(np.float32) / 32768
            rms = float(np.sqrt(np.mean(x * x)))
            if on_level:
                on_level(rms)
            if len(ambient) < 4:                      # first 0.4 s: measure room noise
                ambient.append(rms)
                threshold = max(0.02, 3 * float(np.mean(ambient)))
                chunks.append(x)
                continue
            chunks.append(x)
            heard_for += 0.1
            if rms > threshold:
                spoke, quiet = True, 0.0
            elif spoke:
                quiet += 0.1
                if quiet >= SILENCE_END:
                    break
            if not spoke:
                chunks = chunks[-8:]                  # keep only a short lead-in
                if heard_for > WAIT_SPEECH:
                    return None
            elif heard_for > MAX_PHRASE:
                break
    finally:
        proc.kill()
        proc.wait()
    return np.concatenate(chunks) if spoke and chunks else None


def transcribe(samples):
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL, device="cpu", compute_type="int8")
    segs, _ = _model.transcribe(samples, language="en", beam_size=1, vad_filter=True,
                                condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segs).strip()


def model_loaded():
    return _model is not None
