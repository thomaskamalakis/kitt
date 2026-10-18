"""Non-modal voice tuning panel: sliders for the tts effect parameters."""
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QSlider, QVBoxLayout)

import llm
import tts

# key, label, min, max, step
SLIDERS = [
    ("speed", "Pitch / speed", 0.70, 1.10, 0.01),
    ("length", "Pace (slower →)", 0.6, 1.6, 0.01),
    ("ring_mix", "Metallic (ring mod)", 0.0, 0.5, 0.01),
    ("ring_hz", "Ring freq (Hz)", 30, 300, 1),
    ("comb_gain", "Resonance", 0.0, 0.8, 0.01),
    ("comb_ms", "Resonance delay (ms)", 1.0, 12.0, 0.1),
    ("presence", "Presence (dB)", 0, 15, 0.5),
    ("band_lo", "Low cut (Hz)", 50, 600, 10),
    ("band_hi", "High cut (Hz)", 2000, 10000, 100),
    ("drive", "Drive", 0.5, 4.0, 0.05),
]


class ModelFetcher(QThread):
    done = pyqtSignal(list, str)   # (model ids, error message)

    def __init__(self, base_url, api_key):
        super().__init__()
        self.base_url, self.api_key = base_url, api_key

    def run(self):
        try:
            self.done.emit(llm.list_models(self.base_url, self.api_key), "")
        except llm.LLMError as e:
            self.done.emit([], str(e))


class TunePanel(QDialog):
    def __init__(self, replay, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Voice tuning")
        self.replay = replay
        self.rows = {}
        form = QFormLayout()
        self.voice_box = QComboBox()
        self.voice_box.addItems([v[0] for v in tts.VOICES])
        self.voice_box.setCurrentIndex(tts.voice_index)
        self.voice_box.currentIndexChanged.connect(tts.set_voice)   # new voice loads on next Replay
        form.addRow("Voice", self.voice_box)
        for key, label, lo, hi, step in SLIDERS:
            s = QSlider(Qt.Horizontal)
            s.setRange(0, int(round((hi - lo) / step)))
            val = QLabel()
            val.setMinimumWidth(50)
            s.valueChanged.connect(lambda v, k=key, lo=lo, st=step, vl=val: self._changed(k, lo + v * st, vl))
            row = QHBoxLayout()
            row.addWidget(s, 1)
            row.addWidget(val)
            form.addRow(label, row)
            self.rows[key] = (s, val, lo, step)
        form.addRow(QLabel("— LLM (LiteLLM / Qwen) —"))
        self.llm_fields = {}
        for key, label, hint in (("base_url", "Base URL", "https://host/v1"),
                                 ("api_key", "API key", "")):
            e = QLineEdit(llm.CONFIG.get(key, ""))
            e.setPlaceholderText(hint)
            if key == "api_key":
                e.setEchoMode(QLineEdit.Password)
            e.textChanged.connect(lambda v, k=key: llm.CONFIG.__setitem__(k, v.strip()))
            e.editingFinished.connect(self.fetch_models)   # list models once URL + key are filled in
            form.addRow(label, e)
            self.llm_fields[key] = e
        self.model_box = QComboBox()
        self.model_box.setEditable(True)   # still typeable if the proxy can't list models
        self.model_box.setCurrentText(llm.CONFIG.get("model", ""))
        self.model_box.currentTextChanged.connect(lambda v: llm.CONFIG.__setitem__("model", v.strip()))
        refresh = QPushButton("Refresh")
        refresh.setFocusPolicy(Qt.NoFocus)
        refresh.clicked.connect(self.fetch_models)
        mrow = QHBoxLayout()
        mrow.addWidget(self.model_box, 1)
        mrow.addWidget(refresh)
        form.addRow("Model", mrow)
        self.model_status = QLabel("")
        form.addRow("", self.model_status)
        self.fetcher = None
        buttons = QHBoxLayout()
        for text, fn in (("Replay", self.replay), ("Reset", self._reset), ("Save", self._save)):
            b = QPushButton(text)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(fn)
            buttons.addWidget(b)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addLayout(buttons)
        self.resize(460, 500)
        self._load(tts.PARAMS)
        self.fetch_models()

    def fetch_models(self):
        base, key = llm.CONFIG.get("base_url", ""), llm.CONFIG.get("api_key", "")
        if not base or not key or (self.fetcher and self.fetcher.isRunning()):
            return
        self.model_status.setText("Loading models...")
        self.fetcher = ModelFetcher(base, key)
        self.fetcher.done.connect(self._models)
        self.fetcher.start()

    def _models(self, models, error):
        if error:
            self.model_status.setText(error)
            return
        current = llm.CONFIG.get("model", "")
        self.model_box.blockSignals(True)
        self.model_box.clear()
        self.model_box.addItems(models)
        self.model_box.setCurrentText(current if current in models else "")
        self.model_box.blockSignals(False)
        if current not in models and models:
            self.model_box.setCurrentIndex(0)   # triggers the CONFIG update
        self.model_status.setText(f"{len(models)} models available")

    def _changed(self, key, value, label):
        value = round(value, 3)
        tts.PARAMS[key] = value
        label.setText(f"{value:g}")

    def _load(self, params):
        for key, (s, val, lo, step) in self.rows.items():
            s.setValue(int(round((params[key] - lo) / step)))
            val.setText(f"{params[key]:g}")

    def _reset(self):
        tts.PARAMS.update(tts.DEFAULTS)
        self._load(tts.PARAMS)

    def _save(self):
        tts.save_params()
        llm.save_config()
        if self.parent():
            self.parent().status.setText("Settings saved")
        self.accept()   # close the panel
