"""Non-modal voice tuning panel: sliders for the tts effect parameters."""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel,
                             QPushButton, QSlider, QVBoxLayout)

import tts

# key, label, min, max, step
SLIDERS = [
    ("speed", "Pitch / speed", 0.70, 1.10, 0.01),
    ("ring_mix", "Metallic (ring mod)", 0.0, 0.5, 0.01),
    ("ring_hz", "Ring freq (Hz)", 30, 300, 1),
    ("comb_gain", "Resonance", 0.0, 0.8, 0.01),
    ("comb_ms", "Resonance delay (ms)", 1.0, 12.0, 0.1),
    ("band_lo", "Low cut (Hz)", 50, 600, 10),
    ("band_hi", "High cut (Hz)", 2000, 10000, 100),
    ("drive", "Drive", 0.5, 4.0, 0.05),
]


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
        buttons = QHBoxLayout()
        for text, fn in (("Replay", self.replay), ("Reset", self._reset), ("Save", self._save)):
            b = QPushButton(text)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(fn)
            buttons.addWidget(b)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addLayout(buttons)
        self.resize(420, 360)
        self._load(tts.PARAMS)

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
