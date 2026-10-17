"""The orange KITT voice-modulator: three columns of segments driven by speech."""
import time

import numpy as np
from PyQt5.QtCore import QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QWidget

COLUMNS = 3
ROWS = (19, 23, 19)  # segments per column (odd: lit outward from the centre row)
BANDS = ((80, 400), (400, 1500), (1500, 6000))   # low / mid / high per column
WINDOW = 1024
BAR_SPACING = 0.7   # gap between bars, as a fraction of bar width


class VoiceBox(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 280)
        self._samples = None
        self._sr = 22050
        self._t0 = 0.0
        self._levels = [0.0] * COLUMNS
        self._peak = 1e-6
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def start(self, samples, sr, t0=None):
        self._samples, self._sr = samples, sr
        self._t0 = time.monotonic() if t0 is None else t0
        # normalise the level against the loudest part of this utterance
        self._peak = max((self._band_levels(i).sum() for i in range(0, len(samples), WINDOW // 2)),
                         default=1.0) + 1e-9
        self._timer.start()

    def stop(self):
        # keep the timer running so the bars decay to zero instead of freezing
        self._samples = None

    def _band_levels(self, pos):
        seg = self._samples[pos:pos + WINDOW]
        if len(seg) < WINDOW:
            seg = np.pad(seg, (0, WINDOW - len(seg)))
        mag = np.abs(np.fft.rfft(seg * np.hanning(WINDOW)))
        f = np.fft.rfftfreq(WINDOW, 1.0 / self._sr)
        return np.array([np.sqrt(np.mean(mag[(f >= lo) & (f < hi)] ** 2)) for lo, hi in BANDS])

    def _tick(self):
        target = [0.0] * COLUMNS
        if self._samples is not None:
            pos = int((time.monotonic() - self._t0) * self._sr)
            if pos >= len(self._samples):
                self.stop()
            else:
                level = float(np.clip(self._band_levels(pos).sum() / self._peak, 0, 1))
                target = [level] * COLUMNS     # all three columns move in sync
        for i, v in enumerate(target):
            v = 0.0 if v < 0.06 else v         # noise gate
            cur = self._levels[i]
            cur = v if v > cur else cur * 0.8  # fast attack, slower decay
            self._levels[i] = 0.0 if cur < 0.04 else cur
        if self._samples is None and not any(self._levels):
            self._timer.stop()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(0, 0, 0))
        p.setPen(Qt.NoPen)
        w, h = self.width(), self.height()
        gap = 3
        col_w = w * 0.19                        # bar width
        col_gap = col_w * BAR_SPACING           # space between bars
        left = (w - (COLUMNS * col_w + (COLUMNS - 1) * col_gap)) / 2   # group centred
        # segment height is set by the tallest column so all share the same grid
        seg_h = (h - (max(ROWS) - 1) * gap) / max(ROWS)
        for c in range(COLUMNS):
            n = ROWS[c]
            mid = n // 2
            x = left + c * (col_w + col_gap)
            top = (h - (n * seg_h + (n - 1) * gap)) / 2
            lit = int(round(self._levels[c] * (mid + 1)))
            for r in range(n):
                d = abs(r - mid)
                if d < lit:
                    t = 1.0 - d / (mid + 1.0)          # brightest at the centre
                    col = QColor(int(70 + 185 * t), 0, 0)
                else:
                    col = QColor(45, 0, 0)
                p.setBrush(col)
                p.drawRect(QRectF(x, top + r * (seg_h + gap), col_w, seg_h))
