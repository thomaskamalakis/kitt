import json
import os
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QPlainTextEdit, QApplication, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QVBoxLayout, QWidget, QGridLayout)

import llm
import tts
from tune import TunePanel
from voicebox import VoiceBox


def play_wav(path):
    for cmd in (["pw-play", path], ["aplay", "-q", path]):
        try:
            return subprocess.Popen(cmd)
        except FileNotFoundError:
            continue
    return None


HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.json")
HISTORY_MAX = 20
CELL = 110   # vertical pitch between pill centres
PILL_W, PILL_H = 88, 54


class HistoryEdit(QLineEdit):
    """QLineEdit with shell-like Up/Down recall of the last HISTORY_MAX sentences."""

    def __init__(self):
        super().__init__()
        try:
            with open(HISTORY_FILE) as f:
                self.history = json.load(f)[-HISTORY_MAX:]
        except (OSError, ValueError):
            self.history = []
        self._idx = len(self.history)
        self._draft = ""

    def remember(self, text):
        if text and (not self.history or self.history[-1] != text):
            self.history = (self.history + [text])[-HISTORY_MAX:]
            try:
                with open(HISTORY_FILE, "w") as f:
                    json.dump(self.history, f)
            except OSError:
                pass
        self._idx = len(self.history)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Up and self.history:
            if self._idx == len(self.history):
                self._draft = self.text()
            self._idx = max(0, self._idx - 1)
            self.setText(self.history[self._idx])
        elif e.key() == Qt.Key_Down and self._idx < len(self.history):
            self._idx += 1
            self.setText(self._draft if self._idx == len(self.history) else self.history[self._idx])
        else:
            super().keyPressEvent(e)


def kitt_button(text, bg, pill=False, fg="#000", cruise=False, width=None):
    b = QPushButton(text)
    b.setCheckable(True)
    b.setChecked(True)
    b.setFocusPolicy(Qt.NoFocus)
    radius = PILL_H // 2 if pill else 14 if cruise else 3
    size = 28 if pill else 17 if cruise else 18
    b.setFixedHeight(PILL_H if pill else 50 if cruise else 34)
    if pill:
        b.setFixedWidth(PILL_W)
    elif cruise:
        b.setFixedWidth(120)
    elif width:
        b.setFixedWidth(width)
    b.setStyleSheet(
        f"QPushButton{{background:{bg};color:{fg};border:none;border-radius:{radius}px;"
        f"font:bold {size}px sans-serif;padding:2px 4px}}"
        "QPushButton:!checked{background:#1c1c1c;color:#444}")
    return b


class StatusBox(QPlainTextEdit):
    """Fixed-height message area (scrolls if long) so the layout never shifts."""

    def __init__(self, lines=4):
        super().__init__()
        self.setReadOnly(True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setFrameShape(QPlainTextEdit.NoFrame)
        self.setFixedHeight(self.fontMetrics().lineSpacing() * lines + 8)

    def setText(self, text):
        self.setPlainText(text)


class Worker(QThread):
    ready = pyqtSignal(object, int)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)
    reply = pyqtSignal(str)

    def __init__(self, text, chat=False, history=()):
        super().__init__()
        self.text, self.chat, self.history = text, chat, list(history)

    def run(self):
        try:
            tts.progress = self.progress.emit
            text = self.text
            if self.chat:
                text = llm.ask(self.history, self.text)
                self.reply.emit(text)
            self.ready.emit(*tts.synthesize(text))
        except Exception as e:
            self.failed.emit(str(e))


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("K.I.T.T.")
        self.setStyleSheet(
            "QWidget{background:#000;color:#ff2a0a;font:14px monospace}"
            "QLineEdit{border:1px solid #6a0a05;padding:6px}")
        self.box = VoiceBox()
        self.edit = HistoryEdit()
        self.edit.setPlaceholderText("Say something to KITT...")
        self.btn = QPushButton("Speak")
        self.status = StatusBox()
        self.status.setText("Ready")
        self.history = []      # chat messages [{role, content}, ...]
        self.last_spoken = ""  # what Tune > Replay repeats
        row = QHBoxLayout()
        row.addWidget(self.edit)
        row.addWidget(self.btn)
        self.chat_btn = QPushButton("Echo")
        self.chat_btn.setCheckable(True)
        self.chat_btn.setFocusPolicy(Qt.NoFocus)
        self.chat_btn.setToolTip("Echo: repeat the sentence. Chat: ask Qwen and speak its answer")
        self.chat_btn.setStyleSheet("QPushButton{border:1px solid #ff2a0a;padding:6px 14px}"
                                    "QPushButton:checked{background:#ff2a0a;color:#000}")
        self.chat_btn.toggled.connect(lambda on: self.chat_btn.setText("Chat" if on else "Echo"))
        row.addWidget(self.chat_btn)
        self.tune_btn = QPushButton("Tune")
        self.tune_btn.setFocusPolicy(Qt.NoFocus)
        self.tune_btn.setStyleSheet("QPushButton{border:1px solid #ff2a0a;padding:6px 14px}")
        self.tune_btn.clicked.connect(self.open_tune)
        row.addWidget(self.tune_btn)
        self.btn.setStyleSheet("QPushButton{border:1px solid #ff2a0a;padding:6px 14px}QPushButton:disabled{color:#6a0a05;border-color:#6a0a05}")
        RED, ORG, YEL, BLU = "#e8200a", "#ff8c3c", "#f5f000", "#2a50ff"
        top = QHBoxLayout()
        topcol = QVBoxLayout()
        for t, c in (("POWER", RED), ("MIN RPM", ORG), ("FUEL ON", YEL), ("IGNITORS", YEL)):
            topcol.addWidget(kitt_button(t, c, width=150), alignment=Qt.AlignHCenter)
        top.addStretch(1)
        top.addLayout(topcol)
        top.addStretch(1)
        mid = QGridLayout()
        mid.setHorizontalSpacing(6)
        mid.setVerticalSpacing(0)
        for r in range(4):
            mid.setRowMinimumHeight(r, CELL)   # every pill sits in an equal cell
        mid.addWidget(kitt_button("AIR", YEL, True), 0, 0, alignment=Qt.AlignCenter)
        mid.addWidget(kitt_button("OIL", YEL, True), 1, 0, alignment=Qt.AlignCenter)
        mid.addWidget(kitt_button("P1", "#e8400a", True), 2, 0, alignment=Qt.AlignCenter)
        mid.addWidget(kitt_button("P2", "#e8400a", True), 3, 0, alignment=Qt.AlignCenter)
        self.box.setFixedSize(150, 2 * CELL)   # spans the AIR and OIL cells
        mid.addWidget(self.box, 0, 1, 2, 1)
        mid.addWidget(kitt_button("S1", YEL, True), 0, 2, alignment=Qt.AlignCenter)
        mid.addWidget(kitt_button("S2", YEL, True), 1, 2, alignment=Qt.AlignCenter)
        mid.addWidget(kitt_button("P3", "#e8400a", True), 2, 2, alignment=Qt.AlignCenter)
        mid.addWidget(kitt_button("P4", "#e8400a", True), 3, 2, alignment=Qt.AlignCenter)
        bottom = QVBoxLayout()   # sits between P1/P2 and P3/P4
        # AUTO centred on P1, NORMAL between, PURSUIT centred on P2
        bottom.setContentsMargins(0, CELL // 2 - 25, 0, CELL // 2 - 25)
        bottom.setSpacing(5)
        bottom.addWidget(kitt_button("AUTO\nCRUISE", ORG, cruise=True), alignment=Qt.AlignHCenter)
        bottom.addWidget(kitt_button("NORMAL\nCRUISE", YEL, cruise=True), alignment=Qt.AlignHCenter)
        bottom.addWidget(kitt_button("PURSUIT", BLU, fg="#fff", cruise=True), alignment=Qt.AlignHCenter)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addSpacing(8)   # small gap between the top buttons and the bars
        mid.addLayout(bottom, 2, 1, 2, 1)
        midrow = QHBoxLayout()           # keep the pill/bars group compact and centred
        midrow.addStretch(1)
        midrow.addLayout(mid)
        midrow.addStretch(1)
        lay.addLayout(midrow)
        lay.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(self.status)
        self.edit.returnPressed.connect(self.speak)
        self.btn.clicked.connect(self.speak)
        self.worker = None
        self.proc = None

    def speak(self):
        text = self.edit.text().strip()
        if not text or not self.btn.isEnabled():
            return
        self.edit.remember(text)
        self._run(text, self.chat_btn.isChecked())

    def replay(self):
        if self.last_spoken and self.btn.isEnabled():
            self._run(self.last_spoken)

    def open_tune(self):
        if not hasattr(self, "tune"):
            self.tune = TunePanel(self.replay, self)
        self.tune.show()
        self.tune.raise_()

    def _run(self, text, chat=False):
        self.last_spoken = text
        self._reply = None
        self.btn.setEnabled(False)
        self.edit.setEnabled(False)
        self.status.setText("Downloading voice..." if tts.needs_download() else "Thinking...")
        self.worker = Worker(text, chat, self.history)
        self.worker.reply.connect(lambda r: self._got_reply(text, r))
        self.worker.ready.connect(self.play)
        self.worker.failed.connect(self.error)
        self.worker.progress.connect(lambda p: self.status.setText(f"Downloading voice... {p}%"))
        self.worker.start()

    def _got_reply(self, question, reply):
        self._reply = reply
        self.last_spoken = reply
        self.history += [{"role": "user", "content": question},
                         {"role": "assistant", "content": reply}]
        del self.history[:-2 * llm.MAX_TURNS]

    def play(self, samples, sr):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((samples * 32767).astype(np.int16).tobytes())
        self.proc = play_wav(path)
        if self.proc is None:
            return self.error("No audio player (pw-play/aplay) found")
        self.status.setText("KITT: " + self._reply if self._reply else "Speaking...")
        self.box.start(samples, sr, time.monotonic() + 0.05)
        self._path = path
        from PyQt5.QtCore import QTimer
        self._poll = QTimer(self)
        self._poll.timeout.connect(self._check_done)
        self._poll.start(100)

    def _check_done(self):
        if self.proc.poll() is not None:
            self._poll.stop()
            self.box.stop()
            os.unlink(self._path)
            self.finish("Ready")

    def error(self, msg):
        self.finish("Error: " + msg)

    def finish(self, msg):
        self.status.setText(msg)
        self.btn.setEnabled(True)
        self.edit.setEnabled(True)
        self.edit.clear()
        self.edit.setFocus()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.resize(400, 770)
    win.show()
    sys.exit(app.exec_())
