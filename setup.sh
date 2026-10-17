#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
mkdir -p voices
B=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high
for f in en_US-ryan-high.onnx en_US-ryan-high.onnx.json; do
  [ -f voices/$f ] || curl -fL -o voices/$f $B/$f
done
