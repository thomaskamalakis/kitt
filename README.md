# KITT

A desktop app where you talk to **K.I.T.T.** from *Knight Rider*: type or speak a sentence, and a
KITT-style voice answers while the classic red voice-box bars animate in sync, on a dashboard
modelled after the car's.

- **Echo** mode repeats what you type, in KITT's voice.
- **Chat** mode (default) sends it to an LLM behind an OpenAI-compatible endpoint (e.g. LiteLLM)
  and speaks the answer. The model can look up the **weather** and **search the web**.
- **Mic** lets you speak to KITT (English); speech is transcribed locally.
- **Tune** lets you shape the voice, pick the Piper voice and configure the LLM.

Everything except the LLM and the two lookup services runs locally.

## Requirements

- Linux with PipeWire (`pw-play`/`pw-record`) or ALSA (`aplay`/`arecord`)
- Python 3.10+ with system **PyQt5** and **numpy** (e.g. `sudo apt install python3-pyqt5 python3-numpy`)
- Internet for the first-run model downloads, and for Chat

## Install and run

    ./setup.sh                  # creates .venv (with system site packages), installs requirements
    .venv/bin/python app.py

Voice models (Piper, from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)) are
downloaded on first use into `voices/`. The speech model (faster-whisper `base.en`, ~140 MB) is
downloaded the first time you press **Mic**.

## Usage

- **Speak** or Enter sends the prompt. **Up/Down** recalls your last 20 prompts.
- **Echo/Chat** toggles between repeating and asking the LLM.
- **Mic**: speak in English, pause when done; press again to cancel.
- **Tune → Replay** re-speaks the last answer with the current settings, so you can adjust by ear.

### Chat (LiteLLM / Qwen)

Open **Tune → LLM**, enter the base URL (`https://<host>/v1`) and your API key, then pick a model
from the list (**Refresh** fetches it from the server). Settings are saved to `llm.json`, which is
gitignored and readable only by you. Environment variables are used as a fallback:

    export LITELLM_BASE_URL=https://<litellm-host>/v1
    export LITELLM_MODEL=<model-alias>
    export LITELLM_API_KEY=<key>

The prompt asks for short plain-text answers and turns off Qwen3 "thinking"; any model served
through the OpenAI chat-completions API with tool calling should work.

### Internet tools

In Chat mode the model may call **get_weather** (Open-Meteo, no key) and **web_search**
(DuckDuckGo through `ddgs`, no key). Place names and search text are sent to those services.

### Voice tuning

The voice is Piper TTS followed by a numpy effect chain (`kitt_effect` in `tts.py`): pitch/speed,
pace, ring modulation, a short comb resonance, a matching EQ, band-pass and soft clipping. The
defaults and your saved choices live in `voice.json`; the Tune sliders edit them. The `eq` curve was
measured against a reference clip of the original voice; it is just a list of dB gains per 1/3-octave
band, so you can reset it to `[]` for a neutral sound.

## Files

| File | Purpose |
|---|---|
| `app.py` | main window and dashboard UI |
| `voicebox.py` | animated voice-box bars |
| `tts.py` | Piper voices and the KITT effect chain |
| `stt.py` | microphone recording and local transcription |
| `llm.py`, `tools.py` | LLM client with tool calling; weather and web search |
| `tune.py` | Tune panel |

## Disclaimer

Unofficial fan project. *Knight Rider* and K.I.T.T. belong to their respective owners; this
project is not affiliated with or endorsed by them. It does not include any audio from the show.

## License

[MIT](LICENSE)
