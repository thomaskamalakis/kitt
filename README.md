# KITT voice

Type a sentence, KITT repeats it in a KITT-like voice with the orange voice-box animation.

    ./setup.sh                 # venv + piper-tts + voice model (needs system PyQt5, numpy)
    .venv/bin/python app.py

Voice tuning constants are at the top of `tts.py`.

## Chat with Qwen (LiteLLM)

The **Echo/Chat** toggle next to *Speak* switches between repeating your sentence and
sending it to the Qwen model behind a LiteLLM proxy (OpenAI-compatible API), then speaking the answer.
Configure it in **Tune → LLM** (saved to the gitignored `llm.json`, readable only by you), or with environment variables as a fallback:

    export LITELLM_BASE_URL=https://<litellm-host>/v1
    export LITELLM_MODEL=<qwen-model-alias>
    export LITELLM_API_KEY=<key>
    .venv/bin/python app.py
