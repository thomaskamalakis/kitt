"""Ask the Qwen model behind the Department's LiteLLM proxy (OpenAI-compatible API)."""
import json
import os
import re

import requests

SYSTEM_PROMPT = (
    "You are K.I.T.T., the Knight Industries Two Thousand: a calm, polite, precise and "
    "slightly witty artificial intelligence, the talking car from Knight Rider. "
    "Answer in English in one to three short sentences. Your reply is spoken aloud, so use "
    "plain text only: no markdown, no lists, no emoji."
)
MAX_TURNS = 10


class LLMError(Exception):
    pass


CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm.json")   # gitignored: holds the API key
ENV = {"base_url": "LITELLM_BASE_URL", "model": "LITELLM_MODEL", "api_key": "LITELLM_API_KEY"}

# Saved settings win; environment variables are the fallback.
CONFIG = {k: os.environ.get(v, "") for k, v in ENV.items()}
try:
    with open(CONFIG_FILE) as f:
        CONFIG.update({k: str(v) for k, v in json.load(f).items() if k in ENV and v})
except (OSError, ValueError):
    pass


def save_config():
    fd = os.open(CONFIG_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(CONFIG, f, indent=2)


def _env(key):
    value = CONFIG.get(key, "").strip()
    if not value:
        raise LLMError(f"LLM {key.replace('_', ' ')} is not set (open Tune)")
    return value


def clean(reply):
    """Make a model reply suitable for speech."""
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    reply = re.sub(r"<think>.*", "", reply, flags=re.S)           # unterminated block
    reply = re.sub(r"[*_`#>~|]+", "", reply)
    reply = re.sub(r"^\s*[-•]\s+", "", reply, flags=re.M)
    reply = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️]", "", reply)
    return re.sub(r"\s+", " ", reply).strip()


def list_models(base_url=None, api_key=None):
    """Return the model ids the proxy offers (GET /models, OpenAI-compatible)."""
    base_url = (base_url if base_url is not None else CONFIG.get("base_url", "")).strip()
    api_key = (api_key if api_key is not None else CONFIG.get("api_key", "")).strip()
    if not base_url or not api_key:
        raise LLMError("Base URL and API key are needed first")
    try:
        r = requests.get(base_url.rstrip("/") + "/models",
                         headers={"Authorization": "Bearer " + api_key}, timeout=10)
    except requests.RequestException as e:
        raise LLMError(f"Cannot reach LLM server: {type(e).__name__}")
    if r.status_code == 401:
        raise LLMError("LLM rejected the API key (401)")
    if not r.ok:
        raise LLMError(f"LLM error {r.status_code}")
    try:
        return sorted({m["id"] for m in r.json()["data"]})
    except (ValueError, KeyError, TypeError):
        raise LLMError("Unexpected response from LLM server")


def ask(history, text):
    """Send the conversation so far plus `text`; return the cleaned reply."""
    url = _env("base_url").rstrip("/") + "/chat/completions"
    body = {
        "model": _env("model"),
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     *history[-2 * MAX_TURNS:],
                     {"role": "user", "content": text}],
        "temperature": 0.7,
        "max_tokens": 400,
        # Qwen3 "thinking" can eat the whole token budget and leave the reply empty;
        # spoken answers don't need it.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Authorization": "Bearer " + _env("api_key")}
    try:
        r = requests.post(url, json=body, headers=headers, timeout=60)
        if r.status_code == 400:      # some models/providers reject the extra parameter
            body.pop("chat_template_kwargs")
            r = requests.post(url, json=body, headers=headers, timeout=60)
    except requests.Timeout:
        raise LLMError("LLM request timed out")
    except requests.RequestException as e:
        raise LLMError(f"Cannot reach LLM server: {type(e).__name__}")
    if r.status_code == 401:
        raise LLMError("LLM rejected the API key (401)")
    if not r.ok:
        raise LLMError(f"LLM error {r.status_code}: {r.text[:120]}")
    try:
        choice = r.json()["choices"][0]
        reply = clean(choice["message"]["content"] or "")
    except (ValueError, KeyError, IndexError, TypeError):
        raise LLMError("Unexpected response from LLM server")
    if not reply:
        why = " (ran out of tokens while thinking)" if choice.get("finish_reason") == "length" else ""
        raise LLMError("LLM returned an empty reply" + why)
    return reply
