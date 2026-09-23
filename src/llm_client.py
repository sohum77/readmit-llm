"""
llm_client.py
-------------
Provider-agnostic LLM inference module.
"""

import os
import json
import time
import urllib.request
import urllib.error

PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
OLLAMA_HOST_URL = os.environ.get("OLLAMA_HOST_URL", "http://172.20.128.1:11434")

MODELS = {
    "ollama": "llama3.2",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-6",
}

REQUEST_TIMEOUT = 60
MAX_RETRIES = 2
OLLAMA_NUM_PREDICT = 400


def call_llm(prompt: str, temperature: float = 0.0) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 2):
        start = time.time()
        try:
            if PROVIDER == "ollama":
                result = _call_ollama(prompt, temperature)
            elif PROVIDER == "openai":
                result = _call_openai(prompt, temperature)
            elif PROVIDER == "anthropic":
                result = _call_anthropic(prompt, temperature)
            else:
                raise ValueError(f"Unknown provider: {PROVIDER}")
            elapsed = time.time() - start
            print(f"    [{PROVIDER}] call took {elapsed:.1f}s (attempt {attempt})")
            return result
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            print(f"    [{PROVIDER}] attempt {attempt} failed: {e}")
            if attempt <= MAX_RETRIES:
                time.sleep(2 * attempt)
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES + 1} attempts: {last_err}")


def _call_ollama(prompt, temperature):
    payload = json.dumps({
        "model": MODELS["ollama"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
        return json.loads(r.read())["response"]


def _call_openai(prompt, temperature):
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=MODELS["openai"],
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        timeout=REQUEST_TIMEOUT,
    )
    return resp.choices[0].message.content


def _call_anthropic(prompt, temperature):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODELS["anthropic"],
        max_tokens=512,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
        timeout=REQUEST_TIMEOUT,
    )
    return resp.content[0].text