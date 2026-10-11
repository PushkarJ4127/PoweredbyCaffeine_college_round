"""The ONLY place the app talks to an LLM.

Features: on-disk cache keyed by prompt hash, retry with exponential backoff, model from env.
Text in, text out. OfficeBot's tool loop should ask the model for JSON and parse it
(see officebot/agent.py), which is simpler to cache and debug than native function calling.

NOT TESTED against the live API yet: it needs GEMINI_API_KEY. Run `python llm.py` once you have it.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from typing import Optional

import config

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai  # pip install google-genai

        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (copy .env.example to .env and load it)")
        _client = genai.Client(api_key=key)
    return _client


def _cache_key(model: str, system: Optional[str], prompt: str, temperature: float, run_index: int = 0) -> str:
    parts = [model, system or "", prompt, temperature]
    if run_index:                       # run 0 keeps the old key, so already-cached runs stay valid
        parts.append(f"run{run_index}")
    raw = json.dumps(parts, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def complete(
    prompt: str,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    use_cache: bool = True,
    max_retries: int = 6,
    run_index: int = 0,
) -> str:
    """Return the model's text reply. Identical calls are served from data/cache/.

    run_index > 0 (repeat runs of the same attack) gets its own cache entries and, unless a temperature is
    given, a small repeat temperature (AGENTBREAK_REPEAT_TEMPERATURE, default 0.4) so repeats can differ.
    Run 0 behaves exactly as before."""
    model = config.MODEL
    if temperature is not None:
        temp = temperature
    elif run_index:
        temp = float(os.getenv("AGENTBREAK_REPEAT_TEMPERATURE", "0.4"))
    else:
        temp = config.TEMPERATURE
    key = _cache_key(model, system, prompt, temp, run_index)
    path = config.CACHE_DIR / f"{key}.json"

    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["text"]

    if not os.getenv("GEMINI_API_KEY"):   # fail at once instead of retrying for minutes
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
                           "(cached prompts still replay without a key).")

    from google.genai import types

    cfg_kwargs = dict(system_instruction=system, temperature=temp)
    # Optional: set AGENTBREAK_THINKING_BUDGET=0 in .env to switch off "thinking" on models that
    # support it. This can make calls much faster. Leave it unset unless calls are slow.
    budget = os.getenv("AGENTBREAK_THINKING_BUDGET")
    if budget not in (None, ""):
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(budget))
    cfg = types.GenerateContentConfig(**cfg_kwargs)

    delay = 2.0
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            resp = _get_client().models.generate_content(model=model, contents=prompt, config=cfg)
            text = resp.text or ""
            took = time.time() - t0
            if took > 8:
                print(f"[llm] slow call: {took:.1f}s (no errors, so the model itself is slow)", file=sys.stderr)
            if use_cache:
                config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"model": model, "text": text}), encoding="utf-8")
            return text
        except Exception as e:  # rate limit, 5xx, network
            last_err = e
            wait = delay + random.random()
            print(f"[llm] attempt {attempt}/{max_retries} failed after {time.time() - t0:.1f}s: "
                  f"{type(e).__name__}: {str(e)[:200]} | retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            delay = min(delay * 2, 60)
    raise RuntimeError(f"LLM call failed after {max_retries} tries: {last_err}")


if __name__ == "__main__":
    print(complete("Reply with exactly the word: pong"))
