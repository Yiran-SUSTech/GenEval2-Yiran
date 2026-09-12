import json
import os
import winreg

import requests

BASE = "https://openrouter.ai/api/v1"


def get_key():
    k = os.environ.get("OPENROUTER_API_KEY", "")
    if not k:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as r:
            k, _ = winreg.QueryValueEx(r, "OPENROUTER_API_KEY")
    return k


headers = {"Authorization": "Bearer " + get_key(), "Content-Type": "application/json"}

models = requests.get(BASE + "/models", timeout=30).json().get("data", [])
ids = [m["id"] for m in models]


def pick(prefix, pattern=None, exclude=()):
    for mid in ids:
        if mid.startswith(prefix) and (pattern is None or pattern in mid) \
                and not any(x in mid for x in exclude):
            return mid
    return None


targets = [
    ("google text flash", pick("google/", "flash", ("image", "tts", "thinking", "lite"))),
    ("google gemma", pick("google/gemma")),
    ("anthropic haiku", pick("anthropic/", "haiku")),
    ("mistral control", pick("mistralai/")),
]

print(f"{'label':20s} {'model':45s} status | detail")
for label, model in targets:
    if not model:
        print(f"{label:20s} {'<none found>':45s}")
        continue
    try:
        r = requests.post(BASE + "/chat/completions", headers=headers,
                          json={"model": model, "messages": [{"role": "user", "content": "hi"}],
                                "max_tokens": 5}, timeout=60)
        if r.status_code == 200:
            print(f"{label:20s} {model:45s} 200 OK")
        else:
            try:
                err = r.json().get("error", {})
                meta = err.get("metadata") or {}
                prev = meta.get("previous_errors") or []
                provs = [(p.get("provider"), p.get("code")) for p in prev]
                print(f"{label:20s} {model:45s} {r.status_code} | {str(err.get('message'))[:80]} | "
                      f"provider_name={meta.get('provider_name')} tried={provs}")
            except Exception:
                print(f"{label:20s} {model:45s} {r.status_code} | {r.text[:120]}")
    except Exception as e:
        print(f"{label:20s} {model:45s} EXC {type(e).__name__} {str(e)[:100]}")

print("\nDONE")
