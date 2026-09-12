import base64
import json
import os
import winreg

import requests

BASE = "https://openrouter.ai/api/v1"
PROXY = {"https": "http://127.0.0.1:7890", "http": "http://127.0.0.1:7890"}


def get_key():
    k = os.environ.get("OPENROUTER_API_KEY", "")
    if not k:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as r:
            k, _ = winreg.QueryValueEx(r, "OPENROUTER_API_KEY")
    return k


headers = {"Authorization": "Bearer " + get_key(), "Content-Type": "application/json"}

print("===== 1. Image-generation models on OpenRouter =====")
models = []
r = requests.get(BASE + "/images/models", timeout=30, proxies=PROXY)
print("GET /images/models ->", r.status_code)
if r.status_code == 200:
    body = r.json()
    data = body.get("data", body) if isinstance(body, dict) else body
    for m in data:
        if not isinstance(m, dict):
            continue
        mid = m.get("id", "?")
        pr = m.get("pricing", {}) or {}
        print(f"  {mid:55s} in={pr.get('prompt')} out={pr.get('completion')} img={pr.get('image')}")
        models.append(mid)
else:
    print(r.text[:300])

if not models:
    r2 = requests.get(BASE + "/models", timeout=30, proxies=PROXY)
    if r2.status_code == 200:
        for m in r2.json().get("data", []):
            arch = m.get("architecture", {}) or {}
            if "image" in (arch.get("output_modalities") or []):
                print(" ", m.get("id"))
                models.append(m.get("id"))

print("\n===== 2. OpenAI text probe (is OpenAI also blocked?) =====")
try:
    r = requests.post(BASE + "/chat/completions", headers=headers, proxies=PROXY,
                      json={"model": "openai/gpt-4o-mini",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 5}, timeout=60)
    print(f"openai/gpt-4o-mini -> {r.status_code} : {r.text[:150]}")
except Exception as e:
    print("EXC:", type(e).__name__, str(e)[:150])

print("\n===== 3. Seedream image test (costs a few cents) =====")
seed = [m for m in models if "seedream" in (m or "").lower()]
print("seedream models found:", seed)
if seed:
    model = seed[0]
    for payload in ({"model": model, "prompt": "a green backpack and a pig",
                     "aspect_ratio": "1:1", "resolution": "1K"},
                    {"model": model, "prompt": "a green backpack and a pig"}):
        try:
            r = requests.post(BASE + "/images", headers=headers, proxies=PROXY,
                              json=payload, timeout=180)
            print(f"HTTP {r.status_code}: {r.text[:400]}")
            if r.status_code == 200:
                d = r.json()
                print("usage:", d.get("usage"))
                item = (d.get("data") or [{}])[0]
                if item.get("b64_json"):
                    img = base64.b64decode(item["b64_json"])
                    out = r"D:\NewMetric\GenEval2\geneval2_gemini31_flash\seedream_test.png"
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with open(out, "wb") as f:
                        f.write(img)
                    print(f"SAVED {out} ({len(img)} bytes)")
                elif item.get("url"):
                    print("URL:", item["url"][:160])
                break
            elif r.status_code == 400 and payload.get("aspect_ratio"):
                print("-- retrying without aspect_ratio/resolution --")
                continue
            else:
                break
        except Exception as e:
            print("EXC:", type(e).__name__, str(e)[:150])
            break

print("\nDONE")
