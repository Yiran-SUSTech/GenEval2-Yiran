import base64
import json
import os
import re
import sys

import requests

BASE = "https://openrouter.ai/api/v1"
OUTDIR = r"D:\NewMetric\GenEval2\geneval2_gemini31_flash"


def get_key():
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            key, _ = winreg.QueryValueEx(k, "OPENROUTER_API_KEY")
    return key


def sec(title):
    print(f"\n===== {title} =====")


key = get_key()
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

sec("1. Python effective proxy config")
print(json.dumps(requests.utils.getproxies(), indent=2))

sec("2. Exit IPs: generic geo service vs openrouter.ai (Cloudflare trace)")
try:
    g = requests.get("https://ipinfo.io/json", timeout=20).json()
    print(f"ipinfo.io sees      : {g.get('ip')}  ({g.get('country')}, {g.get('city')}, {g.get('org')})")
except Exception as e:
    print("ipinfo.io FAILED:", type(e).__name__, str(e)[:150])
openrouter_exit_ip = None
try:
    t = requests.get("https://openrouter.ai/cdn-cgi/trace", timeout=20).text
    for line in t.splitlines():
        if line.startswith("ip="):
            openrouter_exit_ip = line.split("=", 1)[1]
    if openrouter_exit_ip:
        gg = requests.get(f"https://ipinfo.io/{openrouter_exit_ip}/json", timeout=20).json()
        print(f"openrouter.ai sees : {openrouter_exit_ip}  ({gg.get('country')}, {gg.get('city')}, {gg.get('org')})")
    else:
        print("cdn-cgi/trace ok but no ip= line")
except Exception as e:
    print("cdn-cgi/trace FAILED:", type(e).__name__, str(e)[:150])

sec("3. Key + credits")
balance = None
for path in ("/credits", "/auth/key"):
    try:
        r = requests.get(BASE + path, headers=headers, timeout=20)
        print(f"GET {path} -> {r.status_code}: {r.text[:200]}")
        if r.status_code == 200 and path == "/credits":
            balance = r.json().get("data", {}).get("total_credits")
    except Exception as e:
        print(f"GET {path} EXCEPTION: {type(e).__name__} {str(e)[:150]}")

sec("4. Model availability via models list")
google_text = non_google = None
try:
    models = requests.get(BASE + "/models", timeout=30).json().get("data", [])
    google_text = next((m["id"] for m in models
                        if re.fullmatch(r"google/gemini-[\d.]+-flash", m["id"])), None)
    non_google = next((m["id"] for m in models if m["id"].startswith("deepseek/deepseek-chat")
                       and "reasoner" not in m["id"]), None)
    print(f"google text model  : {google_text}")
    print(f"non-google model   : {non_google}")
except Exception as e:
    print("models list FAILED:", type(e).__name__, str(e)[:150])


def chat_probe(model):
    try:
        r = requests.post(BASE + "/chat/completions", headers=headers,
                          json={"model": model, "messages": [{"role": "user", "content": "hi"}],
                                "max_tokens": 5}, timeout=60)
        msg = r.text[:250].replace("\n", " ")
        return f"HTTP {r.status_code}: {msg}"
    except Exception as e:
        return f"EXCEPTION: {type(e).__name__} {str(e)[:150]}"


sec("5. Chat probes (each costs a fraction of a cent)")
if balance == 0:
    print("SKIPPED: credit balance is 0")
else:
    if google_text:
        print(f"[google  ] {google_text}\n  -> {chat_probe(google_text)}")
    if non_google:
        print(f"[control ] {non_google}\n  -> {chat_probe(non_google)}")

sec("6. The real test: images endpoint (costs ~$0.07)")
if balance == 0:
    print("SKIPPED: credit balance is 0")
else:
    payload = {"model": "google/gemini-3.1-flash-image",
               "prompt": "a green backpack and a pig",
               "aspect_ratio": "1:1", "resolution": "1K"}
    try:
        r = requests.post(BASE + "/images", headers=headers, json=payload, timeout=180)
        print(f"HTTP {r.status_code}: {r.text[:400]}")
        if r.status_code == 200:
            d = r.json()
            print("usage:", d.get("usage"))
            img = base64.b64decode(d["data"][0]["b64_json"])
            dest = os.path.join(OUTDIR, "0.png")
            if os.path.exists(dest):
                dest = os.path.join(OUTDIR, "diag_preview.png")
            os.makedirs(OUTDIR, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(img)
            print(f"SAVED: {dest} ({len(img)} bytes)")
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__} {str(e)[:200]}")

print("\nDONE")
