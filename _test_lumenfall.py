"""Verify Lumenfall API access for Nano Banana 2 (Gemini 3.1 Flash Image).

Steps:
  1. Register at https://lumenfall.ai/create-account  (free credits, no card)
  2. Create an API key (starts with lmnfl_) in the dashboard
  3. Run:  python _test_lumenfall.py lmnfl_your_key_here
"""
import base64
import json
import sys

import requests

BASE = "https://api.lumenfall.ai/openai/v1"

key = sys.argv[1] if len(sys.argv) > 1 else ""
if not key:
    sys.exit("usage: python _test_lumenfall.py lmnfl_your_key_here")
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

print("===== 1. Key check + model discovery =====")
r = requests.get(f"{BASE}/models", headers=headers, timeout=30)
print(f"GET /models -> {r.status_code}")
if r.status_code != 200:
    sys.exit(f"Key/endpoint problem: {r.text[:300]}")
ids = [m.get("id") for m in r.json().get("data", [])]
print(f"total models visible: {len(ids)}")
nb = [i for i in ids if "gemini-3.1-flash-image" in i or "nano-banana" in i.lower()]
print(f"Nano Banana 2 candidates: {nb or '(none found)'}")
if not nb:
    sys.exit("No Nano Banana 2 model ID visible for this account")

def route_pref(i):
    # fal / replicate routes avoid Google's regional checks on the upstream call
    if i.startswith("fal/"):
        return 0
    if i.startswith("replicate/"):
        return 1
    return 2

model = min(nb, key=route_pref)
print(f"selected model for test: {model}")

print("\n===== 2. One test generation (~$0.07 of free credits) =====")
payload = {"model": model, "prompt": "a green backpack and a pig", "size": "1024x1024"}
try:
    r = requests.post(f"{BASE}/images/generations", headers=headers,
                      json=payload, timeout=180)
except Exception as e:
    sys.exit(f"network error: {type(e).__name__}: {str(e)[:200]}")
print(f"HTTP {r.status_code}")
if r.status_code != 200:
    print(r.text[:500])
    sys.exit("generation failed — see error above (403/region or param issue)")

d = r.json()
if d.get("usage"):
    print("usage:", json.dumps(d.get("usage")))
item = (d.get("data") or [{}])[0]
if item.get("b64_json"):
    img = base64.b64decode(item["b64_json"])
    print("response contains b64 image")
elif item.get("url"):
    print("response contains URL, downloading...")
    img = requests.get(item["url"], timeout=120).content
else:
    sys.exit(f"unexpected response shape: {list(item.keys())}")

out = r"D:\NewMetric\GenEval2\geneval2_gemini31_flash\lumenfall_test.png"
with open(out, "wb") as f:
    f.write(img)
from PIL import Image
print(f"SAVED {out} — {Image.open(out).size}")
print("\nLUMENFALL WORKS — ready to wire up the sampling backend")
