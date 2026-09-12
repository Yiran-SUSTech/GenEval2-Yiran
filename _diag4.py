import base64
import json
import os
import winreg

import requests

PROXY = {"https": "http://127.0.0.1:7890", "http": "http://127.0.0.1:7890"}
key = os.environ.get("OPENROUTER_API_KEY", "")
if not key:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as r:
        key, _ = winreg.QueryValueEx(r, "OPENROUTER_API_KEY")
headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}

payloads = [
    {"model": "bytedance-seed/seedream-4.5", "prompt": "a green backpack and a pig",
     "aspect_ratio": "1:1", "resolution": "1K"},
    {"model": "bytedance-seed/seedream-4.5", "prompt": "a green backpack and a pig"},
]

for payload in payloads:
    try:
        r = requests.post("https://openrouter.ai/api/v1/images", headers=headers,
                          proxies=PROXY, json=payload, timeout=180)
        print(f"HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code == 200:
            d = r.json()
            print("usage:", json.dumps(d.get("usage")))
            item = (d.get("data") or [{}])[0]
            print("response fields:", list(item.keys()))
            if item.get("b64_json"):
                img = base64.b64decode(item["b64_json"])
                out = r"D:\NewMetric\GenEval2\geneval2_gemini31_flash\seedream45_test.png"
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "wb") as f:
                    f.write(img)
                print(f"SAVED {out} ({len(img)} bytes)")
            break
        if r.status_code == 400 and len(payload) > 2:
            print("-- retrying without aspect_ratio/resolution --")
            continue
        break
    except Exception as e:
        print("EXC:", type(e).__name__, str(e)[:200])
        break
