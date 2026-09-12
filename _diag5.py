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

payload = {"model": "bytedance-seed/seedream-4.5", "prompt": "seven green croissants",
           "aspect_ratio": "1:1"}
r = requests.post("https://openrouter.ai/api/v1/images", headers=headers,
                  proxies=PROXY, json=payload, timeout=180)
print(f"HTTP {r.status_code}: {r.text[:250]}")
if r.status_code == 200:
    d = r.json()
    print("usage cost:", d.get("usage", {}).get("cost"))
    img = base64.b64decode(d["data"][0]["b64_json"])
    out = r"D:\NewMetric\GenEval2\geneval2_gemini31_flash\seedream45_ar_test.png"
    with open(out, "wb") as f:
        f.write(img)
    from PIL import Image
    print("SAVED, size:", Image.open(out).size)
