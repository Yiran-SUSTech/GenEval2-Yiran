import base64
import importlib.util

spec = importlib.util.spec_from_file_location("sample_images", r"D:\NewMetric\GenEval2\sample_images.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

GEN = mod.GenerationError
FATAL = mod.FatalAPIError
REAL_POST_JSON = mod.post_json


class Args:
    model = "gemini-3.1-flash-image-preview"
    base_url = "https://api.lumenfall.ai/openai/v1"
    size = "1024*1024"
    timeout = 300
    max_retries = 2
    retry_delay = 0.01
    cooldown = None


def run_backend(responses_by_call, downloaded=None):
    """responses_by_call: {call_index: dict returned by post_json, or GenerationError to raise}"""
    calls = []

    def fake_post_json(url, headers, payload, *a, **kw):
        calls.append(dict(payload))
        r = responses_by_call.get(len(calls) - 1)
        if isinstance(r, Exception):
            raise r
        return r

    def fake_download(url, timeout=120, *a, **kw):
        if downloaded is not None and isinstance(downloaded, Exception):
            raise downloaded
        return downloaded or b"pngbytes"

    mod.post_json = fake_post_json
    mod.download_bytes = fake_download
    be = mod.LumenfallBackend(Args(), "lmnfl_test_key")
    return be.generate("a green backpack and a pig"), calls, be


# T1: url response path; size converted from Qwen style "1024*1024" to "1024x1024"
img, calls, be = run_backend({0: {"data": [{"url": "https://cdn/img.png"}]}})
assert img == b"pngbytes"
assert calls == [{"model": Args.model, "prompt": "a green backpack and a pig", "size": "1024x1024"}], calls
print("T1 url path + size 1024*1024 -> 1024x1024 ..... PASS")

# T2: b64_json response path (no download needed)
img, calls, be = run_backend({0: {"data": [{"b64_json": base64.b64encode(b"rawbytes").decode()}]}})
assert img == b"rawbytes" and len(calls) == 1
print("T2 b64_json path, no download ................. PASS")

# T3: 4xx naming size -> drop and retry, then succeed
img, calls, be = run_backend({
    0: GEN('HTTP 400: {"error":{"message":"size is not a supported value"}}'),
    1: {"data": [{"url": "https://cdn/img.png"}]},
})
assert img == b"pngbytes"
assert calls[1] == {"model": Args.model, "prompt": "a green backpack and a pig"}
assert "size" not in be.enabled_fields
print("T3 4xx naming size auto-drops it .............. PASS")

# T4: empty data -> GenerationError
try:
    run_backend({0: {"data": []}})
    raise AssertionError("should have raised")
except GEN:
    print("T4 empty data raises sample failure .......... PASS")

# T5: item without url/b64_json -> GenerationError
try:
    run_backend({0: {"data": [{"revised_prompt": "x"}]}})
    raise AssertionError("should have raised")
except GEN:
    print("T5 no url/b64 key raises sample failure ...... PASS")

# T6: HTTP 402 -> FatalAPIError through real post_json (mocked requests.post)
class FakeResp:
    status_code = 402
    text = '{"error":{"message":"insufficient credits"}}'

class FakeRequests:
    @staticmethod
    def post(url, headers=None, json=None, timeout=None):
        return FakeResp()

real_requests = mod.requests
mod.requests = FakeRequests()
try:
    REAL_POST_JSON("https://api.lumenfall.ai/openai/v1/images/generations", {}, {},
                   timeout=5, max_retries=2, retry_delay=0.01)
    raise AssertionError("should have raised")
except FATAL:
    print("T6 HTTP 402 is fatal (abort, no pointless retry) PASS")
finally:
    mod.requests = real_requests

# T7: backend registered and reachable through BACKENDS
assert mod.BACKENDS["lumenfall"] is mod.LumenfallBackend
assert mod.BACKEND_DEFAULTS["lumenfall"]["api_key_env"] == "LUMENFALL_API_KEY"
print("T7 backend registered in BACKENDS/DEFAULTS .... PASS")

print("\nALL TESTS PASSED")
