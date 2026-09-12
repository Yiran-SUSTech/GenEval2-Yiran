import base64
import importlib.util

spec = importlib.util.spec_from_file_location("sample_images", r"D:\NewMetric\GenEval2\sample_images.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

GEN = mod.GenerationError


class Args:
    model = "bytedance-seed/seedream-4.5"
    base_url = "https://openrouter.ai/api/v1"
    aspect_ratio = "1:1"
    image_size = "1K"
    timeout = 300
    max_retries = 5
    retry_delay = 2.0
    cooldown = None


REAL_PIXEL_400 = ('HTTP 400: {"error":{"message":"bytedance-seed/seedream-4.5 requires at least '
                  '3,686,400 output pixels; size \\"1024x1024\\" is 1,048,576. Use a larger '
                  'resolution such as \\"2K\\", or omit resolution to use the default.","code":400}}')


def run_backend(errors_by_call):
    """errors_by_call: {call_index: GenerationError to raise}; other calls return a valid image."""
    calls = []

    def fake_post_json(url, headers, payload, *a, **kw):
        calls.append(dict(payload))
        err = errors_by_call.get(len(calls) - 1)
        if err is not None:
            raise err
        return {"data": [{"b64_json": base64.b64encode(b"pngbytes").decode()}]}

    mod.post_json = fake_post_json
    be = mod.OpenRouterBackend(Args(), "key")
    return be.generate("a green backpack and a pig"), calls, be


# T1: real Seedream pixel error on call 1 -> drop resolution -> call 2 succeeds
img, calls, be = run_backend({0: GEN(REAL_PIXEL_400)})
assert img == b"pngbytes"
assert calls[0] == {"model": Args.model, "prompt": "a green backpack and a pig",
                    "aspect_ratio": "1:1", "resolution": "1K"}, calls[0]
assert calls[1] == {"model": Args.model, "prompt": "a green backpack and a pig",
                    "aspect_ratio": "1:1"}, calls[1]
assert "resolution" not in be.enabled_fields
print("T1 pixel-min 400 auto-drops resolution ......... PASS")

# T2: aspect_ratio unsupported on call 1 -> dropped, resolution kept
img, calls, be = run_backend({0: GEN('HTTP 400: {"error":{"message":"aspect_ratio value is not '
                                         'one of the allowed options"}}')})
assert img == b"pngbytes"
assert calls[1] == {"model": Args.model, "prompt": "a green backpack and a pig", "resolution": "1K"}
print("T2 aspect_ratio rejection still auto-drops ..... PASS")

# T3: content-policy 400 mentioning no field -> GenerationError propagates (no infinite loop)
try:
    run_backend({0: GEN('HTTP 400: {"error":{"message":"Your request was rejected as a result '
                                'of safety"}}')})
    raise AssertionError("should have raised")
except GEN:
    print("T3 unrelated 400 propagates as sample failure .. PASS")

# T4: pixel error arriving when resolution is already dropped -> no field left -> clean failure
try:
    run_backend({0: GEN(REAL_PIXEL_400), 1: GEN(REAL_PIXEL_400)})
    raise AssertionError("should have raised")
except GEN:
    print("T4 un-droppable error terminates cleanly ........... PASS")

# T5: no errors, default path unchanged
img, calls, be = run_backend({})
assert img == b"pngbytes" and len(calls) == 1
print("T5 clean 200 path unchanged ..................... PASS")

print("\nALL TESTS PASSED")
