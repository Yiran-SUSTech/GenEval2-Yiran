# Copyright: Meta Platforms, Inc. and affiliates

import argparse
import base64
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

BACKEND_DEFAULTS = {
    "minimax": {
        "model": "image-01",
        "base_url": "https://api.minimaxi.com",
        "api_key_env": "MINIMAX_API_KEY",
    },
    "qwen": {
        "model": "qwen-image-3.0-pro",
        "base_url": "https://dashscope.aliyuncs.com",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "gemini": {
        "model": "gemini-3.1-flash-image",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key_env": "GEMINI_API_KEY",
    },
}

FATAL_KEYWORDS = [
    "invalidapikey",
    "invalidapikey",
    "invalidtoken",
    "authentication",
    "unauthorized",
    "modelnotfound",
    "invalidmodel",
    "accessdenied",
    "permissiondenied",
]


class FatalAPIError(Exception):
    """Configuration error that would fail identically on every call."""


class GenerationError(Exception):
    """This prompt failed (content policy, generation error, exhausted retries)."""


def is_fatal_error(text):
    compact = re.sub(r"[^a-z0-9]", "", text.lower())
    if any(keyword in compact for keyword in FATAL_KEYWORDS):
        return True
    # Billing / daily-quota exhaustion ("You exceeded your current quota, please check your plan
    # and billing details"). Unlike per-minute rate limits, waiting 10s cannot fix these, so abort.
    if "exceededyourcurrentquota" in compact or "checkyourplanandbilling" in compact:
        return "perminute" not in compact
    return False


class RateLimitCooldown:
    """After the first HTTP 429, pause before every request for the rest of the run."""

    def __init__(self, wait_seconds):
        self.wait_seconds = wait_seconds
        self._active = False
        self._lock = threading.Lock()

    @property
    def active(self):
        return self._active

    def trigger(self):
        with self._lock:
            if not self._active:
                self._active = True
                print(f"  [THROTTLE] Rate limit hit — pausing {self.wait_seconds:.1f}s before every request from now on")

    def wait_if_active(self):
        if self._active:
            time.sleep(self.wait_seconds)


def post_json(url, headers, payload, timeout, max_retries, retry_delay, cooldown=None):
    last_error = ""
    for attempt in range(max_retries + 1):
        if attempt == 0 and cooldown is not None:
            cooldown.wait_if_active()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            last_error = f"network error: {type(e).__name__}: {e}"
            delay = retry_delay * (2**attempt)
        else:
            if resp.status_code == 200:
                return resp.json()
            body = resp.text[:1000]
            if is_fatal_error(body):
                raise FatalAPIError(f"HTTP {resp.status_code}: {body}")
            if resp.status_code == 429:
                if cooldown is not None:
                    cooldown.trigger()
                    delay = cooldown.wait_seconds
                else:
                    delay = retry_delay * (2**attempt)
                last_error = f"HTTP 429: {body}"
            elif resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}: {body}"
                delay = retry_delay * (2**attempt)
            else:
                raise GenerationError(f"HTTP {resp.status_code}: {body}")
        if attempt < max_retries:
            print(f"  [RETRY] {last_error} — retrying in {delay:.1f}s")
            time.sleep(delay)
    raise GenerationError(last_error)


def download_bytes(url, timeout=120, max_retries=3, retry_delay=2.0):
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            last_error = f"download failed: {type(e).__name__}: {e}"
        if attempt < max_retries:
            delay = retry_delay * (2**attempt)
            print(f"  [RETRY] {last_error} — retrying in {delay:.1f}s")
            time.sleep(delay)
    raise GenerationError(last_error)


def save_png(path, data):
    image = Image.open(BytesIO(data))
    image.save(path, "PNG")


class MiniMaxBackend:
    def __init__(self, args, api_key):
        self.model = args.model
        self.base_url = args.base_url.rstrip("/")
        self.aspect_ratio = args.aspect_ratio
        self.seed = args.seed
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.args = args

    def generate(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "aspect_ratio": self.aspect_ratio,
            "response_format": "url",
            "n": 1,
            "prompt_optimizer": False,
            "aigc_watermark": False,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        data = post_json(
            f"{self.base_url}/v1/image_generation",
            self.headers,
            payload,
            self.args.timeout,
            self.args.max_retries,
            self.args.retry_delay,
            cooldown=self.args.cooldown,
        )
        status = data.get("base_resp", {})
        if status.get("status_code", 0) != 0:
            raise GenerationError(f"MiniMax error {status.get('status_code')}: {status.get('status_msg')}")
        urls = data.get("data", {}).get("image_urls") or []
        if not urls:
            raise GenerationError("MiniMax returned no image urls (content filter?)")
        return download_bytes(urls[0], timeout=self.args.timeout)


class QwenBackend:
    def __init__(self, args, api_key):
        self.model = args.model
        self.base_url = args.base_url.rstrip("/")
        self.size = args.size
        self.seed = args.seed
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.args = args

    def generate(self, prompt):
        parameters = {"prompt_extend": False, "n": 1, "watermark": False}
        if self.size and self.size != "auto":
            parameters["size"] = self.size
        if self.seed is not None:
            parameters["seed"] = self.seed
        payload = {
            "model": self.model,
            "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]},
            "parameters": parameters,
        }
        data = post_json(
            f"{self.base_url}/api/v1/services/aigc/multimodal-generation/generation",
            self.headers,
            payload,
            self.args.timeout,
            self.args.max_retries,
            self.args.retry_delay,
            cooldown=self.args.cooldown,
        )
        if data.get("code"):
            raise GenerationError(f"Qwen error {data.get('code')}: {data.get('message')}")
        choices = data.get("output", {}).get("choices") or []
        for choice in choices:
            for item in choice.get("message", {}).get("content", []):
                if item.get("image"):
                    return download_bytes(item["image"], timeout=self.args.timeout)
        raise GenerationError("Qwen returned no image url (content filter?)")


class GeminiBackend:
    def __init__(self, args, api_key):
        self.model = args.model
        self.base_url = args.base_url.rstrip("/")
        self.aspect_ratio = args.aspect_ratio
        self.image_size = args.image_size
        if args.auth_scheme == "bearer":
            self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        else:
            self.headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        self.args = args

    def generate(self, prompt):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "aspectRatio": self.aspect_ratio,
                    "imageSize": self.image_size,
                },
            },
        }
        data = post_json(
            f"{self.base_url}/models/{self.model}:generateContent",
            self.headers,
            payload,
            self.args.timeout,
            self.args.max_retries,
            self.args.retry_delay,
            cooldown=self.args.cooldown,
        )
        block_reason = (data.get("promptFeedback") or {}).get("blockReason")
        if block_reason:
            raise GenerationError(f"blocked by safety filter: {block_reason}")
        candidates = data.get("candidates") or []
        if not candidates:
            raise GenerationError("Gemini returned no candidates")
        for part in (candidates[0].get("content") or {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
        raise GenerationError(f"no image part in response (finishReason={candidates[0].get('finishReason')})")


BACKENDS = {"minimax": MiniMaxBackend, "qwen": QwenBackend, "gemini": GeminiBackend}


def load_prompts(benchmark_path):
    prompts = []
    with open(benchmark_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompts.append(record["prompt"])
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Sample benchmark images from API-based T2I models")
    parser.add_argument("--backend", type=str, required=True, choices=list(BACKEND_DEFAULTS))
    parser.add_argument("--benchmark_data", type=str, default="./geneval2_data.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default=None, help="Overrides the backend default model")
    parser.add_argument("--api_key", type=str, default=None, help="Overrides the backend default env var")
    parser.add_argument("--base_url", type=str, default=None, help="Overrides the backend default base URL")
    parser.add_argument("--aspect_ratio", type=str, default="1:1", help="MiniMax / Gemini aspect ratio")
    parser.add_argument("--size", type=str, default="1024*1024", help="Qwen resolution W*H ('auto' to omit)")
    parser.add_argument("--image_size", type=str, default="1K", help="Gemini imageSize: 512 / 1K / 2K / 4K")
    parser.add_argument("--auth_scheme", type=str, default="x-goog-api-key", choices=["x-goog-api-key", "bearer"],
                        help="Gemini auth header style (bearer for OpenAI-style relays)")
    parser.add_argument("--seed", type=int, default=0, help="Seed passed to MiniMax / Qwen (best-effort)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_delay", type=float, default=2.0)
    parser.add_argument("--rate_limit_wait", type=float, default=10.0,
                        help="Seconds to wait after an HTTP 429, and before every following request once throttled")
    parser.add_argument("--timeout", type=int, default=300, help="Per-request timeout in seconds")
    parser.add_argument("--limit", type=int, default=None, help="Only sample the first N missing prompts (smoke test)")
    args = parser.parse_args()

    defaults = BACKEND_DEFAULTS[args.backend]
    args.model = args.model or defaults["model"]
    args.base_url = args.base_url or defaults["base_url"]
    args.cooldown = RateLimitCooldown(args.rate_limit_wait)
    api_key = args.api_key or os.environ.get(defaults["api_key_env"], "")
    if not api_key:
        raise SystemExit(f"API key missing: pass --api_key or set {defaults['api_key_env']}")

    prompts = load_prompts(args.benchmark_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    todo = [(i, p) for i, p in enumerate(prompts) if not (output_dir / f"{i}.png").exists()]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"Backend: {args.backend} | model: {args.model}")
    print(f"Prompts: {len(prompts)} | already sampled: {len(prompts) - len(todo) if args.limit is None else '?'} | to sample: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    backend = BACKENDS[args.backend](args, api_key)

    def work(item):
        index, prompt = item
        data = backend.generate(prompt)
        save_png(output_dir / f"{index}.png", data)
        return index

    failed = []
    ok_count = 0
    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    futures = {executor.submit(work, item): item for item in todo}
    try:
        for future in tqdm(as_completed(futures), total=len(futures), desc="Sampling"):
            item = futures[future]
            try:
                future.result()
                ok_count += 1
            except FatalAPIError as e:
                print(f"\n[FATAL] {e}")
                for pending in futures:
                    pending.cancel()
                raise SystemExit(1)
            except Exception as e:
                failed.append({"index": item[0], "prompt": item[1], "error": str(e)[:300]})
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    print(f"Done: {ok_count} generated, {len(failed)} failed, {len(prompts) - len(todo) if args.limit is None else 0} previously existing")
    if failed:
        failed_path = output_dir / "failed_samples.json"
        failed_path.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Failures saved to {failed_path} — rerun the same command to retry only missing images")


if __name__ == "__main__":
    main()
