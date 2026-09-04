# Copyright: Meta Platforms, Inc. and affiliates

import argparse
import base64
import json
import math
import mimetypes
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from threading import Lock

from PIL import Image
from tqdm import tqdm
from scipy.stats import gmean

from benchmark_schema import build_answer_list, load_benchmark_records, load_image_manifest, resolve_image_path

DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_API_MODEL = "qwen3.6-plus"
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

ABORT_REJECT_THRESHOLD = 30


class ImageRejectedError(Exception):
    """The VQA judge cannot score this particular image (corrupt/unsupported file or content filter)."""


class LocalVQAJudge:
    """VQA judge running a local Qwen3-VL model via transformers."""

    def __init__(self, model_name):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model_name = model_name
        self.torch = torch
        print(f"Loading local VQA model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
        )

    def begin_sample(self):
        pass

    @property
    def sample_failures(self):
        return 0

    def send_message_with_image(self, prompt, image_filepath, answer_list=None):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_filepath},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        scores = outputs.scores[0]
        probs = self.torch.nn.functional.softmax(scores, dim=-1)

        if answer_list:
            lm_prob = 0.0
            for answer in answer_list:
                ans_token_id = self.processor.tokenizer.encode(answer)[0]
                lm_prob += probs[0, ans_token_id].item()
        else:
            lm_prob = None

        argmax_token = self.processor.batch_decode([self.torch.argmax(probs)])[0]
        return argmax_token.strip(), lm_prob


def _normalized_error_text(e):
    return re.sub(r"[^a-z0-9]", "", str(e).lower())


def is_retryable_error(e):
    status_code = getattr(getattr(e, "response", None), "status_code", None)
    if status_code in (400, 401, 403, 422):
        return False
    error_str = _normalized_error_text(e)
    non_retryable = [
        "datainspectionfailed",
        "contentfilter",
        "inappropriatecontent",
        "invalidapikey",
        "authentication",
        "invalidrequest",
    ]
    return not any(keyword in error_str for keyword in non_retryable)


def is_fatal_config_error(e):
    """Errors that would fail identically on every call (auth, request parameters, quota)."""
    error_str = _normalized_error_text(e)
    fatal_keywords = [
        "invalidapikey",
        "authentication",
        "unauthorized",
        "modelfound",
        "modelnotfound",
        "modelnotexist",
        "rangeoftoplogprobs",
        "toplogprobs",
        "insufficientbalance",
        "arrears",
    ]
    return any(keyword in error_str for keyword in fatal_keywords)


def is_image_rejection(e):
    """Errors where the API rejects this particular image (bad payload or content filter)."""
    error_str = _normalized_error_text(e)
    rejection_keywords = [
        "urldoesnotappeartobevalid",
        "invalidimage",
        "notavalidimage",
        "imageisinvalid",
        "failedtodecode",
        "datainspectionfailed",
        "contentfilter",
        "inappropriatecontent",
    ]
    return any(keyword in error_str for keyword in rejection_keywords)


def answer_prob_from_logprobs(top_logprobs, answer_list):
    """Sum probabilities of answer-variant tokens among the top-k token logprobs.

    Matching tolerates casing/whitespace differences between the variant strings
    and the tokenizer's token texts (e.g. token "YES" vs variant "Yes").
    """
    wanted = set(answer_list)
    wanted_norm = {variant.strip().lower() for variant in answer_list}
    total = 0.0
    for lp in top_logprobs or []:
        token = lp.token
        if token in wanted or token.strip().lower() in wanted_norm:
            total += math.exp(lp.logprob)
    return total


class APIVQAJudge:
    """VQA judge calling an OpenAI-compatible multimodal API (e.g. DashScope qwen3.6-plus).

    Soft-TIFA needs the probability mass on the answer tokens, so each request
    generates one token and asks for its top-k logprobs.
    """

    def __init__(self, model_name, api_base, api_key, top_logprobs=5, max_retries=5, retry_delay=2.0):
        from openai import OpenAI

        if not api_key:
            raise SystemExit("API key missing: pass --api_key or set the DASHSCOPE_API_KEY environment variable")
        self.model_name = model_name
        self.top_logprobs = top_logprobs
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.failed_calls = 0
        self.consecutive_rejects = 0
        self.abort_reject_threshold = ABORT_REJECT_THRESHOLD
        self._lock = Lock()
        self._local = threading.local()
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        print(f"Using API VQA model: {model_name} @ {api_base}")

    def begin_sample(self):
        self._local.sample_failures = 0

    @property
    def sample_failures(self):
        return getattr(self._local, "sample_failures", 0)

    def _register_hard_failure(self):
        with self._lock:
            self.failed_calls += 1
            self.consecutive_rejects += 1
            should_abort = self.consecutive_rejects >= self.abort_reject_threshold
        self._local.sample_failures = getattr(self._local, "sample_failures", 0) + 1
        if should_abort:
            raise SystemExit(
                f"Aborting: {self.consecutive_rejects} consecutive failed/rejected API calls. "
                "Isolated broken images are skipped automatically, so this many in a row points to a "
                "systematic problem (all images unreadable, an API-side change, or account quota)."
            )

    def _create_with_retry(self, **kwargs):
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                last_exception = e
                if not is_retryable_error(e):
                    if is_fatal_config_error(e):
                        raise SystemExit(
                            "Fatal API configuration error (aborting — every call would fail): "
                            f"{e}"
                        )
                    raise
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    print(f"  [RETRY] VQA API call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    print(f"  [RETRY] Retrying in {delay:.1f}s...")
                    time.sleep(delay)
        raise last_exception

    def _encode_image(self, image_filepath):
        data = Path(image_filepath).read_bytes()
        mime = mimetypes.guess_type(str(image_filepath))[0] or "image/png"
        return base64.b64encode(data).decode("utf-8"), mime

    def send_message_with_image(self, prompt, image_filepath, answer_list=None):
        b64_image, mime = self._encode_image(image_filepath)
        try:
            completion = self._create_with_retry(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_image}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=self.top_logprobs,
                extra_body={"enable_thinking": False},
            )
        except Exception as e:
            if is_image_rejection(e):
                self._register_hard_failure()
                raise ImageRejectedError(
                    f"VQA API rejected the image {image_filepath}: {type(e).__name__}: {e}"
                ) from e
            self._register_hard_failure()
            print(f"  [WARN] VQA API call failed for {image_filepath}: {type(e).__name__}: {e}; scoring as 0")
            return "", 0.0

        with self._lock:
            self.consecutive_rejects = 0

        choice = completion.choices[0]
        pred = (choice.message.content or "").strip()
        if not answer_list:
            return pred, None

        logprobs = getattr(choice, "logprobs", None)
        entries = getattr(logprobs, "content", None) if logprobs else None
        top_logprobs = getattr(entries[0], "top_logprobs", None) if entries else None
        if not top_logprobs:
            raise RuntimeError(
                f"API response for model '{self.model_name}' has no token logprobs; "
                "Soft-TIFA soft scores cannot be computed. The model or endpoint may not "
                "support the logprobs parameter."
            )
        return pred, answer_prob_from_logprobs(top_logprobs, answer_list)


def resolve_vqa_model_name(args):
    if args.vqa_backend == "local":
        return args.vqa_model or DEFAULT_LOCAL_MODEL
    return args.vqa_model or DEFAULT_API_MODEL


def build_judge(args):
    if args.vqa_backend == "local":
        return LocalVQAJudge(resolve_vqa_model_name(args))
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY", "")
    return APIVQAJudge(
        resolve_vqa_model_name(args),
        api_base=args.api_base,
        api_key=api_key,
        top_logprobs=args.top_logprobs,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )


def vqa_score(judge, prompt, image_filepath):
    message_prompt = f'Does this image show "{prompt}"? Answer the question with Yes or No.'
    _, ans_prob = judge.send_message_with_image(message_prompt, image_filepath, answer_list=["Yes", "yes", " Yes", " yes"])
    return ans_prob


def score_atoms(judge, atoms, image_filepath, hard=False):
    score_list = []
    correct = 0.0
    for atom in atoms:
        answer_list = build_answer_list(atom)
        question = atom["question"]
        if not question.rstrip().endswith((".", "?")):
            question = f"{question}."
        prompt = f"{question} Answer in one word."
        pred, ans_prob = judge.send_message_with_image(prompt, image_filepath, answer_list=answer_list)
        if hard:
            normalized_pred = pred.strip().lower()
            normalized_answers = {answer.strip().lower() for answer in answer_list}
            matched = normalized_pred in normalized_answers
            score_list.append(1 if matched else 0)
            correct += 1 if matched else 0
        else:
            score_value = float(ans_prob or 0.0)
            score_list.append(score_value)
            correct += score_value
    return correct / max(len(atoms), 1), score_list


def tifa(judge, atoms, image_filepath):
    return score_atoms(judge, atoms, image_filepath, hard=True)


def soft_tifa(judge, atoms, image_filepath):
    return score_atoms(judge, atoms, image_filepath, hard=False)


def safe_gmean(values):
    if not values:
        return 0.0
    clipped = [max(float(value), 1e-12) for value in values]
    return float(gmean(clipped))


def summarize_results(results):
    by_task = {}
    for result in results:
        by_task.setdefault(result["task_type"], []).append(result)

    summary = {
        "overall_am": 100.0 * mean(result["sample_am"] for result in results) if results else 0.0,
        "overall_gm": 100.0 * mean(result["sample_gm"] for result in results) if results else 0.0,
        "task_summaries": {},
    }

    task_gms = []
    for task_type, task_results in by_task.items():
        task_am = 100.0 * mean(result["sample_am"] for result in task_results)
        task_gm = 100.0 * mean(result["sample_gm"] for result in task_results)
        summary["task_summaries"][task_type] = {
            "sample_count": len(task_results),
            "am": task_am,
            "gm": task_gm,
        }
        task_gms.append(task_gm)

    summary["joint_task_macro_avg"] = mean(task_gms) if task_gms else 0.0
    return summary


def legacy_score_lists(results):
    return [result["atom_scores"] for result in results]


def score_record(judge, record, image_manifest, method):
    image_filepath = resolve_image_path(record, image_manifest)
    if method == "vqascore":
        score = vqa_score(judge, record["prompt"], image_filepath)
        atom_scores = [score]
        sample_am = float(score)
        sample_gm = float(score)
    elif method == "tifa":
        sample_am, atom_scores = tifa(judge, record["atoms"], image_filepath)
        sample_gm = safe_gmean(atom_scores)
    elif method in {"soft_tifa_am", "soft_tifa_gm"}:
        sample_am, atom_scores = soft_tifa(judge, record["atoms"], image_filepath)
        sample_gm = safe_gmean(atom_scores)
    else:
        raise NotImplementedError

    return {
        "sample_id": record["sample_id"],
        "task_type": record["task_type"],
        "prompt": record["prompt"],
        "condition": record["condition"],
        "metadata": record["metadata"],
        "atom_questions": [atom["question"] for atom in record["atoms"]],
        "atom_skills": [atom.get("skill", "unspecified") for atom in record["atoms"]],
        "atom_scores": atom_scores,
        "sample_am": sample_am,
        "sample_gm": sample_gm,
    }


def describe_image(image_path):
    try:
        size_mb = Path(image_path).stat().st_size / 1e6
        with Image.open(image_path) as image:
            return f"file_size={size_mb:.2f}MB format={image.format} dimensions={image.size} mode={image.mode}"
    except Exception as e:
        return f"unreadable by PIL: {type(e).__name__}: {e}"


def image_path_or_none(record, image_data):
    try:
        return resolve_image_path(record, image_data)
    except KeyError:
        return None


def preflight_images(records, image_data):
    """Decode every image before spending API budget; unreadable ones are skipped and reported."""
    broken = {}
    for record in tqdm(records, desc="Pre-flight", unit="img"):
        try:
            image_path = resolve_image_path(record, image_data)
        except KeyError as e:
            broken[record["sample_id"]] = f"missing from image manifest: {e}"
            continue
        if not Path(image_path).is_file():
            broken[record["sample_id"]] = f"image file not found: {image_path}"
            continue
        try:
            with Image.open(image_path) as image:
                image.load()
        except Exception as e:
            broken[record["sample_id"]] = f"unreadable image: {type(e).__name__}: {e}"
    return broken


def load_checkpoint(partial_path, args, vqa_model_name):
    """Load {sample_id: result} from the checkpoint of a previous interrupted run."""
    with open(partial_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    if not lines:
        return {}
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError:
        raise SystemExit(f"Corrupt checkpoint file: {partial_path} — delete it and rerun.")
    expected = {
        "method": args.method,
        "vqa_backend": args.vqa_backend,
        "vqa_model": vqa_model_name,
        "benchmark": args.benchmark_data,
        "image_manifest": args.image_filepath_data,
    }
    for key, value in expected.items():
        if header.get(key) != value:
            raise SystemExit(
                f"Checkpoint {partial_path} was written for {key}={header.get(key)!r}, but this run uses "
                f"{value!r}. Use a different --output_file or delete {partial_path} to start fresh."
            )
    checkpointed = {}
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            print("[WARN] Skipped one corrupt checkpoint line (interrupted write)")
            continue
        checkpointed[result["sample_id"]] = result
    return checkpointed


def main():
    parser = argparse.ArgumentParser(description="Evaluate T2I or C2I images with Soft-TIFA")
    parser.add_argument("--benchmark_data", type=str, required=True, help="Path to benchmark data")
    parser.add_argument("--image_filepath_data", type=str, required=True, help="Path to JSON image manifest")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["vqascore", "tifa", "soft_tifa_am", "soft_tifa_gm"],
        help="Method name",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSON filepath; an interrupted run can be resumed by rerunning the same command "
        "(checkpoint kept at <output>.partial.jsonl)",
    )
    parser.add_argument(
        "--legacy_output_file",
        type=str,
        required=False,
        default=None,
        help="Optional path to save legacy score_lists JSON",
    )
    parser.add_argument(
        "--vqa_backend",
        type=str,
        required=False,
        default="local",
        choices=["local", "api"],
        help="VQA judge backend: local transformers model or OpenAI-compatible API",
    )
    parser.add_argument(
        "--vqa_model",
        type=str,
        required=False,
        default=None,
        help=(
            "VQA model: local path/HF id for --vqa_backend local "
            f"(default {DEFAULT_LOCAL_MODEL}), or API model name for --vqa_backend api "
            f"(default {DEFAULT_API_MODEL})"
        ),
    )
    parser.add_argument(
        "--api_base",
        type=str,
        required=False,
        default=DEFAULT_API_BASE,
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=False,
        default=None,
        help="API key (defaults to the DASHSCOPE_API_KEY environment variable)",
    )
    parser.add_argument(
        "--top_logprobs",
        type=int,
        required=False,
        default=5,
        help="Number of top token logprobs to request when scoring via API (DashScope allows at most 5)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        required=False,
        default=5,
        help="Retries per API call for transient failures",
    )
    parser.add_argument(
        "--retry_delay",
        type=float,
        required=False,
        default=2.0,
        help="Base delay in seconds between API retries (doubles each retry)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        required=False,
        default=1,
        help="Parallel API requests (API backend only; local backend always runs sequentially)",
    )
    args = parser.parse_args()

    benchmark_data = load_benchmark_records(args.benchmark_data)
    image_data = load_image_manifest(args.image_filepath_data)

    vqa_model_name = resolve_vqa_model_name(args)
    partial_path = Path(args.output_file).with_suffix(".partial.jsonl")

    checkpointed = {}
    if partial_path.exists():
        checkpointed = load_checkpoint(partial_path, args, vqa_model_name)
        if checkpointed:
            print(f"Resuming: {len(checkpointed)} scored sample(s) loaded from {partial_path}")

    judge = build_judge(args)

    concurrency = args.concurrency
    if concurrency > 1 and args.vqa_backend != "api":
        print("[WARN] --concurrency > 1 is only supported with --vqa_backend api; running sequentially")
        concurrency = 1

    todo_records = [record for record in benchmark_data if record["sample_id"] not in checkpointed]
    broken = preflight_images(todo_records, image_data)
    if broken:
        print(f"[WARN] {len(broken)} image(s) unreadable; they will be skipped and reported in failed_samples:")
        for sample_id, reason in list(broken.items())[:10]:
            print(f"  - {sample_id}: {reason}")
        if len(broken) > 10:
            print(f"  ... and {len(broken) - 10} more")
    todo = [record for record in todo_records if record["sample_id"] not in broken]

    failed_samples = []
    for record in todo_records:
        if record["sample_id"] in broken:
            image_path = image_path_or_none(record, image_data)
            failed_samples.append(
                {
                    "sample_id": record["sample_id"],
                    "image": image_path,
                    "error": broken[record["sample_id"]],
                    "diagnostics": describe_image(image_path) if image_path else "no image path",
                    "stage": "preflight",
                }
            )

    is_fresh_checkpoint = not partial_path.exists()
    partial_handle = open(partial_path, "a", encoding="utf-8")
    if is_fresh_checkpoint:
        partial_handle.write(
            json.dumps(
                {
                    "method": args.method,
                    "vqa_backend": args.vqa_backend,
                    "vqa_model": vqa_model_name,
                    "benchmark": args.benchmark_data,
                    "image_manifest": args.image_filepath_data,
                }
            )
            + "\n"
        )

    def worker(record):
        judge.begin_sample()
        try:
            result = score_record(judge, record, image_data, args.method)
        except ImageRejectedError as e:
            return record, None, str(e), False
        return record, result, None, judge.sample_failures == 0

    results_by_id = dict(checkpointed)

    if todo:
        executor = ThreadPoolExecutor(max_workers=concurrency)
        futures = [executor.submit(worker, record) for record in todo]
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                record, result, rejection, clean = future.result()
                if rejection is not None:
                    image_path = image_path_or_none(record, image_data)
                    entry = {
                        "sample_id": record["sample_id"],
                        "image": image_path,
                        "error": rejection,
                        "diagnostics": describe_image(image_path) if image_path else "no image path",
                        "stage": "runtime",
                    }
                    failed_samples.append(entry)
                    print(f"  [SKIP] VQA API rejected the image (sample {record['sample_id']}): {image_path}")
                    print(f"         {rejection}")
                    print(f"         {entry['diagnostics']}")
                    continue
                results_by_id[record["sample_id"]] = result
                if clean:
                    partial_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    partial_handle.flush()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            partial_handle.close()
    else:
        partial_handle.close()

    results = [
        results_by_id[record["sample_id"]]
        for record in benchmark_data
        if record["sample_id"] in results_by_id
    ]

    summary = summarize_results(results)
    output = {
        "method": args.method,
        "vqa_backend": args.vqa_backend,
        "vqa_model": judge.model_name,
        "summary": summary,
        "scored_samples": len(results),
        "failed_samples": failed_samples,
        "results": results,
    }
    failed_calls = getattr(judge, "failed_calls", 0)
    if failed_calls:
        output["failed_api_calls"] = failed_calls
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    legacy_output_file = args.legacy_output_file or args.output_file.replace(".json", "_score_lists.json")
    with open(legacy_output_file, "w", encoding="utf-8") as handle:
        json.dump(legacy_score_lists(results), handle)

    partial_path.unlink(missing_ok=True)

    if failed_samples:
        print(
            f"[WARN] {len(failed_samples)} image(s) could not be scored and are excluded from the summary "
            f"— see 'failed_samples' in {args.output_file}"
        )
    if failed_calls:
        print(f"[WARN] {failed_calls} API call(s) failed after retries and were scored as 0")

    if args.method == "soft_tifa_gm":
        total_score = summary["overall_gm"]
    else:
        total_score = summary["overall_am"]
    print(f"Score: {total_score}")

    if not results:
        raise SystemExit("No samples were scored — check the warnings above and 'failed_samples' in the output file.")


if __name__ == "__main__":
    main()
