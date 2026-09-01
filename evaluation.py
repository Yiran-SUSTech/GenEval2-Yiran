# Copyright: Meta Platforms, Inc. and affiliates

import argparse
import base64
import json
import math
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from threading import Lock

from tqdm import tqdm
from scipy.stats import gmean

from benchmark_schema import build_answer_list, load_benchmark_records, load_image_manifest, resolve_image_path

DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_API_MODEL = "qwen3.6-plus"
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


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


def is_retryable_error(e):
    status_code = getattr(getattr(e, "response", None), "status_code", None)
    if status_code in (400, 401, 403, 422):
        return False
    error_str = str(e).lower()
    non_retryable = [
        "datainspectionfailed",
        "data inspection failed",
        "content_filter",
        "content filter",
        "inappropriate content",
        "invalid_api_key",
        "authentication",
        "invalid request",
    ]
    return not any(keyword in error_str for keyword in non_retryable)


def is_fatal_config_error(e):
    """Auth or request-parameter errors that would fail identically on every call."""
    error_str = str(e).lower()
    fatal_keywords = [
        "invalid_api_key",
        "authentication",
        "unauthorized",
        "invalid_parameter",
        "range of top_logprobs",
        "'top_logprobs'",
    ]
    return any(keyword in error_str for keyword in fatal_keywords)


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

    def __init__(self, model_name, api_base, api_key, top_logprobs=20, max_retries=5, retry_delay=2.0):
        from openai import OpenAI

        if not api_key:
            raise SystemExit("API key missing: pass --api_key or set the DASHSCOPE_API_KEY environment variable")
        self.model_name = model_name
        self.top_logprobs = top_logprobs
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.failed_calls = 0
        self._lock = Lock()
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        print(f"Using API VQA model: {model_name} @ {api_base}")

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
            with self._lock:
                self.failed_calls += 1
            print(f"  [WARN] VQA API call failed for {image_filepath}: {type(e).__name__}: {e}; scoring as 0")
            return "", 0.0

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


def build_judge(args):
    if args.vqa_backend == "local":
        return LocalVQAJudge(args.vqa_model or DEFAULT_LOCAL_MODEL)
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY", "")
    return APIVQAJudge(
        args.vqa_model or DEFAULT_API_MODEL,
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
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON filepath")
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
    judge = build_judge(args)

    concurrency = args.concurrency
    if concurrency > 1 and args.vqa_backend != "api":
        print("[WARN] --concurrency > 1 is only supported with --vqa_backend api; running sequentially")
        concurrency = 1

    def worker(record):
        return score_record(judge, record, image_data, args.method)

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(tqdm(pool.map(worker, benchmark_data), total=len(benchmark_data), desc="Evaluating"))
    else:
        results = [worker(record) for record in tqdm(benchmark_data, desc="Evaluating")]

    summary = summarize_results(results)
    output = {
        "method": args.method,
        "vqa_backend": args.vqa_backend,
        "vqa_model": judge.model_name,
        "summary": summary,
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

    if failed_calls:
        print(f"[WARN] {failed_calls} API call(s) failed after retries and were scored as 0")

    if args.method == "soft_tifa_gm":
        total_score = summary["overall_gm"]
    else:
        total_score = summary["overall_am"]
    print(f"Score: {total_score}")


if __name__ == "__main__":
    main()
