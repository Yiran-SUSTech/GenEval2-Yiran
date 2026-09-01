# Copyright: Meta Platforms, Inc. and affiliates

import json
import argparse
from statistics import mean

import torch
from tqdm import tqdm
from scipy.stats import gmean
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from benchmark_schema import build_answer_list, load_benchmark_records, load_image_manifest, resolve_image_path


print("Loading Qwen")
qwen_processor = AutoProcessor.from_pretrained(
    "/mnt/afs/zhengmingkai/zyr/GenEval2-Yiran/Qwen3-VL-8B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/mnt/afs/zhengmingkai/zyr/GenEval2-Yiran/Qwen3-VL-8B-Instruct",
    dtype="auto",
    device_map="auto",
)


def construct_message_with_image(prompt, image_filepath):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_filepath},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def send_message_with_image(prompt, image_filepath, answer_list=None):
    messages = construct_message_with_image(prompt, image_filepath)
    inputs = qwen_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(qwen_model.device)
    outputs = qwen_model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    scores = outputs.scores[0]
    probs = torch.nn.functional.softmax(scores, dim=-1)

    if answer_list:
        lm_prob = 0.0
        for answer in answer_list:
            ans_token_id = qwen_processor.tokenizer.encode(answer)[0]
            lm_prob += probs[0, ans_token_id].item()
    else:
        lm_prob = None

    argmax_token = qwen_processor.batch_decode([torch.argmax(probs)])[0]
    return argmax_token.strip(), lm_prob


def vqa_score(prompt, image_filepath):
    message_prompt = f'Does this image show "{prompt}"? Answer the question with Yes or No.'
    _, ans_prob = send_message_with_image(message_prompt, image_filepath, answer_list=["Yes", "yes", " yes", " Yes"])
    return ans_prob


def score_atoms(atoms, image_filepath, hard=False):
    score_list = []
    correct = 0.0
    for atom in atoms:
        answer_list = build_answer_list(atom)
        question = atom["question"]
        if not question.rstrip().endswith((".", "?")):
            question = f"{question}."
        prompt = f"{question} Answer in one word."
        pred, ans_prob = send_message_with_image(prompt, image_filepath, answer_list=answer_list)
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


def tifa(atoms, image_filepath):
    return score_atoms(atoms, image_filepath, hard=True)


def soft_tifa(atoms, image_filepath):
    return score_atoms(atoms, image_filepath, hard=False)


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
    args = parser.parse_args()

    benchmark_data = load_benchmark_records(args.benchmark_data)
    image_data = load_image_manifest(args.image_filepath_data)
    results = []

    for record in tqdm(benchmark_data):
        image_filepath = resolve_image_path(record, image_data)
        if args.method == "vqascore":
            score = vqa_score(record["prompt"], image_filepath)
            atom_scores = [score]
            sample_am = float(score)
            sample_gm = float(score)
        elif args.method == "tifa":
            sample_am, atom_scores = tifa(record["atoms"], image_filepath)
            sample_gm = safe_gmean(atom_scores)
        elif args.method in {"soft_tifa_am", "soft_tifa_gm"}:
            sample_am, atom_scores = soft_tifa(record["atoms"], image_filepath)
            sample_gm = safe_gmean(atom_scores)
        else:
            raise NotImplementedError

        results.append(
            {
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
        )

    summary = summarize_results(results)
    output = {
        "method": args.method,
        "summary": summary,
        "results": results,
    }
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    legacy_output_file = args.legacy_output_file or args.output_file.replace(".json", "_score_lists.json")
    with open(legacy_output_file, "w", encoding="utf-8") as handle:
        json.dump(legacy_score_lists(results), handle)

    if args.method == "soft_tifa_gm":
        total_score = summary["overall_gm"]
    else:
        total_score = summary["overall_am"]
    print(f"Score: {total_score}")


if __name__ == "__main__":
    main()
