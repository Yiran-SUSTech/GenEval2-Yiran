# Copyright: Meta Platforms, Inc. and affiliates

import json
import argparse
from collections import defaultdict

from scipy.stats import gmean

from benchmark_schema import load_benchmark_records


def safe_gmean(values):
    if not values:
        return 0.0
    clipped = [max(float(value), 1e-12) for value in values]
    return float(gmean(clipped))


def average(values):
    return sum(values) / len(values) if values else 0.0


def per_skill_analysis(all_score_lists, all_skill_lists):
    skill_buckets = defaultdict(list)
    for score_list, skill_list in zip(all_score_lists, all_skill_lists):
        for score, skill in zip(score_list, skill_list):
            skill_buckets[skill].append(score)
    return {skill: 100 * average(scores) for skill, scores in skill_buckets.items()}


def per_atomicity_analysis(all_score_lists, atomicity_list):
    all_atomicity_dict = defaultdict(list)
    for score_list, atomicity in zip(all_score_lists, atomicity_list):
        all_atomicity_dict[int(atomicity)].append(safe_gmean(score_list))
    return {atomicity: 100 * average(scores) for atomicity, scores in sorted(all_atomicity_dict.items())}


def load_legacy_analysis_inputs(benchmark_data_path, score_data_path):
    benchmark_data = load_benchmark_records(benchmark_data_path)
    all_score_lists = json.load(open(score_data_path, encoding="utf-8"))
    return benchmark_data, all_score_lists


def load_rich_results(score_data_path):
    data = json.load(open(score_data_path, encoding="utf-8"))
    if isinstance(data, dict) and "results" in data:
        return data
    return None


def analyze_t2i_records(records):
    score_lists = [record["atom_scores"] for record in records]
    skill_lists = [record.get("atom_skills", []) for record in records]
    atomicity_list = []
    filtered_scores = []
    filtered_skills = []
    for record, score_list, skill_list in zip(records, score_lists, skill_lists):
        if record.get("metadata", {}).get("atom_count") is not None:
            atomicity_list.append(record["metadata"]["atom_count"])
            filtered_scores.append(score_list)
            filtered_skills.append(skill_list)
    print("Per Atom Type Analysis (Soft-TIFA AM)")
    for skill, accuracy in per_skill_analysis(score_lists, skill_lists).items():
        print(f"{skill}: {round(accuracy, 2)}")
    print()
    if atomicity_list:
        print("Per Atomicity Analysis (Soft-TIFA GM)")
        for atomicity, accuracy in per_atomicity_analysis(filtered_scores, atomicity_list).items():
            print(f"Atomicity={atomicity}: {round(accuracy, 2)}")
        print()


def analyze_c2i_records(records):
    per_superclass = defaultdict(list)
    per_macro_domain = defaultdict(list)
    per_class = defaultdict(list)
    for record in records:
        sample_score = record.get("sample_gm", record.get("sample_am", 0.0))
        metadata = record.get("metadata", {})
        if metadata.get("superclass_name"):
            per_superclass[metadata["superclass_name"]].append(sample_score)
        if metadata.get("macro_domain"):
            per_macro_domain[metadata["macro_domain"]].append(sample_score)
        if metadata.get("class_name"):
            per_class[metadata["class_name"]].append(sample_score)

    print("Per Superclass Analysis (Soft-TIFA C2I)")
    for key, values in sorted(per_superclass.items()):
        print(f"{key}: {round(100 * average(values), 2)}")
    print()

    print("Per Macro Domain Analysis")
    for key, values in sorted(per_macro_domain.items()):
        print(f"{key}: {round(100 * average(values), 2)}")
    print()

    if per_class:
        macro_average = 100 * average([average(values) for values in per_class.values()])
        print(f"Per Class Macro Average: {round(macro_average, 2)}")
        print()


def analyze_rich_results(data):
    results = data["results"]
    grouped = defaultdict(list)
    for record in results:
        grouped[record.get("task_type", "unknown")].append(record)

    summary = data.get("summary", {})
    if summary:
        print("Summary")
        if "overall_am" in summary:
            print(f"overall_am: {round(summary['overall_am'], 2)}")
        if "overall_gm" in summary:
            print(f"overall_gm: {round(summary['overall_gm'], 2)}")
        if "joint_task_macro_avg" in summary:
            print(f"joint_task_macro_avg: {round(summary['joint_task_macro_avg'], 2)}")
        print()

    if "t2i" in grouped:
        print("T2I Analysis")
        analyze_t2i_records(grouped["t2i"])

    if "c2i" in grouped:
        print("C2I Analysis")
        analyze_c2i_records(grouped["c2i"])


def analyze_legacy(benchmark_data_path, score_data_path):
    benchmark_data, all_score_lists = load_legacy_analysis_inputs(benchmark_data_path, score_data_path)
    all_skill_lists = [record["metadata"].get("skills", [atom.get("skill", "unspecified") for atom in record["atoms"]]) for record in benchmark_data]
    atomicity_list = [record["metadata"].get("atom_count") for record in benchmark_data if record["metadata"].get("atom_count") is not None]

    for score_list, skill_list in zip(all_score_lists, all_skill_lists):
        assert len(score_list) == len(skill_list)

    print("Per Atom Type Analysis (Soft-TIFA AM)")
    for skill, accuracy in per_skill_analysis(all_score_lists, all_skill_lists).items():
        print(f"{skill}: {round(accuracy, 2)}")
    print()

    if atomicity_list:
        print("Per Atomicity Analysis (Soft-TIFA GM)")
        filtered_score_lists = [score_list for score_list, record in zip(all_score_lists, benchmark_data) if record["metadata"].get("atom_count") is not None]
        for atomicity, accuracy in per_atomicity_analysis(filtered_score_lists, atomicity_list).items():
            print(f"Atomicity={atomicity}: {round(accuracy, 2)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze T2I/C2I Soft-TIFA performance")
    parser.add_argument(
        "--benchmark_data",
        type=str,
        required=False,
        default="./geneval2_data.jsonl",
        help="Benchmark data path for legacy score list analysis",
    )
    parser.add_argument("--score_data", type=str, required=True, help="Score JSON path")
    args = parser.parse_args()

    rich_results = load_rich_results(args.score_data)
    if rich_results is not None:
        analyze_rich_results(rich_results)
    else:
        analyze_legacy(args.benchmark_data, args.score_data)


if __name__ == "__main__":
    main()
