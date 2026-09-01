# Copyright: Meta Platforms, Inc. and affiliates

import json
import argparse
from collections import defaultdict
from pathlib import Path

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


def t2i_data(records):
    score_lists = [record["atom_scores"] for record in records]
    skill_lists = [record.get("atom_skills", []) for record in records]
    per_skill = per_skill_analysis(score_lists, skill_lists)

    atomicity_list = []
    filtered_scores = []
    for record, score_list in zip(records, score_lists):
        if record.get("metadata", {}).get("atom_count") is not None:
            atomicity_list.append(record["metadata"]["atom_count"])
            filtered_scores.append(score_list)

    per_atomicity = per_atomicity_analysis(filtered_scores, atomicity_list) if atomicity_list else {}
    return {"per_skill": per_skill, "per_atomicity": per_atomicity}


def c2i_data(records):
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

    data = {
        "per_superclass": {key: 100 * average(values) for key, values in sorted(per_superclass.items())},
        "per_macro_domain": {key: 100 * average(values) for key, values in sorted(per_macro_domain.items())},
    }
    if per_class:
        data["per_class"] = {key: 100 * average(values) for key, values in per_class.items()}
        data["per_class_macro_average"] = 100 * average([average(values) for values in per_class.values()])
    return data


def legacy_t2i_data(benchmark_data_path, score_data_path):
    benchmark_data, all_score_lists = load_legacy_analysis_inputs(benchmark_data_path, score_data_path)
    all_skill_lists = [
        record["metadata"].get("skills", [atom.get("skill", "unspecified") for atom in record["atoms"]])
        for record in benchmark_data
    ]
    for score_list, skill_list in zip(all_score_lists, all_skill_lists):
        assert len(score_list) == len(skill_list)

    per_skill = per_skill_analysis(all_score_lists, all_skill_lists)
    filtered = [
        (score_list, record["metadata"]["atom_count"])
        for score_list, record in zip(all_score_lists, benchmark_data)
        if record["metadata"].get("atom_count") is not None
    ]
    filtered_score_lists = [score_list for score_list, _ in filtered]
    atomicity_list = [atomicity for _, atomicity in filtered]
    per_atomicity = per_atomicity_analysis(filtered_score_lists, atomicity_list) if atomicity_list else {}
    return {"per_skill": per_skill, "per_atomicity": per_atomicity}


def render_report(data, t2i_header):
    lines = []
    if data.get("summary"):
        lines.append("Summary")
        for key in ("overall_am", "overall_gm", "joint_task_macro_avg"):
            if key in data["summary"]:
                lines.append(f"{key}: {round(data['summary'][key], 2)}")
        lines.append("")

    t2i = data.get("t2i")
    if t2i:
        if t2i_header:
            lines.append("T2I Analysis")
        lines.append("Per Atom Type Analysis (Soft-TIFA AM)")
        for skill, accuracy in t2i["per_skill"].items():
            lines.append(f"{skill}: {round(accuracy, 2)}")
        lines.append("")
        if t2i["per_atomicity"]:
            lines.append("Per Atomicity Analysis (Soft-TIFA GM)")
            for atomicity, accuracy in t2i["per_atomicity"].items():
                lines.append(f"Atomicity={atomicity}: {round(accuracy, 2)}")
            lines.append("")

    c2i = data.get("c2i")
    if c2i:
        lines.append("C2I Analysis")
        lines.append("Per Superclass Analysis (Soft-TIFA C2I)")
        for key, accuracy in c2i["per_superclass"].items():
            lines.append(f"{key}: {round(accuracy, 2)}")
        lines.append("")
        lines.append("Per Macro Domain Analysis")
        for key, accuracy in c2i["per_macro_domain"].items():
            lines.append(f"{key}: {round(accuracy, 2)}")
        lines.append("")
        if c2i.get("per_class_macro_average") is not None:
            lines.append(f"Per Class Macro Average: {round(c2i['per_class_macro_average'], 2)}")
            lines.append("")

    return "\n".join(lines)


def save_results(save_dir, score_data_path, data, report):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(score_data_path).stem
    json_path = save_dir / f"{stem}_analysis.json"
    txt_path = save_dir / f"{stem}_analysis.txt"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(report + "\n", encoding="utf-8")
    return json_path, txt_path


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
    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        default=None,
        help="Directory to save analysis results (created if missing)",
    )
    args = parser.parse_args()

    rich_results = load_rich_results(args.score_data)
    data = {"score_data": args.score_data}
    if rich_results is not None:
        if rich_results.get("summary"):
            data["summary"] = rich_results["summary"]
        grouped = defaultdict(list)
        for record in rich_results["results"]:
            grouped[record.get("task_type", "unknown")].append(record)
        if "t2i" in grouped:
            data["t2i"] = t2i_data(grouped["t2i"])
        if "c2i" in grouped:
            data["c2i"] = c2i_data(grouped["c2i"])
        report = render_report(data, t2i_header=True)
    else:
        data["t2i"] = legacy_t2i_data(args.benchmark_data, args.score_data)
        report = render_report(data, t2i_header=False)

    print(report)

    if args.save_dir:
        json_path, txt_path = save_results(args.save_dir, args.score_data, data, report)
        print(f"\nAnalysis results saved to:\n  {json_path}\n  {txt_path}")


if __name__ == "__main__":
    main()
