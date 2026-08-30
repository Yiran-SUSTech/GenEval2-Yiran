import json
from pathlib import Path


YES_VARIANTS = ["Yes", "yes", " Yes", " yes"]
NO_VARIANTS = ["No", "no", " No", " no"]


NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def infer_answer_type(question, answer):
    answer_text = str(answer).strip()
    question_text = str(question).strip().lower()
    answer_lower = answer_text.lower()
    if question_text.startswith("how many"):
        return "count"
    if answer_lower in {"yes", "no"}:
        return "binary"
    if answer_lower in NUMBER_WORDS or answer_text.isdigit():
        return "count"
    return "string"


def atom_from_legacy(question, answer, skill=None):
    return {
        "question": question,
        "answer": answer,
        "answer_type": infer_answer_type(question, answer),
        "skill": skill or "unspecified",
        "weight": 1.0,
    }


def normalize_atoms(record):
    if "atoms" in record:
        atoms = []
        for atom in record["atoms"]:
            normalized = {
                "question": atom["question"],
                "answer": atom["answer"],
                "answer_type": atom.get("answer_type") or infer_answer_type(atom["question"], atom["answer"]),
                "skill": atom.get("skill", "unspecified"),
                "weight": float(atom.get("weight", 1.0)),
            }
            if "answer_aliases" in atom:
                normalized["answer_aliases"] = list(atom["answer_aliases"])
            atoms.append(normalized)
        return atoms

    if "vqa_list" in record:
        skills = record.get("skills", [])
        atoms = []
        for index, pair in enumerate(record["vqa_list"]):
            question, answer = pair
            skill = skills[index] if index < len(skills) else "unspecified"
            atoms.append(atom_from_legacy(question, answer, skill=skill))
        return atoms

    raise ValueError("Benchmark record must contain either 'atoms' or 'vqa_list'.")


def normalize_record(record, index):
    atoms = normalize_atoms(record)
    task_type = record.get("task_type")
    if task_type is None:
        task_type = "t2i" if "vqa_list" in record and "prompt" in record else "c2i"

    prompt = record.get("prompt")
    if prompt is None and task_type == "c2i":
        prompt = record.get("condition", {}).get("class_name")

    sample_id = record.get("sample_id") or f"{task_type}_{index:06d}"
    metadata = dict(record.get("metadata", {}))
    if "atom_count" in record and "atom_count" not in metadata:
        metadata["atom_count"] = record["atom_count"]
    if "skills" in record and "skills" not in metadata:
        metadata["skills"] = list(record["skills"])

    if "condition" in record:
        condition = record["condition"]
    elif task_type == "t2i":
        condition = {"type": "text", "text": prompt}
    else:
        condition = {"type": "class", "class_name": prompt}

    return {
        "sample_id": sample_id,
        "task_type": task_type,
        "prompt": prompt,
        "condition": condition,
        "atoms": atoms,
        "metadata": metadata,
    }


def load_benchmark_records(benchmark_path):
    benchmark_path = Path(benchmark_path)
    if benchmark_path.suffix.lower() == ".jsonl":
        records = []
        with benchmark_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                records.append(normalize_record(json.loads(line), index))
        return records

    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [normalize_record(record, index) for index, record in enumerate(data)]
    raise ValueError("Benchmark data must be JSONL or a JSON list.")


def load_image_manifest(manifest_path):
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Image filepath data must be a JSON object.")
    return data


def resolve_image_path(record, image_manifest):
    sample_id = record["sample_id"]
    if sample_id in image_manifest:
        return image_manifest[sample_id]
    prompt = record.get("prompt")
    if prompt in image_manifest:
        return image_manifest[prompt]
    raise KeyError(f"Missing filepath for sample_id={sample_id} prompt={prompt}")


def build_answer_list(atom):
    answer_type = atom.get("answer_type") or infer_answer_type(atom["question"], atom["answer"])
    answer_text = str(atom["answer"]).strip()
    answer_lower = answer_text.lower()

    if answer_type == "binary":
        return YES_VARIANTS if answer_lower == "yes" else NO_VARIANTS

    if answer_type == "count":
        variants = [answer_text, answer_text.capitalize(), f" {answer_text}", f" {answer_text.capitalize()}"]
        if answer_lower in NUMBER_WORDS:
            numeric = NUMBER_WORDS[answer_lower]
            variants.extend([numeric, f" {numeric}"])
        elif answer_text.isdigit():
            for word, number in NUMBER_WORDS.items():
                if number == answer_text:
                    variants.extend([word, word.capitalize(), f" {word}", f" {word.capitalize()}"])
                    break
        return list(dict.fromkeys(variants))

    aliases = atom.get("answer_aliases", [answer_text])
    variants = []
    for alias in aliases:
        alias = str(alias).strip()
        variants.extend([alias, alias.capitalize(), f" {alias}", f" {alias.capitalize()}"])
    return list(dict.fromkeys(variants))
