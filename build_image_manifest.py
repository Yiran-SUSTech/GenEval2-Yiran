import argparse
import json
import re
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def extract_index(path):
    runs = re.findall(r"\d+", path.stem)
    if not runs:
        return None
    # zero-padded index runs are typically the longest digit group in a filename
    best = max(range(len(runs)), key=lambda i: (len(runs[i]), i))
    return int(runs[best])


def build_path_map(images, num_prompts, numbering):
    index_to_path = {}
    for image in images:
        index = extract_index(image)
        if index is None:
            continue
        if index in index_to_path:
            raise SystemExit(f"Duplicate image index {index}: {index_to_path[index]} vs {image}")
        index_to_path[index] = image

    if index_to_path:
        if numbering == "auto":
            zero_hits = sum(1 for i in range(num_prompts) if i in index_to_path)
            one_hits = sum(1 for i in range(num_prompts) if i + 1 in index_to_path)
            offset = 0 if zero_hits >= one_hits else 1
        else:
            offset = {"zero": 0, "one": 1}[numbering]
        print(f"Using {'1-based' if offset else '0-based'} image numbering")
        return {i: index_to_path.get(i + offset) for i in range(num_prompts)}

    if len(images) == num_prompts:
        print("No digits found in filenames; falling back to natural sort order")
        return dict(enumerate(images))

    raise SystemExit(f"Cannot map {len(images)} images to {num_prompts} prompts without numeric filenames")


def main():
    parser = argparse.ArgumentParser(description="Build the GenEval2 image manifest (sample_id -> image path)")
    parser.add_argument("--benchmark_data", default="geneval2_data.jsonl")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--numbering", choices=["auto", "zero", "one"], default="auto")
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_data)
    prompts = [json.loads(line) for line in benchmark_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    image_dir = Path(args.image_dir)
    images = sorted(
        (p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=natural_key,
    )

    path_map = build_path_map(images, len(prompts), args.numbering)

    manifest = {}
    missing = []
    for i, record in enumerate(prompts):
        sample_id = record.get("sample_id") or f"t2i_{i:06d}"
        image = path_map.get(i)
        if image is None:
            missing.append(i)
            continue
        manifest[sample_id] = str(image.resolve())

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Prompts: {len(prompts)} | images found: {len(images)} | matched: {len(manifest)} | missing: {len(missing)}")
    if missing:
        preview = ", ".join(map(str, missing[:20]))
        print(f"Missing prompt indices (0-based): {preview}")
    for sample_id in list(manifest)[:3]:
        print(f"  {sample_id} -> {manifest[sample_id]}")
    print(f"Manifest written to {output_path}")


if __name__ == "__main__":
    main()
