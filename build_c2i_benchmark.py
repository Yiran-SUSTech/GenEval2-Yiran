import json
import argparse
from pathlib import Path

from imagenet_superclasses import build_c2i_record


def main():
    parser = argparse.ArgumentParser(description="Build ImageNet C2I benchmark for Soft-TIFA")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--include_negative", action="store_true", help="Add one negative superclass atom per sample")
    parser.add_argument("--class_ids", type=int, nargs="*", default=None, help="Optional subset of ImageNet class ids")
    args = parser.parse_args()

    class_ids = args.class_ids if args.class_ids else list(range(1000))
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for class_id in class_ids:
            record = build_c2i_record(class_id, include_negative=args.include_negative)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
