import argparse
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Verify sampled benchmark images: count, missing indices, corrupt files")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--expected_count", type=int, default=800)
    parser.add_argument("--delete", action="store_true", help="Delete corrupt files after reporting")
    args = parser.parse_args()

    directory = Path(args.image_dir)
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    files = {}
    unnamed = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() != ".png":
            continue
        if path.stem.isdigit():
            files[int(path.stem)] = path
        else:
            unnamed.append(path.name)

    missing = [i for i in range(args.expected_count) if i not in files]
    out_of_range = sorted(i for i in files if i < 0 or i >= args.expected_count)

    corrupt = []
    for index in sorted(files):
        try:
            with Image.open(files[index]) as image:
                image.load()
        except Exception as e:
            corrupt.append((index, f"{type(e).__name__}: {e}"))

    print(f"Directory: {directory}")
    print(f"PNG files with numeric names: {len(files)}")
    if unnamed:
        print(f"Non-numeric filenames (ignored): {len(unnamed)} — e.g. {unnamed[:3]}")
    if missing:
        shown = missing[:20]
        more = f" ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        print(f"MISSING indices ({len(missing)}): {shown}{more}")
    else:
        print(f"No missing indices in [0, {args.expected_count})")
    if out_of_range:
        print(f"Out-of-range indices: {out_of_range}")
    if corrupt:
        print(f"CORRUPT/unreadable ({len(corrupt)}):")
        for index, error in corrupt[:20]:
            print(f"  {index}.png: {error}")
        if len(corrupt) > 20:
            print(f"  ... and {len(corrupt) - 20} more")
    else:
        print("No corrupt files")

    if args.delete and corrupt:
        for index, _ in corrupt:
            files[index].unlink()
        print(f"Deleted {len(corrupt)} corrupt file(s) — rerun sampling to regenerate them")

    if not missing and not corrupt and not out_of_range and not unnamed:
        print("OK: image set complete and readable")
    else:
        print("INCOMPLETE: fix the issues above, then rerun sampling to fill the gaps")


if __name__ == "__main__":
    main()
