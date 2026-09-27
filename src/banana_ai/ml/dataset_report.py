from pathlib import Path
import argparse
from collections import Counter
from PIL import Image


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def report(root: str):
    root_path = Path(root)

    if not root_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root_path}")

    counts = Counter()
    bad = []
    sizes = []

    for class_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        for path in class_dir.rglob("*"):
            if path.suffix.lower() not in SUPPORTED:
                continue
            counts[class_dir.name] += 1
            try:
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    sizes.append(img.size)
            except Exception as exc:
                bad.append((str(path), str(exc)))

    print("\n[Banana] Dataset report")
    print("=" * 40)
    total = sum(counts.values())

    for name, count in sorted(counts.items()):
        pct = 100 * count / total if total else 0
        print(f"{name:12s} {count:6d} ({pct:5.1f}%)")

    print(f"\nTotal images: {total}")
    if sizes:
        print(f"Example image size: {sizes[0]}")
    print(f"Unreadable images: {len(bad)}")

    if bad:
        print("\nFirst unreadable files:")
        for path, error in bad[:10]:
            print(f"- {path}: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report BananaAI dataset contents.")
    parser.add_argument(
        "--dataset-root",
        default="data",
        help="Directory containing train/, valid/ (or val/), and test/ folders.",
    )
    args = parser.parse_args()
    root = Path(args.dataset_root)
    report(root / "train")
    for split in ("valid", "val", "test"):
        split_path = root / split
        if split_path.is_dir():
            print()
            report(split_path)
