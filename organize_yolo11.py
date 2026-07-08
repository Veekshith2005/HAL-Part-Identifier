"""
Organise CVAT YOLO 1.1 export into HAL-Part-Identifier folder structure.

CVAT export contains:
  obj.names
  obj.data
  train.txt
  obj_train_data/
      image1.jpg
      image1.txt  ...

Puts files into:
  raw_images1/<CLASS_NAME>/               <- original images grouped by class
  datasets/aerospace_parts/images/all/
  datasets/aerospace_parts/labels/all/
  datasets/aerospace_parts/dataset.yaml

Usage:
  python organize_yolo11.py  cvat_export_folder
  python organize_yolo11.py  cvat_export_folder --output datasets/aerospace_parts
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


# ──────────────────────────────────────────────────────────────────────────────
def load_class_names(cvat_dir: Path) -> list:
    names_file = cvat_dir / "obj.names"
    if not names_file.exists():
        raise FileNotFoundError(f"obj.names not found in {cvat_dir}")
    names = [l.strip() for l in names_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"  Classes ({len(names)}): {names}")
    return names


# ──────────────────────────────────────────────────────────────────────────────
def collect_pairs(cvat_dir: Path):
    data_dir = cvat_dir / "obj_train_data"
    if not data_dir.exists():
        raise FileNotFoundError(f"obj_train_data/ not found in {cvat_dir}")

    pairs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for img in data_dir.glob(ext):
            lbl = img.with_suffix(".txt")
            if lbl.exists():
                pairs.append((img, lbl))
            else:
                print(f"  ⚠  No label for {img.name} — skipped")

    if not pairs:
        raise RuntimeError(f"No image+label pairs found in {data_dir}")
    print(f"  Image-label pairs found: {len(pairs)}")
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
def copy_to_dataset(pairs, class_names, output_dir: Path, train_split: float):
    img_all = output_dir / "images" / "all"
    lbl_all = output_dir / "labels" / "all"
    img_all.mkdir(parents=True, exist_ok=True)
    lbl_all.mkdir(parents=True, exist_ok=True)

    # Copy everything to /all/
    for img, lbl in pairs:
        shutil.copy2(img, img_all / img.name)
        shutil.copy2(lbl, lbl_all / lbl.name)

    # ── Train / Val split ──────────────────────────────────────────────────
    random.seed(42)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * train_split)
    splits  = {"train": shuffled[:n_train], "val": shuffled[n_train:]}

    for split, files in splits.items():
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, lbl in files:
            shutil.copy2(img, output_dir / "images" / split / img.name)
            shutil.copy2(lbl, output_dir / "labels" / split / lbl.name)
        print(f"  {split:5s}: {len(files)} images")

    # ── dataset.yaml ──────────────────────────────────────────────────────
    yaml_path = output_dir / "dataset.yaml"
    content = {
        "path":  str(output_dir.resolve().as_posix()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(class_names),
        "names": class_names,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False,
                  sort_keys=False, allow_unicode=True)
    print(f"  dataset.yaml written → {yaml_path}")

    return splits


# ──────────────────────────────────────────────────────────────────────────────
def copy_to_raw(pairs, class_names, raw_dir: Path):
    """
    Group original images into raw_images1/<CLASS_NAME>/ folders.
    Uses the first class_id found in the label file as the folder name.
    Matches your existing raw_images1/ folder used by core.get_reference_image().
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    class_counts = defaultdict(int)

    for img, lbl in pairs:
        lines = lbl.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            continue
        first_class_id = int(lines[0].split()[0])
        if first_class_id >= len(class_names):
            print(f"  ⚠  Unknown class id {first_class_id} in {lbl.name} — skipped")
            continue

        class_name = class_names[first_class_id].replace(" ", "_")
        dest_dir   = raw_dir / class_name
        dest_dir.mkdir(exist_ok=True)
        shutil.copy2(img, dest_dir / img.name)
        class_counts[class_name] += 1

    print("  Raw images per class:")
    for cls, cnt in sorted(class_counts.items()):
        print(f"    {cls}: {cnt}")


# ──────────────────────────────────────────────────────────────────────────────
def print_summary(output_dir: Path, raw_dir: Path, splits: dict, class_names: list):
    print()
    print("=" * 60)
    print("  ✅  DATASET ORGANISED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Classes : {len(class_names)}")
    print(f"  Train   : {len(splits['train'])} images")
    print(f"  Val     : {len(splits['val'])} images")
    print()
    print("  Folder layout:")
    print(f"  ├── {raw_dir}/              ← originals by class (used by Flask app)")
    print(f"  └── {output_dir}/")
    print(f"      ├── dataset.yaml        ← used by train.py")
    print(f"      ├── images/train|val|all/")
    print(f"      └── labels/train|val|all/")
    print()
    print("  Next steps (run from retrain/ project folder):")
    print("  1.  python generate_synthetic.py")
    print("  2.  python prepare_dataset.py")
    print("  3.  python train.py")
    print()
    print("  Then copy trained_models/retrained_v2.pt into")
    print("  HAL-Part-Identifier/trained_models/ and update")
    print("  PART_MODEL_PATH in core.py")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Organise CVAT YOLO 1.1 export for HAL-Part-Identifier"
    )
    parser.add_argument(
        "cvat_dir",
        help="Path to CVAT export folder (contains obj.names + obj_train_data/)"
    )
    parser.add_argument(
        "--output",
        default="datasets/aerospace_parts",
        help="Dataset output dir (default: datasets/aerospace_parts)"
    )
    parser.add_argument(
        "--raw",
        default="raw_images1",          # ← your project folder name
        help="Raw images output dir (default: raw_images1)"
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.85,                   # ← 85/15 split matching prepare_dataset.py
        help="Train split ratio (default: 0.85)"
    )
    args = parser.parse_args()

    cvat_dir   = Path(args.cvat_dir)
    output_dir = Path(args.output)
    raw_dir    = Path(args.raw)

    print("=" * 60)
    print("  CVAT YOLO 1.1 → HAL-Part-Identifier organiser")
    print("=" * 60)
    print(f"  Source : {cvat_dir.resolve()}")
    print(f"  Dataset: {output_dir.resolve()}")
    print(f"  Raw    : {raw_dir.resolve()}")
    print()

    print("Step 1: Reading class names…")
    class_names = load_class_names(cvat_dir)

    print("Step 2: Collecting image-label pairs…")
    pairs = collect_pairs(cvat_dir)

    print("Step 3: Copying to dataset structure…")
    splits = copy_to_dataset(pairs, class_names, output_dir, args.train)

    print("Step 4: Copying originals to raw_images1/<CLASS>/…")
    copy_to_raw(pairs, class_names, raw_dir)

    print_summary(output_dir, raw_dir, splits, class_names)


if __name__ == "__main__":
    main()