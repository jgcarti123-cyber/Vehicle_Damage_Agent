"""Stage 1 dataset: download CarDD from Roboflow, validate, and verify splits.

Run directly:
    python stage1_detection/dataset.py download   # pull from Roboflow
    python stage1_detection/dataset.py validate    # check image/label pairing
    python stage1_detection/dataset.py split        # stratified 80/10/10 (if needed)

============================================================================
  WHERE TO PUT YOUR ROBOFLOW API KEY
============================================================================
Preferred (spec says "no hardcoded secrets"): put it in a .env file at the
project root:

    ROBOFLOW_API_KEY=your_key_here

If you'd rather hardcode it (as you asked), paste it into the
`ROBOFLOW_API_KEY_HARDCODED` constant just below. The env var wins if both
are set.
============================================================================
"""

from __future__ import annotations

import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

# --- Optional .env support -------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv not installed yet
    pass

import os

# ===========================================================================
#  Roboflow API key is read from .env (ROBOFLOW_API_KEY) — see .env / .env.example.
#  Keep this empty so no secret lives in source control.
ROBOFLOW_API_KEY_HARDCODED = ""
# ===========================================================================

# --- Roboflow dataset location ----------------------------------------------
# From the dataset's "Download Dataset" -> "show download code" panel:
#   rf.workspace("sindhu").project("car_dent_scratch_detection-1").version(9)
ROBOFLOW_WORKSPACE = "sindhu"
ROBOFLOW_PROJECT = "car_dent_scratch_detection-1"
ROBOFLOW_VERSION = 9
ROBOFLOW_FORMAT = "yolov11"

# --- Class merge: 17 location-based source classes -> 4 damage-type buckets --
# The Roboflow dataset labels damage by LOCATION (front-bumper-dent, roof-dent,
# Headlight-Damage, ...). We collapse these into 4 damage TYPE classes that feed
# Stage 2/3 cleanly. See CLAUDE.md "Key Design Decisions".
SOURCE_CLASS_NAMES = [
    "Bodypanel-Dent", "Front-Windscreen-Damage", "Headlight-Damage",
    "Rear-windscreen-Damage", "RunningBoard-Dent", "Sidemirror-Damage",
    "Signlight-Damage", "Taillight-Damage", "bonnet-dent", "boot-dent",
    "doorouter-dent", "fender-dent", "front-bumper-dent", "pillar-dent",
    "quaterpanel-dent", "rear-bumper-dent", "roof-dent",
]
# Target taxonomy (new class ids):
TARGET_CLASS_NAMES = ["dent", "glass_damage", "light_damage", "mirror_damage"]
TARGET_ID = {name: i for i, name in enumerate(TARGET_CLASS_NAMES)}

# old source class id -> new target class id
CLASS_MERGE_MAP: dict[int, int] = {
    0: TARGET_ID["dent"],          # Bodypanel-Dent
    1: TARGET_ID["glass_damage"],  # Front-Windscreen-Damage
    2: TARGET_ID["light_damage"],  # Headlight-Damage
    3: TARGET_ID["glass_damage"],  # Rear-windscreen-Damage
    4: TARGET_ID["dent"],          # RunningBoard-Dent
    5: TARGET_ID["mirror_damage"], # Sidemirror-Damage
    6: TARGET_ID["light_damage"],  # Signlight-Damage
    7: TARGET_ID["light_damage"],  # Taillight-Damage
    8: TARGET_ID["dent"],          # bonnet-dent
    9: TARGET_ID["dent"],          # boot-dent
    10: TARGET_ID["dent"],         # doorouter-dent
    11: TARGET_ID["dent"],         # fender-dent
    12: TARGET_ID["dent"],         # front-bumper-dent
    13: TARGET_ID["dent"],         # pillar-dent
    14: TARGET_ID["dent"],         # quaterpanel-dent
    15: TARGET_ID["dent"],         # rear-bumper-dent
    16: TARGET_ID["dent"],         # roof-dent
}

# Roboflow ships these source splits; we pool them all then re-split 80/10/10.
RAW_SOURCE_SPLITS = ("train", "valid", "test")

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"
RAW_DATASET_DIR = DATA_RAW / ROBOFLOW_PROJECT

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = (0.8, 0.1, 0.1)
SEED = 42


def _resolve_api_key() -> str:
    """Return the Roboflow API key from env or the hardcoded constant."""
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip() or ROBOFLOW_API_KEY_HARDCODED.strip()
    if not key:
        raise RuntimeError(
            "No Roboflow API key found. Set ROBOFLOW_API_KEY in .env or paste it "
            "into ROBOFLOW_API_KEY_HARDCODED at the top of dataset.py."
        )
    return key


def download() -> Path:
    """Download the dataset from Roboflow into data/raw/. Returns the dataset dir."""
    from roboflow import Roboflow

    api_key = _resolve_api_key()
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    dataset = version.download(ROBOFLOW_FORMAT, location=str(DATA_RAW / ROBOFLOW_PROJECT))

    location = Path(dataset.location)
    print(f"Downloaded CarDD to: {location}")
    return location


def _pair_images_labels(images_dir: Path, labels_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return (orphans, valid) image paths relative to images_dir.

    An image is valid if a matching .txt label exists. Orphans have no label.
    """
    orphans, valid = [], []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        label = labels_dir / f"{img.stem}.txt"
        (valid if label.exists() else orphans).append(img)
    return orphans, valid


def _read_class_ids(label_path: Path) -> list[int]:
    """Parse YOLO label file -> list of integer class ids (one per box)."""
    ids: list[int] = []
    text = label_path.read_text().strip()
    if not text:
        return ids
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5:
            ids.append(int(float(parts[0])))
    return ids


def validate() -> bool:
    """Validate that each split has paired images/labels and report class counts."""
    ok = True
    for split in SPLITS:
        images_dir = DATA_SPLITS / split / "images"
        labels_dir = DATA_SPLITS / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            print(f"[{split}] MISSING images/ or labels/ dir")
            ok = False
            continue

        orphans, valid = _pair_images_labels(images_dir, labels_dir)
        class_counts: Counter[int] = Counter()
        empty = 0
        for img in valid:
            ids = _read_class_ids(labels_dir / f"{img.stem}.txt")
            if not ids:
                empty += 1
            class_counts.update(ids)

        print(f"[{split}] images={len(valid)} orphans(no-label)={len(orphans)} "
              f"empty-labels={empty}")
        print(f"[{split}] class distribution: {dict(sorted(class_counts.items()))}")
        if orphans:
            ok = False
    if ok:
        print("VALIDATION OK")
    else:
        print("VALIDATION FOUND ISSUES (see above)")
    return ok


def split(source_images: Path, source_labels: Path) -> None:
    """Stratified 80/10/10 split from a flat source into data/splits/.

    Use this only if the Roboflow export is NOT already split. Stratifies on the
    *first* class id in each label so every class appears in every split.
    """
    random.seed(SEED)
    _, valid = _pair_images_labels(source_images, source_labels)

    by_class: dict[int, list[Path]] = defaultdict(list)
    for img in valid:
        ids = _read_class_ids(source_labels / f"{img.stem}.txt")
        primary = ids[0] if ids else -1
        by_class[primary].append(img)

    assignments: dict[Path, str] = {}
    for cls, imgs in by_class.items():
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * SPLIT_RATIOS[0])
        n_val = int(n * SPLIT_RATIOS[1])
        for i, img in enumerate(imgs):
            if i < n_train:
                assignments[img] = "train"
            elif i < n_train + n_val:
                assignments[img] = "val"
            else:
                assignments[img] = "test"

    for split_name in SPLITS:
        (DATA_SPLITS / split_name / "images").mkdir(parents=True, exist_ok=True)
        (DATA_SPLITS / split_name / "labels").mkdir(parents=True, exist_ok=True)

    for img, split_name in assignments.items():
        shutil.copy2(img, DATA_SPLITS / split_name / "images" / img.name)
        shutil.copy2(
            source_labels / f"{img.stem}.txt",
            DATA_SPLITS / split_name / "labels" / f"{img.stem}.txt",
        )
    print(f"Split {len(assignments)} images into {SPLIT_RATIOS} train/val/test")


def _remap_label_text(raw_text: str) -> str:
    """Rewrite a YOLO label file's class ids via CLASS_MERGE_MAP.

    Drops any box whose source class id is not in the merge map. Keeps the
    geometry columns untouched.
    """
    out_lines: list[str] = []
    for line in raw_text.strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        src_id = int(float(parts[0]))
        if src_id not in CLASS_MERGE_MAP:
            continue
        parts[0] = str(CLASS_MERGE_MAP[src_id])
        out_lines.append(" ".join(parts))
    return "\n".join(out_lines)


def prepare() -> None:
    """Pool the raw Roboflow splits, remap 17->4 classes, re-split 80/10/10.

    Stratifies each image by the *rarest* target class it contains (by global
    frequency) so minority classes (e.g. mirror_damage) are spread across splits.
    Writes remapped labels + copied images into data/splits/.
    """
    random.seed(SEED)
    if not RAW_DATASET_DIR.exists():
        raise RuntimeError(f"Raw dataset not found at {RAW_DATASET_DIR}. Run 'download' first.")

    # 1. Collect every (image, remapped-label-text) pair across raw splits.
    pairs: list[tuple[Path, str]] = []
    global_counts: Counter[int] = Counter()
    for src in RAW_SOURCE_SPLITS:
        images_dir = RAW_DATASET_DIR / src / "images"
        labels_dir = RAW_DATASET_DIR / src / "labels"
        if not images_dir.exists():
            continue
        _, valid = _pair_images_labels(images_dir, labels_dir)
        for img in valid:
            remapped = _remap_label_text((labels_dir / f"{img.stem}.txt").read_text())
            if not remapped:
                continue  # skip images with no boxes after remap
            pairs.append((img, remapped))
            for line in remapped.splitlines():
                global_counts[int(line.split()[0])] += 1

    print(f"Pooled {len(pairs)} labelled images; merged class counts: "
          f"{ {TARGET_CLASS_NAMES[k]: v for k, v in sorted(global_counts.items())} }")

    # 2. Stratify key = rarest target class present in the image.
    def stratify_key(text: str) -> int:
        ids = {int(l.split()[0]) for l in text.splitlines()}
        return min(ids, key=lambda c: global_counts[c])

    by_key: dict[int, list[tuple[Path, str]]] = defaultdict(list)
    for img, text in pairs:
        by_key[stratify_key(text)].append((img, text))

    # 3. Fresh split dirs.
    for split_name in SPLITS:
        for sub in ("images", "labels"):
            d = DATA_SPLITS / split_name / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    # 4. Per-stratum 80/10/10 assignment.
    counts = {s: 0 for s in SPLITS}
    for _, group in by_key.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * SPLIT_RATIOS[0])
        n_val = int(n * SPLIT_RATIOS[1])
        for i, (img, text) in enumerate(group):
            split_name = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            shutil.copy2(img, DATA_SPLITS / split_name / "images" / img.name)
            (DATA_SPLITS / split_name / "labels" / f"{img.stem}.txt").write_text(text + "\n")
            counts[split_name] += 1
    print(f"Wrote splits -> {counts}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if cmd == "download":
        download()
    elif cmd == "prepare":
        prepare()
    elif cmd == "validate":
        validate()
    elif cmd == "split":
        if len(sys.argv) < 4:
            print("usage: dataset.py split <source_images_dir> <source_labels_dir>")
            sys.exit(1)
        split(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        print(f"unknown command: {cmd!r} (use download | prepare | validate | split)")
        sys.exit(1)


if __name__ == "__main__":
    main()
