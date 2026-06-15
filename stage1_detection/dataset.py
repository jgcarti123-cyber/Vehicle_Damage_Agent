"""Stage 1 dataset: download one or more Roboflow/Kaggle datasets, remap classes,
pool images, and produce a stratified 80/10/10 train/val/test split.

Run directly:
    python stage1_detection/dataset.py download   # pull all configured datasets
    python stage1_detection/dataset.py prepare    # pool + remap + stratified split
    python stage1_detection/dataset.py validate   # check image/label pairing
    python stage1_detection/dataset.py split <img_dir> <lbl_dir>  # manual split

============================================================================
  SECRETS
============================================================================
Preferred: put your Roboflow API key in a .env file at the project root:

    ROBOFLOW_API_KEY=your_key_here

Fallback: paste it into ROBOFLOW_API_KEY_HARDCODED below (not committed).

Kaggle datasets (workspace=None) are not downloaded by this script — fetch
them manually (kagglehub / Kaggle UI) into data/raw/<name>/ first.
============================================================================
"""

from __future__ import annotations

import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict

# --- Optional .env support -------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os

# ===========================================================================
#  API key — env var wins; hardcoded is fallback only.
ROBOFLOW_API_KEY_HARDCODED = ""
# ===========================================================================


# ---------------------------------------------------------------------------
# TARGET TAXONOMY  (shared across all source datasets)
# ---------------------------------------------------------------------------
TARGET_CLASS_NAMES = ["dent", "scratch", "glass_damage", "light_damage", "mirror_damage"]
TARGET_ID = {name: i for i, name in enumerate(TARGET_CLASS_NAMES)}

# Shorthand for readability in the per-dataset remaps below
_D = TARGET_ID["dent"]
_S = TARGET_ID["scratch"]
_G = TARGET_ID["glass_damage"]
_L = TARGET_ID["light_damage"]
_M = TARGET_ID["mirror_damage"]


# ---------------------------------------------------------------------------
# DATASET REGISTRY
# ---------------------------------------------------------------------------
# Each entry describes one dataset (Roboflow or pre-downloaded/Kaggle).
#
#   name        : short slug used for raw download dir (data/raw/<name>/) and
#                 as filename prefix (prevents collisions across datasets)
#   workspace   : Roboflow workspace slug, or None if pre-downloaded (Kaggle etc.)
#   project     : Roboflow project slug (ignored if workspace is None)
#   version     : version number (int, ignored if workspace is None)
#   format      : Roboflow export format (yolov11 is compatible with Ultralytics)
#   source_classes : list of class names IN ORDER (index == source class id)
#   class_remap : {source_class_id -> target_class_id}
#                 Omit a source id to DROP that class entirely.
#   raw_splits  : (optional) names of the split subdirs under data/raw/<name>/
#                 that contain images/ + labels/. Defaults to RAW_SOURCE_SPLITS
#                 (Roboflow's "train"/"valid"/"test"). Override for datasets
#                 that use different names (e.g. CarDD uses "val" not "valid").
#
# To add a new dataset:
#   1. Find it on Roboflow Universe (or download manually for Kaggle etc.)
#   2. Note its ACTUAL class list (check data.yaml after downloading — doc
#      assumptions about class lists/versions are frequently wrong)
#   3. Map each class name to one of: _D, _S, _G, _L, _M  (or omit to drop)
#   4. Append an entry here — the rest (download / pool / split) is automatic.

class DatasetConfig(TypedDict, total=False):
    name: str
    workspace: str | None
    project: str
    version: int
    format: str
    source_classes: list[str]
    class_remap: dict[int, int]
    raw_splits: tuple[str, ...]


DATASETS: list[DatasetConfig] = [
    # ------------------------------------------------------------------
    # Dataset 1: Car Dent & Scratch Detection — Sindhu (Roboflow)
    # https://universe.roboflow.com/sindhu/car_dent_scratch_detection-1
    # 17 location-specific classes → dent/glass_damage/light_damage/mirror_damage
    # (no scratch contribution from this dataset)
    # ------------------------------------------------------------------
    {
        "name": "sindhu_v9",
        "workspace": "sindhu",
        "project": "car_dent_scratch_detection-1",
        "version": 9,
        "format": "yolov11",
        "source_classes": [
            "Bodypanel-Dent",           # 0
            "Front-Windscreen-Damage",  # 1
            "Headlight-Damage",         # 2
            "Rear-windscreen-Damage",   # 3
            "RunningBoard-Dent",        # 4
            "Sidemirror-Damage",        # 5
            "Signlight-Damage",         # 6
            "Taillight-Damage",         # 7
            "bonnet-dent",              # 8
            "boot-dent",                # 9
            "doorouter-dent",           # 10
            "fender-dent",              # 11
            "front-bumper-dent",        # 12
            "pillar-dent",              # 13
            "quaterpanel-dent",         # 14
            "rear-bumper-dent",         # 15
            "roof-dent",                # 16
        ],
        "class_remap": {
            0:  _D,   # Bodypanel-Dent
            1:  _G,   # Front-Windscreen-Damage
            2:  _L,   # Headlight-Damage
            3:  _G,   # Rear-windscreen-Damage
            4:  _D,   # RunningBoard-Dent
            5:  _M,   # Sidemirror-Damage
            6:  _L,   # Signlight-Damage
            7:  _L,   # Taillight-Damage
            8:  _D,   # bonnet-dent
            9:  _D,   # boot-dent
            10: _D,   # doorouter-dent
            11: _D,   # fender-dent
            12: _D,   # front-bumper-dent
            13: _D,   # pillar-dent
            14: _D,   # quaterpanel-dent
            15: _D,   # rear-bumper-dent
            16: _D,   # roof-dent
        },
    },

    # ------------------------------------------------------------------
    # Dataset 2: CarDD (Kaggle, pre-downloaded — gabrielfcarvalho/
    # cardd-with-yolo-annotations-images-labels)
    # https://www.kaggle.com/datasets/gabrielfcarvalho/cardd-with-yolo-annotations-images-labels
    # nc=6: dent, scratch, crack, glass shatter, lamp broken, tire flat
    # PRIMARY source for `scratch` (3595 boxes). `crack` merges into `dent`
    # (same body-shop repair pathway). `tire flat` is dropped (not collision
    # damage / no equivalent class).
    # ------------------------------------------------------------------
    {
        "name": "cardd_yolo",
        "workspace": None,   # pre-downloaded manually into data/raw/cardd_yolo/
        "source_classes": [
            "dent",           # 0
            "scratch",        # 1
            "crack",          # 2
            "glass shatter",  # 3
            "lamp broken",    # 4
            "tire flat",      # 5
        ],
        "class_remap": {
            0: _D,   # dent          → dent
            1: _S,   # scratch       → scratch
            2: _D,   # crack         → dent (body surface crack, same repair category)
            3: _G,   # glass shatter → glass_damage
            4: _L,   # lamp broken   → light_damage
            # 5: tire flat → DROP
        },
        "raw_splits": ("train", "val", "test"),   # CarDD uses "val", not "valid"
    },

    # ------------------------------------------------------------------
    # Dataset 3: nivethetha/car-damages-godhu v1 (Roboflow)
    # https://universe.roboflow.com/nivethetha/car-damages-godhu
    # nc=10. Tiny (64 images, 17 surviving boxes) — minor dent/glass diversity.
    # No scratch or mirror instances in this version despite class names.
    # ------------------------------------------------------------------
    {
        "name": "nivethetha_v1",
        "workspace": "nivethetha",
        "project": "car-damages-godhu",
        "version": 1,
        "format": "yolov11",
        "source_classes": [
            "bonnet-dent",        # 0
            "bumper-dent",        # 1
            "bumper-scratch",     # 2
            "door-crack",         # 3
            "door-dent",          # 4
            "door-scratch",       # 5
            "headlight-broken",   # 6
            "roof-crushed",       # 7
            "windshield-broken",  # 8
            "windshield-scratch", # 9
        ],
        "class_remap": {
            0: _D,   # bonnet-dent        → dent
            1: _D,   # bumper-dent        → dent
            2: _S,   # bumper-scratch     → scratch
            3: _D,   # door-crack         → dent
            4: _D,   # door-dent          → dent
            5: _S,   # door-scratch       → scratch
            6: _L,   # headlight-broken   → light_damage
            7: _D,   # roof-crushed       → dent
            8: _G,   # windshield-broken  → glass_damage
            9: _S,   # windshield-scratch → scratch
        },
    },

    # ------------------------------------------------------------------
    # Dataset 4: damagedetection-hloj4/damagelocation v7 (Roboflow)
    # https://universe.roboflow.com/damagedetection-hloj4/damagelocation
    # nc=1: ["Dent"]. Adds ~53 extra dent boxes for annotation diversity.
    # ------------------------------------------------------------------
    {
        "name": "damagelocation_v7",
        "workspace": "damagedetection-hloj4",
        "project": "damagelocation",
        "version": 7,
        "format": "yolov11",
        "source_classes": [
            "Dent",  # 0
        ],
        "class_remap": {
            0: _D,   # Dent → dent
        },
    },
]


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_SPLITS    = PROJECT_ROOT / "data" / "splits"
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS         = ("train", "val", "test")
SPLIT_RATIOS   = (0.8, 0.1, 0.1)
RAW_SOURCE_SPLITS = ("train", "valid", "test")   # Roboflow's default split names
SEED           = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip() or ROBOFLOW_API_KEY_HARDCODED.strip()
    if not key:
        raise RuntimeError(
            "No Roboflow API key found. Set ROBOFLOW_API_KEY in .env "
            "or paste it into ROBOFLOW_API_KEY_HARDCODED in dataset.py."
        )
    return key


def _raw_dir(ds: DatasetConfig) -> Path:
    """Return the local directory where a dataset's raw download lives."""
    return DATA_RAW / ds["name"]


def _pair_images_labels(images_dir: Path, labels_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return (orphans_without_label, valid_paired) image paths."""
    orphans, valid = [], []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        label = labels_dir / f"{img.stem}.txt"
        (valid if label.exists() else orphans).append(img)
    return orphans, valid


def _read_class_ids(label_path: Path) -> list[int]:
    """Parse a YOLO label file → list of integer class ids."""
    ids: list[int] = []
    text = label_path.read_text().strip()
    if not text:
        return ids
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5:
            ids.append(int(float(parts[0])))
    return ids


def _remap_label_text(raw_text: str, class_remap: dict[int, int]) -> str:
    """Rewrite YOLO label class ids via class_remap; drops unmapped classes."""
    out_lines: list[str] = []
    for line in raw_text.strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        src_id = int(float(parts[0]))
        if src_id not in class_remap:
            continue
        parts[0] = str(class_remap[src_id])
        out_lines.append(" ".join(parts))
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Public commands
# ---------------------------------------------------------------------------

def download() -> None:
    """Download every Roboflow dataset in DATASETS into data/raw/<name>/.

    Datasets with workspace=None (e.g. Kaggle) are skipped — fetch those
    manually into data/raw/<name>/ before running 'prepare'.
    """
    from roboflow import Roboflow

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    rf: "Roboflow | None" = None

    for ds in DATASETS:
        if ds.get("workspace") is None:
            dest = _raw_dir(ds)
            status = "found" if dest.exists() else "MISSING — download manually"
            print(f"[{ds['name']}] Not a Roboflow dataset — {status} at {dest}")
            continue

        dest = _raw_dir(ds)
        if dest.exists():
            print(f"[{ds['name']}] Already downloaded at {dest} — skipping. "
                  f"Delete the folder to re-download.")
            continue

        if rf is None:
            rf = Roboflow(api_key=_resolve_api_key())

        print(f"[{ds['name']}] Downloading {ds['workspace']}/{ds['project']} v{ds['version']} ...")
        project = rf.workspace(ds["workspace"]).project(ds["project"])
        version = project.version(ds["version"])
        version.download(ds["format"], location=str(dest))
        print(f"[{ds['name']}] Saved to {dest}")


def prepare() -> None:
    """Pool all datasets, remap classes, stratified re-split 80/10/10.

    Images from different datasets are prefixed with the dataset name to
    avoid filename collisions (e.g. 'sindhu_v9__img001.jpg').
    Stratification key: rarest target class present in the image, by global
    frequency across the entire pooled set.
    """
    random.seed(SEED)

    # 1. Collect all (image_path, remapped_label_text, dataset_name) triples.
    pairs: list[tuple[Path, str, str]] = []
    global_counts: Counter[int] = Counter()

    for ds in DATASETS:
        raw_dir = _raw_dir(ds)
        if not raw_dir.exists():
            raise RuntimeError(
                f"[{ds['name']}] Raw data not found at {raw_dir}. "
                f"Run 'python stage1_detection/dataset.py download' first "
                f"(or download manually if workspace is None)."
            )
        ds_count = 0
        for src_split in ds.get("raw_splits", RAW_SOURCE_SPLITS):
            images_dir = raw_dir / src_split / "images"
            labels_dir = raw_dir / src_split / "labels"
            if not images_dir.exists():
                continue
            _, valid = _pair_images_labels(images_dir, labels_dir)
            for img in valid:
                raw_text = (labels_dir / f"{img.stem}.txt").read_text()
                remapped = _remap_label_text(raw_text, ds["class_remap"])
                if not remapped:
                    continue   # no boxes survive the remap
                pairs.append((img, remapped, ds["name"]))
                for line in remapped.splitlines():
                    global_counts[int(line.split()[0])] += 1
                ds_count += 1
        print(f"[{ds['name']}] Contributed {ds_count} labelled images")

    print(f"\nPooled {len(pairs)} images total")
    print(f"Merged class counts: "
          f"{ {TARGET_CLASS_NAMES[k]: v for k, v in sorted(global_counts.items())} }\n")

    # 2. Stratify on rarest target class present.
    def stratify_key(text: str) -> int:
        ids = {int(line.split()[0]) for line in text.splitlines()}
        return min(ids, key=lambda c: global_counts[c])

    by_key: dict[int, list[tuple[Path, str, str]]] = defaultdict(list)
    for img, text, ds_name in pairs:
        by_key[stratify_key(text)].append((img, text, ds_name))

    # 3. Fresh split dirs.
    for split_name in SPLITS:
        for sub in ("images", "labels"):
            d = DATA_SPLITS / split_name / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    # 4. 80/10/10 per stratum.
    counts = {s: 0 for s in SPLITS}
    for _, group in sorted(by_key.items()):
        random.shuffle(group)
        n = len(group)
        n_train = int(n * SPLIT_RATIOS[0])
        n_val   = int(n * SPLIT_RATIOS[1])
        for i, (img, text, ds_name) in enumerate(group):
            split_name = ("train" if i < n_train
                          else "val" if i < n_train + n_val
                          else "test")
            # Prefix filename with dataset name to avoid cross-dataset collisions.
            safe_stem = f"{ds_name}__{img.stem}"
            shutil.copy2(img, DATA_SPLITS / split_name / "images" / f"{safe_stem}{img.suffix}")
            (DATA_SPLITS / split_name / "labels" / f"{safe_stem}.txt").write_text(text + "\n")
            counts[split_name] += 1

    print(f"Wrote splits → {counts}")
    print(f"  train : {counts['train']}")
    print(f"  val   : {counts['val']}")
    print(f"  test  : {counts['test']}")


def validate() -> bool:
    """Validate split dirs: check image/label pairing and report class counts."""
    ok = True
    for split_name in SPLITS:
        images_dir = DATA_SPLITS / split_name / "images"
        labels_dir = DATA_SPLITS / split_name / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            print(f"[{split_name}] MISSING images/ or labels/ dir")
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

        named = {TARGET_CLASS_NAMES[k]: v for k, v in sorted(class_counts.items())
                 if k < len(TARGET_CLASS_NAMES)}
        print(f"[{split_name}] images={len(valid)}  orphans={len(orphans)}  "
              f"empty-labels={empty}")
        print(f"[{split_name}] class counts: {named}")
        if orphans:
            ok = False

        missing = set(range(len(TARGET_CLASS_NAMES))) - set(class_counts.keys())
        if missing:
            print(f"[{split_name}] WARNING: missing classes "
                  f"{[TARGET_CLASS_NAMES[m] for m in missing]}")

    if ok:
        print("\nVALIDATION OK")
    else:
        print("\nVALIDATION FOUND ISSUES (see above)")
    return ok


def split(source_images: Path, source_labels: Path) -> None:
    """Stratified 80/10/10 split from a flat source dir (single-dataset utility)."""
    random.seed(SEED)
    _, valid = _pair_images_labels(source_images, source_labels)
    by_class: dict[int, list[Path]] = defaultdict(list)
    for img in valid:
        ids = _read_class_ids(source_labels / f"{img.stem}.txt")
        primary = ids[0] if ids else -1
        by_class[primary].append(img)

    assignments: dict[Path, str] = {}
    for _, imgs in by_class.items():
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * SPLIT_RATIOS[0])
        n_val   = int(n * SPLIT_RATIOS[1])
        for i, img in enumerate(imgs):
            assignments[img] = ("train" if i < n_train
                                else "val" if i < n_train + n_val
                                else "test")

    for split_name in SPLITS:
        (DATA_SPLITS / split_name / "images").mkdir(parents=True, exist_ok=True)
        (DATA_SPLITS / split_name / "labels").mkdir(parents=True, exist_ok=True)

    for img, split_name in assignments.items():
        shutil.copy2(img, DATA_SPLITS / split_name / "images" / img.name)
        shutil.copy2(
            source_labels / f"{img.stem}.txt",
            DATA_SPLITS / split_name / "labels" / f"{img.stem}.txt",
        )
    print(f"Split {len(assignments)} images → {SPLIT_RATIOS} train/val/test")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        print(f"Unknown command: {cmd!r}  (use: download | prepare | validate | split)")
        sys.exit(1)


if __name__ == "__main__":
    main()
