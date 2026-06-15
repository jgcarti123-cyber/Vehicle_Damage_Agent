"""Offline preprocessing pipeline (Albumentations).

Per the spec, Albumentations is used for *preprocessing only* — NOT during
training (Ultralytics' built-in augmentation, configured in config.yaml, handles
training-time augmentation). This keeps us from double-augmenting.

Typical use: run `preprocess_split` once to write cleaned copies into
data/processed/, then point training at those if desired. For the baseline we
train directly on data/splits and treat this as an optional enhancement step.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import albumentations as A
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_preprocessing_pipeline() -> A.Compose:
    """Return the Albumentations preprocessing pipeline (contrast/denoise).

    Bbox params use YOLO format so any boxes are kept consistent if a transform
    ever becomes geometric. The current transforms are photometric only.
    """
    return A.Compose(
        [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.2),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def preprocess_split(split: str, pipeline: A.Compose | None = None) -> int:
    """Apply preprocessing to one split, writing into data/processed/<split>/.

    Images are transformed; labels are copied verbatim (transforms are
    photometric, so boxes are unchanged). Returns the number of images written.
    """
    pipeline = pipeline or get_preprocessing_pipeline()
    src_images = DATA_SPLITS / split / "images"
    src_labels = DATA_SPLITS / split / "labels"
    dst_images = DATA_PROCESSED / split / "images"
    dst_labels = DATA_PROCESSED / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(src_images.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        out = pipeline(image=image, bboxes=[], class_labels=[])["image"]
        cv2.imwrite(str(dst_images / img_path.name), out)
        label = src_labels / f"{img_path.stem}.txt"
        if label.exists():
            shutil.copy2(label, dst_labels / label.name)
        count += 1
    print(f"[{split}] preprocessed {count} images -> {dst_images}")
    return count


if __name__ == "__main__":
    for s in ("train", "val", "test"):
        preprocess_split(s)
