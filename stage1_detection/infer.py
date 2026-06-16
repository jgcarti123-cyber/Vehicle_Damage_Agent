"""Stage 1 inference: image -> DetectionResult (+ annotated image and crops).

Produces the Pydantic `DetectionResult` that Stage 2 / the LangGraph agent
consume, and writes per-region crops to disk (Stage 2 reads these, not the
originals — see CLAUDE.md design decisions).

Examples:
    python stage1_detection/infer.py --weights best.pt --source car.jpg
    python stage1_detection/infer.py --weights best.pt --source images/ --out outputs/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from schemas import DamageRegion, DetectionResult

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _gather_images(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [source]


def infer_image(model, image_path: Path, out_dir: Path, conf: float,
                model_version: str, annotate: bool = True) -> DetectionResult:
    """Run detection on one image, save annotated image + crops, return result."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]

    start = time.perf_counter()
    result = model.predict(source=str(image_path), conf=conf, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    crops_dir = out_dir / "crops" / image_path.stem
    crops_dir.mkdir(parents=True, exist_ok=True)
    names = result.names

    regions: list[DamageRegion] = []
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls.item())
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        xc, yc, bw, bh = (float(v) for v in box.xywhn[0].tolist())

        # Save crop for Stage 2 (clamp to image bounds).
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(w, int(x2)), min(h, int(y2))
        crop_path = crops_dir / f"{cls_id}_{i}.jpg"
        if cx2 > cx1 and cy2 > cy1:
            cv2.imwrite(str(crop_path), image[cy1:cy2, cx1:cx2])

        regions.append(
            DamageRegion(
                class_id=cls_id,
                class_name=names[cls_id],
                confidence=float(box.conf.item()),
                bbox_xyxy=[x1, y1, x2, y2],
                bbox_xywh_norm=[xc, yc, bw, bh],
                crop_path=str(crop_path) if crop_path.exists() else None,
            )
        )

    # Save annotated image (skip with --no-annotate for bulk crop-generation runs).
    out_dir.mkdir(parents=True, exist_ok=True)
    if annotate:
        annotated_path = out_dir / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(annotated_path), result.plot())

    return DetectionResult(
        image_path=str(image_path),
        image_width=w,
        image_height=h,
        num_damages=len(regions),
        regions=regions,
        inference_time_ms=elapsed_ms,
        model_version=model_version,
    )


def run(weights: Path, source: Path, out_dir: Path, conf: float,
        device: str | None = None, annotate: bool = True) -> list[DetectionResult]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if device:
        model.to(device)
    images = _gather_images(source)
    results: list[DetectionResult] = []
    for i, img in enumerate(images, 1):
        res = infer_image(model, img, out_dir, conf, model_version=weights.name,
                          annotate=annotate)
        if annotate:
            (out_dir / f"{img.stem}.json").write_text(res.model_dump_json(indent=2))
        if i % 100 == 0 or i == len(images):
            print(f"[{i}/{len(images)}] {img.name}: {res.num_damages} damage(s) "
                  f"in {res.inference_time_ms:.1f}ms")
        results.append(res)
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 inference")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True, help="image file or directory")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-annotate", action="store_true",
                   help="skip annotated image + JSON output (faster bulk crop generation)")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.weights, a.source, a.out, a.conf, a.device, annotate=not a.no_annotate)
