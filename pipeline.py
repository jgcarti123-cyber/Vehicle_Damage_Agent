"""End-to-end vehicle damage pipeline: Stage 1 detection + Stage 2 severity.

Routing thresholds (severity model confidence):
  >= 0.85          ->  auto_classify           (AI decision, no human needed)
  0.70 – 0.84      ->  suggest_human_confirm   (AI suggests, assessor confirms)
  < 0.70           ->  human_review            (route to human assessor queue)

Annotated image:
  Green  box  = mild
  Orange box  = moderate
  Red    box  = severe
  Dashed box  = human_review (low confidence)

Examples:
    python pipeline.py --image car.jpg
    python pipeline.py --image car.jpg --json
    python pipeline.py --source images/ --out outputs/pipeline/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on the path for stage1/stage2 imports.
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_detection"))

import cv2
import torch
from pydantic import BaseModel, Field

from stage1_detection.schemas import DamageRegion, DetectionResult
from stage2_severity.infer import load_model as _load_severity, predict as _predict_severity
from stage2_severity.train import SEVERITY_CLASSES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STAGE1 = _ROOT / "runs/detect/vehicle-damage-detection/yolo11s-v2-5class/weights/best.pt"
DEFAULT_STAGE2 = _ROOT / "runs/severity/efficientnet-b0-v6/weights/best.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SEVERITY_ORDER = {"mild": 0, "moderate": 1, "severe": 2}

# BGR colors for annotation
SEVERITY_COLORS = {
    "mild":     (34,  139, 34),   # green
    "moderate": (0,   140, 255),  # orange
    "severe":   (0,   0,   220),  # red
}

ROUTING_THRESHOLDS = [
    (0.85, "auto_classify"),
    (0.70, "suggest_human_confirm"),
    (0.00, "human_review"),
]

ROUTING_ORDER = ["auto_classify", "suggest_human_confirm", "human_review"]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SeverityAssessment(BaseModel):
    severity: str = Field(..., description="mild | moderate | severe")
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    routing: str = Field(..., description="auto_classify | suggest_human_confirm | human_review")


class RegionResult(BaseModel):
    class_id: int
    class_name: str
    detection_confidence: float
    bbox_xyxy: list[float]
    crop_path: str | None
    severity: SeverityAssessment | None = None


class PipelineResult(BaseModel):
    image_path: str
    image_width: int
    image_height: int
    num_damages: int
    regions: list[RegionResult]
    overall_severity: str = Field(..., description="Worst severity across all regions")
    overall_routing: str  = Field(..., description="Most cautious routing across all regions")
    detection_time_ms: float
    severity_time_ms: float
    total_time_ms: float
    stage1_model: str
    stage2_model: str


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def routing_decision(confidence: float) -> str:
    for threshold, label in ROUTING_THRESHOLDS:
        if confidence >= threshold:
            return label
    return "human_review"


def worst_routing(routings: list[str]) -> str:
    return max(routings, key=lambda r: ROUTING_ORDER.index(r))


def worst_severity(severities: list[str]) -> str:
    return max(severities, key=lambda s: SEVERITY_ORDER.get(s, -1), default="mild")


# ---------------------------------------------------------------------------
# Stage 1 — YOLO detection
# ---------------------------------------------------------------------------

def _detect(yolo_model, image_path: Path, out_dir: Path,
             conf_threshold: float, annotate: bool) -> tuple[DetectionResult, float]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h, w = image.shape[:2]

    t0 = time.perf_counter()
    result = yolo_model.predict(source=str(image_path), conf=conf_threshold, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    crops_dir = out_dir / "crops" / image_path.stem
    crops_dir.mkdir(parents=True, exist_ok=True)
    names = result.names

    regions: list[DamageRegion] = []
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls.item())
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        xc, yc, bw, bh = (float(v) for v in box.xywhn[0].tolist())

        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(w, int(x2)), min(h, int(y2))
        crop_path = crops_dir / f"{cls_id}_{i}.jpg"
        if cx2 > cx1 and cy2 > cy1:
            cv2.imwrite(str(crop_path), image[cy1:cy2, cx1:cx2])

        regions.append(DamageRegion(
            class_id=cls_id,
            class_name=names[cls_id],
            confidence=float(box.conf.item()),
            bbox_xyxy=[x1, y1, x2, y2],
            bbox_xywh_norm=[xc, yc, bw, bh],
            crop_path=str(crop_path) if crop_path.exists() else None,
        ))

    return DetectionResult(
        image_path=str(image_path),
        image_width=w,
        image_height=h,
        num_damages=len(regions),
        regions=regions,
        inference_time_ms=elapsed_ms,
        model_version=str(DEFAULT_STAGE1.name),
    ), elapsed_ms


# ---------------------------------------------------------------------------
# Stage 2 — severity per crop
# ---------------------------------------------------------------------------

def _classify_regions(
    detection: DetectionResult,
    severity_model,
    device: torch.device,
) -> tuple[list[RegionResult], float]:
    results = []
    t0 = time.perf_counter()

    for region in detection.regions:
        sev_result = None
        if region.crop_path and Path(region.crop_path).exists():
            severity, confidence, probs = _predict_severity(
                severity_model, device, region.crop_path
            )
            sev_result = SeverityAssessment(
                severity=severity,
                confidence=confidence,
                probabilities=probs,
                routing=routing_decision(confidence),
            )
        results.append(RegionResult(
            class_id=region.class_id,
            class_name=region.class_name,
            detection_confidence=region.confidence,
            bbox_xyxy=region.bbox_xyxy,
            crop_path=region.crop_path,
            severity=sev_result,
        ))

    severity_ms = (time.perf_counter() - t0) * 1000.0
    return results, severity_ms


# ---------------------------------------------------------------------------
# Annotated image
# ---------------------------------------------------------------------------

def _annotate(image_path: Path, regions: list[RegionResult], out_dir: Path) -> Path:
    img = cv2.imread(str(image_path))
    if img is None:
        return image_path

    for r in regions:
        x1, y1, x2, y2 = (int(v) for v in r.bbox_xyxy)
        sev  = r.severity.severity if r.severity else "unknown"
        conf = r.severity.confidence if r.severity else 0.0
        routing = r.severity.routing if r.severity else "human_review"
        color = SEVERITY_COLORS.get(sev, (128, 128, 128))

        thickness = 1 if routing == "human_review" else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label = f"{r.class_name} | {sev} {conf:.0%}"
        if routing == "human_review":
            label += " [REVIEW]"
        elif routing == "suggest_human_confirm":
            label += " [CONFIRM?]"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(img, label, (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_pipeline.jpg"
    cv2.imwrite(str(out_path), img)
    return out_path


# ---------------------------------------------------------------------------
# Main pipeline function (importable)
# ---------------------------------------------------------------------------

def run_image(
    image_path: Path,
    yolo_model,
    severity_model,
    device: torch.device,
    out_dir: Path,
    detection_conf: float = 0.25,
    annotate: bool = True,
    stage1_name: str = "yolo11s-v2-5class",
    stage2_name: str = "efficientnet-b0-v6",
) -> PipelineResult:
    t_start = time.perf_counter()

    detection, det_ms = _detect(yolo_model, image_path, out_dir, detection_conf, annotate)
    regions, sev_ms   = _classify_regions(detection, severity_model, device)

    severities = [r.severity.severity for r in regions if r.severity]
    routings   = [r.severity.routing  for r in regions if r.severity]

    overall_sev     = worst_severity(severities)   if severities else "unknown"
    overall_routing = worst_routing(routings)       if routings   else "human_review"

    if annotate:
        annotated_path = _annotate(image_path, regions, out_dir)

    total_ms = (time.perf_counter() - t_start) * 1000.0

    return PipelineResult(
        image_path=str(image_path),
        image_width=detection.image_width,
        image_height=detection.image_height,
        num_damages=len(regions),
        regions=regions,
        overall_severity=overall_sev,
        overall_routing=overall_routing,
        detection_time_ms=round(det_ms, 1),
        severity_time_ms=round(sev_ms, 1),
        total_time_ms=round(total_ms, 1),
        stage1_model=stage1_name,
        stage2_model=stage2_name,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(result: PipelineResult) -> None:
    print(f"\nImage    : {Path(result.image_path).name}")
    print(f"Damages  : {result.num_damages}")
    print(f"Overall  : {result.overall_severity.upper()}  [{result.overall_routing}]")
    print(f"Timing   : detection={result.detection_time_ms:.0f}ms  "
          f"severity={result.severity_time_ms:.0f}ms  "
          f"total={result.total_time_ms:.0f}ms")
    print()
    for i, r in enumerate(result.regions, 1):
        sev  = r.severity.severity    if r.severity else "n/a"
        conf = r.severity.confidence  if r.severity else 0.0
        rout = r.severity.routing     if r.severity else "n/a"
        print(f"  [{i}] {r.class_name:15s} | {sev:8s} {conf:.0%}  [{rout}]")


def main():
    p = argparse.ArgumentParser(description="Vehicle damage detection + severity pipeline")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",  type=Path, help="Single input image")
    src.add_argument("--source", type=Path, help="Directory of images")

    p.add_argument("--stage1-weights", type=Path, default=DEFAULT_STAGE1)
    p.add_argument("--stage2-weights", type=Path, default=DEFAULT_STAGE2)
    p.add_argument("--out",      type=Path, default=Path("outputs/pipeline"))
    p.add_argument("--conf",     type=float, default=0.25, help="YOLO detection threshold")
    p.add_argument("--device",   type=str,   default=None)
    p.add_argument("--json",     action="store_true", help="Print JSON output")
    p.add_argument("--no-annotate", action="store_true")
    args = p.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device          : {device}")
    print(f"Stage1 weights  : {args.stage1_weights}")
    print(f"Stage2 weights  : {args.stage2_weights}")

    from ultralytics import YOLO
    yolo = YOLO(str(args.stage1_weights))

    severity_model, device = _load_severity(args.stage2_weights, device)

    images = (
        [args.image]
        if args.image
        else sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    )
    print(f"Processing {len(images)} image(s)...\n")

    for img_path in images:
        result = run_image(
            image_path=img_path,
            yolo_model=yolo,
            severity_model=severity_model,
            device=device,
            out_dir=args.out,
            detection_conf=args.conf,
            annotate=not args.no_annotate,
        )

        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            _print_result(result)
            if not args.no_annotate:
                print(f"  Annotated → {args.out}/{img_path.stem}_pipeline.jpg")

        (args.out / f"{img_path.stem}_result.json").write_text(
            result.model_dump_json(indent=2)
        )


if __name__ == "__main__":
    main()
