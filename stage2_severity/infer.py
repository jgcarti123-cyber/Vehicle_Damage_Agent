"""Stage 2 severity classifier — inference on a single crop or directory.

Can be used as a CLI or imported as a module by the Stage 1+2 pipeline.

CLI examples:
    python stage2_severity/infer.py --image outputs/stage2_crops/crops/0_img_001_0.jpg
    python stage2_severity/infer.py --crops-dir outputs/stage2_crops/crops --output-csv results.csv

Module usage:
    from stage2_severity.infer import load_model, predict
    model, device = load_model("runs/severity/efficientnet-b0-v6/weights/best.pt")
    severity, confidence, probs = predict(model, device, image_path)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from stage2_severity.train import (
    SEVERITY_CLASSES,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_model,
    get_transforms,
)

DEFAULT_WEIGHTS = Path("runs/severity/efficientnet-b0-v6/weights/best.pt")

_infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Public API (importable)
# ---------------------------------------------------------------------------

def load_model(weights: str | Path, device: torch.device | None = None):
    """Load severity model from weights file. Returns (model, device)."""
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    model = build_model(num_classes=len(SEVERITY_CLASSES)).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    return model, device


@torch.no_grad()
def predict(model, device: torch.device, image_path: str | Path):
    """Run severity inference on a single crop image.

    Returns:
        severity   (str)   — "mild" | "moderate" | "severe"
        confidence (float) — probability of the predicted class (0–1)
        probs      (dict)  — {"mild": p, "moderate": p, "severe": p}
    """
    img = Image.open(image_path).convert("RGB")
    tensor = _infer_transform(img).unsqueeze(0).to(device)
    logits = model(tensor)
    probs_tensor = F.softmax(logits, dim=1).squeeze(0).cpu()
    probs = {cls: round(probs_tensor[i].item(), 4) for i, cls in enumerate(SEVERITY_CLASSES)}
    pred_idx = probs_tensor.argmax().item()
    severity   = SEVERITY_CLASSES[pred_idx]
    confidence = probs[severity]
    return severity, confidence, probs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser(description="Stage 2 severity classifier inference")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",     type=Path, help="Single crop image to classify")
    src.add_argument("--crops-dir", type=Path, help="Directory of crop images (.jpg)")

    p.add_argument("--weights",    type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--device",     type=str,  default=None)
    p.add_argument("--output-csv", type=Path, default=None,
                   help="Save predictions to CSV (only with --crops-dir)")
    p.add_argument("--output-json", action="store_true",
                   help="Print result as JSON (single image mode)")
    args = p.parse_args()

    device = _resolve_device(args.device)
    model, device = load_model(args.weights, device)
    print(f"Device  : {device}")
    print(f"Weights : {args.weights}\n")

    if args.image:
        severity, confidence, probs = predict(model, device, args.image)
        if args.output_json:
            print(json.dumps({"severity": severity, "confidence": confidence,
                              "probabilities": probs}, indent=2))
        else:
            print(f"Image      : {args.image.name}")
            print(f"Severity   : {severity}")
            print(f"Confidence : {confidence:.3f}")
            print(f"Probs      : " +
                  "  ".join(f"{c}={probs[c]:.3f}" for c in SEVERITY_CLASSES))
        return

    # Batch mode over a directory
    crops = sorted(args.crops_dir.rglob("*.jpg"))
    if not crops:
        print(f"No .jpg files found in {args.crops_dir}")
        return
    print(f"Found {len(crops)} crops in {args.crops_dir}\n")

    results = []
    for crop in crops:
        severity, confidence, probs = predict(model, device, crop)
        results.append({
            "crop_path":  str(crop),
            "severity":   severity,
            "confidence": round(confidence, 4),
            **{f"prob_{c}": round(probs[c], 4) for c in SEVERITY_CLASSES},
        })
        print(f"  {crop.name:40s}  {severity:8s}  conf={confidence:.3f}")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["crop_path", "severity", "confidence"] + [f"prob_{c}" for c in SEVERITY_CLASSES]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nPredictions saved → {args.output_csv}")

    # Summary
    from collections import Counter
    counts = Counter(r["severity"] for r in results)
    print(f"\nSummary ({len(results)} crops):")
    for cls in SEVERITY_CLASSES:
        n = counts.get(cls, 0)
        print(f"  {cls:10s}: {n:4d}  ({n/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
