"""Stage 1 evaluation: mAP + per-class metrics on the TEST split.

Run only AFTER training is complete. Never tune hyperparameters using these
numbers — the test split is held out.

Example:
    python stage1_detection/evaluate.py --weights runs/detect/yolo11s-baseline/weights/best.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from train import load_config, write_dataset_yaml  # noqa: E402

# Spec evaluation targets (minimum acceptable) for a quick pass/fail readout.
TARGETS = {
    "mAP@50": 0.50,
    "mAP@50-95": 0.30,
    "precision": 0.60,
    "recall": 0.55,
}


def evaluate(weights: Path, device: str | None = None) -> dict[str, float]:
    """Run validation on the test split and print per-class + summary metrics."""
    from ultralytics import YOLO

    cfg = load_config()
    dataset_yaml = write_dataset_yaml(cfg)
    model = YOLO(str(weights))

    metrics = model.val(
        data=str(dataset_yaml),
        split="test",
        device=device or cfg.get("device", "cpu"),
        verbose=True,
    )

    summary = {
        "mAP@50": float(metrics.box.map50),
        "mAP@50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    print("\n=== TEST SUMMARY ===")
    names = cfg["names"]
    for i, ap50 in enumerate(metrics.box.ap50):
        print(f"  {names.get(i, i):14s} mAP@50={ap50:.3f}")
    print("  ------------------------------")
    for k, v in summary.items():
        flag = "OK " if v >= TARGETS[k] else "LOW"
        print(f"  [{flag}] {k:11s} {v:.3f} (min {TARGETS[k]})")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Stage 1 detector on test split")
    p.add_argument("--weights", type=Path, required=True, help="Path to best.pt")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.weights, args.device)
