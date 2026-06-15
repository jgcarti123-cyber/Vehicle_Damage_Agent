"""Stage 1 training entry point (YOLOv11 via Ultralytics).

Reads hyperparameters from config.yaml, writes a dataset-only YAML for
Ultralytics, optionally logs to Weights & Biases, and trains.

Examples:
    python stage1_detection/train.py                 # full run from config.yaml
    python stage1_detection/train.py --quick         # 3-epoch pipeline smoke test
    python stage1_detection/train.py --device cpu    # override device
    python stage1_detection/train.py --epochs 50 --batch 8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

# Let unsupported MPS ops fall back to CPU instead of erroring (Apple Silicon).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
# Dataset-only YAML handed to Ultralytics (extra training keys stripped out).
DATASET_YAML = HERE / "_dataset.yaml"

# Keys from config.yaml that are dataset definition (everything else is hyperparams).
DATASET_KEYS = {"path", "train", "val", "test", "nc", "names"}
# Keys that are not Ultralytics train() kwargs.
NON_TRAIN_KEYS = DATASET_KEYS | {"model", "project", "name"}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and return the merged Stage 1 config."""
    with path.open() as f:
        return yaml.safe_load(f)


def write_dataset_yaml(cfg: dict[str, Any]) -> Path:
    """Write a dataset-only YAML for Ultralytics, resolving `path` to absolute."""
    data = {k: cfg[k] for k in DATASET_KEYS if k in cfg}
    # Resolve relative dataset path against the config file location.
    data["path"] = str((HERE / data["path"]).resolve())
    with DATASET_YAML.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return DATASET_YAML


def _wandb_api_key() -> str | None:
    """Return a usable W&B API key from env or wandb's own config (netrc), else None."""
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    try:
        import wandb

        return wandb.api.api_key  # reads ~/.netrc / config; None if not logged in
    except Exception:
        return None


def maybe_init_wandb(cfg: dict[str, Any]) -> bool:
    """Enable Ultralytics' W&B integration only if a real key exists. Returns enabled."""
    from ultralytics import settings

    if not _wandb_api_key():
        print("W&B: no API key found (set WANDB_API_KEY in .env or run `wandb login`) "
              "— running without tracking.")
        settings.update({"wandb": False})
        return False
    settings.update({"wandb": True})
    print("W&B: enabled via Ultralytics integration.")
    return True


def train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    cfg = load_config()
    maybe_init_wandb(cfg)
    dataset_yaml = write_dataset_yaml(cfg)

    # Build train() kwargs from config hyperparameters, then apply CLI overrides.
    train_kwargs: dict[str, Any] = {k: v for k, v in cfg.items() if k not in NON_TRAIN_KEYS}
    train_kwargs.update(
        data=str(dataset_yaml),
        project=cfg.get("project", "vehicle-damage-detection"),
        name=cfg.get("name", "yolo11s-baseline"),
    )
    if args.device:
        train_kwargs["device"] = args.device
    if args.epochs is not None:
        train_kwargs["epochs"] = args.epochs
    if args.batch is not None:
        train_kwargs["batch"] = args.batch
    if args.quick:
        train_kwargs.update(epochs=3, name=cfg.get("name", "yolo11s") + "-smoke")

    model = YOLO(cfg.get("model", "yolo11s.pt"))
    print(f"Training {cfg.get('model')} on {train_kwargs['data']} "
          f"(device={train_kwargs.get('device')}, epochs={train_kwargs['epochs']})")
    results = model.train(**train_kwargs)
    print(f"Done. Best weights: {model.trainer.best}")
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage 1 YOLOv11 detector")
    p.add_argument("--device", type=str, default=None, help="mps | cpu | 0 (CUDA)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="3-epoch smoke test")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
