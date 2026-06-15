# Stage 1 — Damage Detection

YOLOv11 object detector that localises vehicle damage regions. Output is
bounding boxes + damage-type labels + confidence, packaged as a Pydantic
`DetectionResult` and saved crops that feed Stage 2 (severity classification).

## Classes

The detector predicts **4 damage-type buckets**, merged from the Roboflow
[`sindhu/car_dent_scratch_detection-1` v9](https://universe.roboflow.com/sindhu/car_dent_scratch_detection-1/dataset/9)
dataset (whose native 17 labels are *location*-based). The merge map lives in
[`dataset.py`](dataset.py) (`CLASS_MERGE_MAP`).

| id | class | merged from |
|----|-------|-------------|
| 0 | `dent` | all 13 `*-dent` location classes |
| 1 | `glass_damage` | front + rear windscreen damage |
| 2 | `light_damage` | headlight, taillight, signlight damage |
| 3 | `mirror_damage` | sidemirror damage |

## Setup

```bash
python3.11 -m venv venv311 && source venv311/bin/activate
pip install -U pip && pip install -r requirements.txt
cp .env.example .env          # then fill ROBOFLOW_API_KEY (and WANDB_API_KEY)
```

## Pipeline

```bash
# 1. Download dataset from Roboflow (reads ROBOFLOW_API_KEY from .env)
python stage1_detection/dataset.py download

# 2. Remap 17->4 classes and re-split 80/10/10 stratified into data/splits/
python stage1_detection/dataset.py prepare

# 3. Validate image/label pairing + class distribution
python stage1_detection/dataset.py validate

# 4. (optional) offline Albumentations preprocessing -> data/processed/
python stage1_detection/augment.py

# 5. Train (MPS on Apple Silicon). Smoke test first:
python stage1_detection/train.py --quick
python stage1_detection/train.py                      # full run from config.yaml

# 6. Evaluate on the held-out TEST split
python stage1_detection/evaluate.py --weights runs/detect/yolo11s-baseline/weights/best.pt

# 7. Inference -> DetectionResult JSON + annotated image + crops
python stage1_detection/infer.py --weights <best.pt> --source <img-or-dir> --out outputs/

# 8. ONNX export + CPU latency benchmark
python stage1_detection/export.py --weights <best.pt> --benchmark
```

## Files

| File | Purpose |
|------|---------|
| `config.yaml` | dataset paths, classes, training hyperparameters |
| `dataset.py` | download / remap / stratified split / validation |
| `augment.py` | offline Albumentations preprocessing (not training-time) |
| `train.py` | training entry point (config-driven, optional W&B) |
| `evaluate.py` | mAP + per-class metrics on the test split |
| `infer.py` | single/batch inference -> `DetectionResult` + crops |
| `export.py` | ONNX export + CPU benchmark |
| `schemas.py` | Pydantic v2 output contract |

## Notes

- **Device:** training defaults to `mps` (Apple Silicon). Override with
  `--device cpu` or `--device 0` (CUDA). `PYTORCH_ENABLE_MPS_FALLBACK=1` is set
  so unsupported MPS ops fall back to CPU.
- **Class imbalance:** `dent` ≈83% of boxes; `mirror_damage` is rarest (87
  boxes). Expect lower per-class mAP on the minority classes at baseline.
- **Test split is held out** — only run `evaluate.py` on it after training; never
  tune against it.
