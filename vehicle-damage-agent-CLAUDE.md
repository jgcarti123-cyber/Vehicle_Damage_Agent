# Vehicle Damage Assessment Agent — Project Spec

## What We're Building

An end-to-end AI system that takes car damage photos as input and produces a structured insurance/repair assessment report as output. The pipeline has 5 stages, each with distinct ML/AI components that feed into each other.

**Full pipeline:**
```
Photo(s) → [Stage 1] Damage Detection (YOLOv11)
         → [Stage 2] Severity Classification (ViT fine-tune)
         → [Stage 3] Repair Cost Prediction (XGBoost)
         → [Stage 4] LangGraph Agent (RAG + tools)
         → [Stage 5] Structured Report (FastAPI + Pydantic)
```

We are currently working on **Stage 1 only**. All design decisions should be consistent with the full architecture but scoped to damage detection for now.

---

## Stage 1 — Damage Detection

### Objective

Train a YOLOv11 object detection model to detect and localise vehicle damage regions in images. Output is bounding boxes + damage type labels + confidence scores. These outputs feed directly into Stage 2 (severity classification crops the detected regions).

### Dataset

**Primary:** CarDD (Car Damage Detection Dataset)
- Download from Roboflow: `https://universe.roboflow.com/car-damage-detection`
- ~2000+ annotated images, YOLO format
- Fallback: search Kaggle for "car damage detection dataset"

**Classes to detect (merge/simplify as needed based on dataset):**
- `scratch`
- `dent`
- `crack`
- `shatter` (glass)
- `broken_part` (bumper, mirror, etc.)
- `rust`

**Data split:** 80/10/10 train/val/test. Use stratified split to ensure all damage classes appear in each split.

### Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `ultralytics` | latest | YOLOv11 training + inference |
| `torch` | 2.x | Backend |
| `albumentations` | latest | Data augmentation |
| `wandb` | latest | Experiment tracking |
| `opencv-python` | latest | Image preprocessing |
| `pydantic` | v2 | Detection output schema |
| `fastapi` | latest | Inference API (Stage 5 prep) |

### Project File Structure

```
vehicle-damage-agent/
├── CLAUDE.md                    ← this file
├── data/
│   ├── raw/                     ← original downloaded dataset
│   ├── processed/               ← cleaned + normalised images
│   └── splits/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
├── stage1_detection/
│   ├── config.yaml              ← dataset + training config
│   ├── dataset.py               ← dataset loading + validation
│   ├── augment.py               ← Albumentations pipeline
│   ├── train.py                 ← training entry point
│   ├── evaluate.py              ← mAP + per-class metrics
│   ├── infer.py                 ← single image + batch inference
│   ├── export.py                ← ONNX export for production
│   └── schemas.py               ← Pydantic output schemas
├── stage2_severity/             ← NOT YET (placeholder)
├── stage3_cost/                 ← NOT YET (placeholder)
├── stage4_agent/                ← NOT YET (placeholder)
├── stage5_api/                  ← NOT YET (placeholder)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── tests/
│   └── test_stage1.py
├── requirements.txt
├── .env.example
└── docker/
    └── Dockerfile.stage1
```

### Training Configuration (`stage1_detection/config.yaml`)

```yaml
# Dataset
path: ../data/splits
train: train/images
val: val/images
test: test/images

# Classes
nc: 6
names:
  0: scratch
  1: dent
  2: crack
  3: shatter
  4: broken_part
  5: rust

# Training hyperparameters (start with these, tune after baseline)
model: yolo11s.pt        # start small: n < s < m < l < x
epochs: 100
imgsz: 640
batch: 16
lr0: 0.01
lrf: 0.1
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
patience: 20             # early stopping
save_period: 10
workers: 4

# Augmentation (Ultralytics built-in)
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 10.0
translate: 0.1
scale: 0.5
shear: 2.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.1

# Logging
project: vehicle-damage-detection
name: yolo11s-baseline
```

### Augmentation Pipeline (`stage1_detection/augment.py`)

Use Albumentations ON TOP of Ultralytics built-in augmentation for preprocessing only (not during training — avoid double-augmenting):

```python
import albumentations as A

def get_preprocessing_pipeline():
    return A.Compose([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),   # contrast enhance
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.2),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

### Output Schema (`stage1_detection/schemas.py`)

Pydantic schema so Stage 2 and the LangGraph agent can consume Stage 1 output cleanly:

```python
from pydantic import BaseModel
from typing import List, Optional

class DamageRegion(BaseModel):
    class_id: int
    class_name: str              # "scratch", "dent", etc.
    confidence: float
    bbox_xyxy: List[float]       # [x1, y1, x2, y2] in pixel coords
    bbox_xywh_norm: List[float]  # normalised YOLO format
    crop_path: Optional[str]     # saved crop for Stage 2 input

class DetectionResult(BaseModel):
    image_path: str
    image_width: int
    image_height: int
    num_damages: int
    regions: List[DamageRegion]
    inference_time_ms: float
    model_version: str
```

### Evaluation Targets

| Metric | Minimum acceptable | Good | Excellent |
|---|---|---|---|
| mAP@50 | 0.50 | 0.65 | 0.75+ |
| mAP@50-95 | 0.30 | 0.42 | 0.55+ |
| Precision | 0.60 | 0.72 | 0.80+ |
| Recall | 0.55 | 0.68 | 0.78+ |
| Inference latency | <500ms | <200ms | <100ms |

Run `evaluate.py` on the **test split only** after training is complete. Never tune hyperparameters using test set results.

### W&B Tracking

Log these every run:
- All training metrics (loss, mAP, P, R per epoch)
- Confusion matrix on val set
- Sample predictions visualised (positive + hard negatives)
- Hyperparameters
- Model size and inference speed

```python
import wandb
wandb.init(project="vehicle-damage-detection", config=hyperparams)
```

### ONNX Export (`stage1_detection/export.py`)

After training, export for production inference:

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640 simplify=True
```

Target: <100ms inference on CPU for the exported model.

---

## Coding Standards

- Python 3.11+
- Type hints on all functions
- Docstrings on all classes and public functions
- Unit tests in `tests/` for data loading, schema validation, and inference output shape
- `.env` for all secrets and paths (never hardcode)
- No Jupyter notebooks for production code — notebooks are for EDA only
- All file paths via `pathlib.Path`, never string concatenation

## What "Done" Looks Like for Stage 1

- [ ] Dataset downloaded, validated, split 80/10/10
- [ ] Baseline YOLOv11s trained to convergence
- [ ] mAP@50 > 0.60 on val set
- [ ] `infer.py` accepts image path, returns `DetectionResult` (Pydantic)
- [ ] `infer.py` saves annotated image + crops of each damage region to disk
- [ ] ONNX model exported and latency benchmarked
- [ ] W&B run logged with all metrics and sample predictions
- [ ] `test_stage1.py` passes
- [ ] README for `stage1_detection/` written

Once all checkboxes are done, Stage 2 begins: feeding the saved crops into a ViT severity classifier.

---

## Key Design Decisions (Do Not Change Without Discussion)

1. **YOLOv11s as baseline** — not YOLOv11n (too small) or YOLOv11m (too slow to iterate). Upgrade to m/l only if mAP plateaus.
2. **YOLO format labels** — not COCO JSON. Ultralytics expects YOLO format natively.
3. **Pydantic v2 output schemas** — strictly typed so downstream stages (LangGraph agent) consume outputs without defensive coding.
4. **Crops saved to disk after detection** — Stage 2 (severity classifier) reads these crops, not the original images. This decouples stages cleanly.
5. **ONNX export from Day 1** — production inference uses ONNX, not the PyTorch `.pt` model. Build with this in mind.
6. **W&B from Day 1** — every training run logged. No "I'll add tracking later."

---

## Implementation Addendum (actual decisions — 2026-06-15)

These supersede the original spec where they conflict; recorded as the build progressed.

1. **Dataset:** Roboflow [`sindhu/car_dent_scratch_detection-1` v9](https://universe.roboflow.com/sindhu/car_dent_scratch_detection-1/dataset/9) (CC BY 4.0), not CarDD. 3071 images; native labels are 17 **location**-based classes (e.g. `front-bumper-dent`, `Headlight-Damage`).
2. **Classes → 4 damage-type buckets:** `dent`, `glass_damage`, `light_damage`, `mirror_damage`. The 17→4 merge map lives in `stage1_detection/dataset.py` (`CLASS_MERGE_MAP`). This replaces the spec's original 6 classes (`scratch`/`crack`/`rust`/etc.), which had no annotations in this dataset.
3. **Class imbalance:** `dent` ≈83% of boxes (3936); `mirror_damage` rarest (87). Splits stratified by the rarest class per image so all 4 appear in train/val/test.
4. **Re-split 80/10/10 stratified** via `dataset.py prepare` (pools Roboflow's original 2311/680/80 and re-splits). Result: train 2399 / val 297 / test 305. 70 background images with no boxes were dropped.
5. **Python 3.11** venv (`venv311/`), not the original 3.9 venv. Apple Silicon → training on **MPS** with CPU fallback; full 100-epoch run deferred (local baseline + smoke tests only).
6. **Secrets in `.env`** (gitignored): `ROBOFLOW_API_KEY`, `WANDB_API_KEY`. No keys in source.
