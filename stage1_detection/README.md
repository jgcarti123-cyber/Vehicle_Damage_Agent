# Stage 1 — Damage Detection

YOLOv11 object detector that localises vehicle damage regions. Output is
bounding boxes + damage-type labels + confidence, packaged as a Pydantic
`DetectionResult` and saved crops that feed Stage 2 (severity classification).

## Classes

The detector predicts **5 damage-type buckets**, pooled from 4 source
datasets (Roboflow `sindhu/car_dent_scratch_detection-1` v9, Kaggle CarDD,
Roboflow `nivethetha/car-damages-godhu` v1, Roboflow
`damagedetection-hloj4/damagelocation` v7). Per-dataset class remaps live in
[`dataset.py`](dataset.py) (`DATASETS` registry). See
`../taxonomy-v2-CLAUDE.md` for the full rationale.

| id | class | primary sources |
|----|-------|-------------|
| 0 | `dent` | all datasets (incl. CarDD `crack`, merged in) |
| 1 | `scratch` | CarDD (primary), nivethetha |
| 2 | `glass_damage` | sindhu, CarDD `glass shatter`, nivethetha |
| 3 | `light_damage` | sindhu, CarDD `lamp broken`, nivethetha |
| 4 | `mirror_damage` | sindhu only — rarest class (87 boxes total) |

`tire_flat` (CarDD) is dropped — not a collision-damage class.

## Setup

```bash
python3.11 -m venv venv311 && source venv311/bin/activate
pip install -U pip && pip install -r requirements.txt
cp .env.example .env          # then fill ROBOFLOW_API_KEY, KAGGLE_USERNAME/KAGGLE_KEY (and WANDB_API_KEY)
```

## Pipeline

```bash
# 1. Download all datasets: sindhu/nivethetha/damagelocation via Roboflow
#    (reads ROBOFLOW_API_KEY from .env), CarDD via kagglehub
#    (reads KAGGLE_USERNAME/KAGGLE_KEY from .env).
python stage1_detection/dataset.py download

# 2. Pool all datasets, remap to the 5-class taxonomy, re-split 80/10/10 stratified
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
- **Class imbalance:** `dent` ≈58% of boxes; `mirror_damage` is rarest (87
  boxes total, sindhu only). Expect lower per-class mAP on `mirror_damage`
  at baseline — it's structurally limited by available data.
- **Test split is held out** — only run `evaluate.py` on it after training; never
  tune against it.
