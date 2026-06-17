# Stage 2 — Damage Severity Classifier

## What This Stage Does

Takes per-region crops produced by Stage 1's `infer.py` and classifies each
crop's damage severity: **mild / moderate / severe / total_loss**.

The model trained here is a fast local EfficientNet-B0 (no cloud API at
inference time). Labels come from a one-time VLM pseudo-labeling pass using
**Gemini 2.0 Flash** (free tier via Google AI Studio — no billing required).

This is the knowledge distillation pattern:
```
Stage 1 crops → Gemini 2.0 Flash (labels, one time) → EfficientNet-B0 (inference, always)
```

---

## Severity Taxonomy (4 Classes)

```
id  label        definition
────────────────────────────────────────────────────────────────────────────
0   mild         Cosmetic only. Visible but no structural impact.
                 Examples: light surface scratch, hairline dent, paint chip.
                 Repair: buff / touch-up, < ₹8,000.

1   moderate     Visible damage, car driveable. Panel/part affected.
                 Examples: cracked tail-light, deep dent, cracked windshield.
                 Repair: part replacement or panel work, ₹8,000 – ₹60,000.

2   severe       Significant structural or functional damage. May not be safe
                 to drive. Examples: crushed door, shattered windshield, broken
                 headlight assembly, large dent crossing panel lines.
                 Repair: body shop, ₹60,000 – ₹2,00,000.

3   total_loss   Write-off threshold. Frame damage, airbag deployment, or
                 multiple severe regions. Repair cost > car market value.
                 Examples: post-collision crumple, flooded interior, fire damage.
```

**Note on total_loss:** This label will be rare in the crop set (Stage 1
typically doesn't detect crops at write-off level cleanly). If total_loss
has fewer than 50 examples after labeling, merge it into `severe` and train
a 3-class model instead. The label_crops.py script flags this automatically.

---

## Prerequisite: Get a Free Gemini API Key

1. Go to https://aistudio.google.com
2. Sign in with a Google account
3. Click **Get API key** → **Create API key in new project**
4. Copy the key. Set it as an env var:
   ```bash
   export GEMINI_API_KEY=your_key_here
   ```
   Or add to `.env` at the project root (already gitignored):
   ```
   GEMINI_API_KEY=your_key_here
   ```

**Free tier limits (Gemini 2.0 Flash):**
- 15 requests per minute
- 1,500 requests per day
- ₹0 cost

For ~3,000 crops this is 2 days of API calls at the free tier.
The label_crops.py script respects rate limits and auto-resumes from a
checkpoint CSV so you can run it across multiple days without re-labeling
already-processed crops.

**Why Gemini 2.0 Flash over GPT-4o?**
GPT-4o costs ~₹42–₹85 for this batch but requires billing setup. Gemini 2.0
Flash is free and within 5% accuracy of GPT-4o on structured visual
classification tasks like this. For a portfolio project, free is better.

---

## Directory Layout

```
stage2_severity/
├── label_crops.py      # Step 1: VLM pseudo-labeling (Gemini 2.0 Flash)
├── train.py            # Step 2: EfficientNet-B0 fine-tuning
├── evaluate.py         # Step 3: per-class accuracy + confusion matrix
├── infer.py            # Step 4: inference wrapper → SeverityResult
├── schemas.py          # Pydantic v2 output schema (SeverityResult)
└── data/               # Created at runtime
    ├── labels.csv      # GPT/Gemini output: crop_path, damage_type, severity, confidence, reasoning
    ├── labels_filtered.csv  # After confidence filtering
    └── splits/
        ├── train/      # Symlinks or copies of crops by class
        ├── val/
        └── test/
```

All crops to be labeled live under `outputs/` from Stage 1 runs. The
label_crops.py script takes a glob pattern so you can point it at any
`outputs/*/crops/**/*.jpg` path.

---

## Step 0 — Generate Crops from Training Images

Before labeling, run Stage 1 inference on the training set images to produce
crops. This gives a large, diverse crop set (not just the smoke-test images).

```bash
# Point at the full training split images
python stage1_detection/infer.py \
    --source data/splits/train/images \
    --output outputs/stage2_crops \
    --save-crops \
    --conf 0.25          # lower threshold → more crops → more training data
```

Expected output: `outputs/stage2_crops/crops/<image_stem>/<class_id>_<i>.jpg`

The filename prefix `<class_id>` encodes the damage type (0=dent, 1=scratch,
2=glass_damage, 3=light_damage, 4=mirror_damage) — used in the labeling prompt.

Also run on val images if you want a larger label set:
```bash
python stage1_detection/infer.py \
    --source data/splits/val/images \
    --output outputs/stage2_crops_val \
    --save-crops \
    --conf 0.25
```

---

## Step 1 — VLM Pseudo-Labeling (`label_crops.py`)

### What it does
- Reads every `.jpg` under a crops directory
- Parses damage type from the filename prefix (`0_0.jpg` → `dent`)
- Sends each crop to Gemini 2.0 Flash with a structured prompt
- Saves result to `stage2_severity/data/labels.csv`
- Checkpoints after every crop — safe to interrupt and resume

### Gemini prompt (structured output)
```
You are a vehicle damage assessor. Rate the severity of the damage visible
in this image crop.

Damage type: {damage_type}  (dent / scratch / glass_damage / light_damage / mirror_damage)

Choose exactly ONE severity level:
- mild:        cosmetic only, no structural impact, < ₹8,000 repair
- moderate:    visible damage, car driveable, ₹8,000 – ₹60,000 repair
- severe:      structural or functional damage, ₹60,000 – ₹2,00,000 repair
- total_loss:  write-off, repair cost > car IDV (insured declared value)

Respond with valid JSON only. No other text:
{
  "severity": "<mild|moderate|severe|total_loss>",
  "confidence": <0.0–1.0>,
  "reasoning": "<one sentence>"
}
```

### Implementation notes
- Use `google-generativeai` Python SDK (pip install google-generativeai)
- Model: `gemini-2.0-flash` (or `gemini-1.5-flash` if 2.0 unavailable)
- Rate limiting: use `time.sleep(4.1)` between calls to stay under 15 RPM
- Parse JSON from response text with `json.loads(response.text)`
- If JSON parse fails or severity is not in valid set → log to `labels_errors.csv`, skip
- Checkpoint: load existing `labels.csv` at startup, skip already-labeled crops

### Output CSV columns
```
crop_path, image_stem, damage_type, severity, confidence, reasoning, model
```

### Confidence filtering
After labeling, run the built-in filter:
```bash
python stage2_severity/label_crops.py filter --min-confidence 0.75
```
This writes `labels_filtered.csv`. Review borderline cases:
```bash
python stage2_severity/label_crops.py review --limit 50
```
Opens a simple terminal viewer showing crop + label for manual verification.

---

## Step 2 — Train EfficientNet-B0 (`train.py`)

### Why EfficientNet-B0
- 5.3M parameters, runs on CPU in <100ms at inference
- Pretrained on ImageNet → transfers well to small damage crop datasets
- Much faster training than ViT-B/16 on limited data
- Torchvision has it built-in: `torchvision.models.efficientnet_b0`

### Training setup
- Input: `labels_filtered.csv` → symlinked into `stage2_severity/data/splits/`
- Split: 70 / 15 / 15 (train/val/test) — stratified by severity class
- Augmentation (train only, via torchvision transforms):
  - RandomHorizontalFlip
  - RandomRotation(15)
  - ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
  - Normalize to ImageNet mean/std
- Freeze backbone for first 5 epochs, unfreeze all for remaining epochs
- Loss: CrossEntropyLoss with class weights (inverse frequency — handles imbalance)
- Optimizer: AdamW lr=1e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingLR over full training
- Early stopping: patience=10 on val accuracy
- Epochs: 50 (early stop will trigger sooner in practice)
- Batch size: 32

### W&B logging (optional)
Check for `WANDB_API_KEY` env var. If set, log metrics. If not, skip silently
(same pattern as Stage 1 train.py). Do NOT auto-enable based on `.netrc`.

### Checkpoint saving
Save best model to `stage2_severity/weights/best.pt` (state dict + class names).
Also save `stage2_severity/weights/class_names.json` for inference.

### If total_loss < 50 examples
Automatically drop to 3-class mode (mild/moderate/severe) and print a warning.
Save class names accordingly so infer.py reads them dynamically.

---

## Step 3 — Evaluate (`evaluate.py`)

### Metrics
- Overall accuracy
- Per-class precision, recall, F1 (sklearn classification_report)
- Confusion matrix (saved to `stage2_severity/outputs/confusion_matrix.png`)
- Per-class sample counts (to show where training data is thin)

### Pass/fail thresholds
```
Overall accuracy    > 0.70  ✅ pass
Per-class F1        > 0.60  ✅ pass (each class)
```

Print a pass/fail summary like Stage 1's evaluate.py.

---

## Step 4 — Inference Integration (`infer.py`)

### Input
- A crop image path (string or Path)
- Optional: damage_type string (for logging; already known from Stage 1)

### Output
A `SeverityResult` Pydantic object (see schemas.py below).

### Implementation
```python
def classify_severity(crop_path: str | Path, damage_type: str | None = None) -> SeverityResult:
    ...
```

Load model once at module level (lazy singleton) — not per call.

---

## Step 5 — Output Schema (`schemas.py`)

```python
from __future__ import annotations
from pydantic import BaseModel, Field

class SeverityResult(BaseModel):
    """Severity classification output for one damage crop (Stage 2)."""

    crop_path: str = Field(..., description="Path to the crop image classified")
    damage_type: str = Field(
        ..., description='Damage class from Stage 1, e.g. "scratch"'
    )
    severity: str = Field(
        ..., description="mild | moderate | severe | total_loss"
    )
    severity_id: int = Field(..., ge=0, description="Integer index for severity class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classifier softmax confidence")
    model_version: str = Field(..., description="Weights file identifier / tag")
```

Extend `DamageRegion` in `stage1_detection/schemas.py` to add an optional
`severity: SeverityResult | None = None` field so the Stage 4 agent gets a
single unified object per damage region.

---

## Build Steps (in order)

```bash
# 0. Generate crops from training images
python stage1_detection/infer.py \
    --source data/splits/train/images \
    --output outputs/stage2_crops \
    --save-crops --conf 0.25

# 1. Label crops with Gemini 2.0 Flash (free, ~2 days at free tier for ~3k crops)
#    Set GEMINI_API_KEY in .env first.
python stage2_severity/label_crops.py \
    --crops-dir outputs/stage2_crops/crops \
    --output stage2_severity/data/labels.csv

# 1b. Filter low-confidence labels
python stage2_severity/label_crops.py filter --min-confidence 0.75

# 2. Train EfficientNet-B0
python stage2_severity/train.py \
    --labels stage2_severity/data/labels_filtered.csv \
    --output stage2_severity/weights

# 3. Evaluate on held-out test split
python stage2_severity/evaluate.py \
    --weights stage2_severity/weights/best.pt \
    --labels stage2_severity/data/labels_filtered.csv

# 4. Smoke test inference
python stage2_severity/infer.py --crop outputs/stage2_crops/crops/car001/0_0.jpg
```

---

## Integration with Stage 1 Pipeline

Once `stage2_severity/infer.py` is working, update `stage1_detection/infer.py`
to call `classify_severity()` on each saved crop and attach the result to
`DamageRegion.severity`. This makes the full pipeline Stage 1 → Stage 2 a
single `python stage1_detection/infer.py --image car.jpg` call.

---

## Expected Outcomes

| Metric | Target |
|---|---|
| Overall accuracy | > 0.70 |
| Per-class F1 | > 0.60 (each class present) |
| Inference latency | < 50ms per crop (CPU) |
| Label cost | ₹0 (Gemini free tier) |

---

## Key Decisions (Do Not Change Without Discussion)

1. **Gemini 2.0 Flash, not GPT-4o.** Free tier is sufficient. Accuracy
   difference is <5% for this task. If accuracy falls short after training,
   re-label the 0.60–0.75 confidence bucket with GPT-4o (~₹42–₹85 total) as a targeted fix.

2. **EfficientNet-B0, not ViT-B/16.** EfficientNet-B0 trains faster, needs
   less data, and hits <50ms CPU latency. ViT requires more data to beat
   EfficientNet on small datasets like ours.

3. **Pseudo-labels, not manual.** Manual labeling of 3,000 crops is days of
   work and highly subjective. VLM pseudo-labels take ~2 hours of API time
   and achieve comparable label quality on coarse severity buckets.

4. **total_loss auto-merge if < 50 examples.** Rather than training a weak
   4th class, merge and document the decision. This is honest ML practice.

5. **Crop-level severity, not image-level.** We classify individual damage
   regions, not the whole car. This is more useful for repair cost estimation
   (Stage 3) and aligns with the Stage 4 agent's per-region report format.
