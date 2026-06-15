# Taxonomy V2 — Add Scratch Class

## What Changed and Why

The original 4-class taxonomy is being expanded to **5 classes** by adding `scratch`.

Scratch is a distinct damage type in insurance assessment (different repair cost from dent —
scratch requires paint work, dent may not). Multiple datasets cover it. This change
makes the model more useful for the core insurance/repair use case.

**tire_flat is dropped** — niche use case, only one data source, not relevant to
collision damage assessment.

**CarDD is now training data**, not an OOD evaluation set. The OOD plan is cancelled.

---

## New Taxonomy (5 Classes)

```
id  name           was
──────────────────────────────
0   dent           same
1   scratch        NEW
2   glass_damage   same (was id 1)
3   light_damage   same (was id 2)
4   mirror_damage  same (was id 3)
```

**Note:** The id numbers shift for glass_damage, light_damage, mirror_damage
(scratch is inserted at id 1). Every place that references class ids must be updated.

---

## Dataset Remaps

### Dataset 1: Sindhu v9 (existing)
`workspace: sindhu / project: car_dent_scratch_detection-1 / version: 9`

Sindhu has no scratch class. Its contribution to scratch = 0.
All other remaps stay the same, just with updated target ids.

```python
"source_classes": [
    "Bodypanel-Dent",           # 0
    "Front-Windscreen-Damage",  # 1
    "Headlight-Damage",         # 2
    "Rear-windscreen-Damage",   # 3
    "RunningBoard-Dent",        # 4
    "Sidemirror-Damage",        # 5
    "Signlight-Damage",         # 6
    "Taillight-Damage",         # 7
    "bonnet-dent",              # 8
    "boot-dent",                # 9
    "doorouter-dent",           # 10
    "fender-dent",              # 11
    "front-bumper-dent",        # 12
    "pillar-dent",              # 13
    "quaterpanel-dent",         # 14
    "rear-bumper-dent",         # 15
    "roof-dent",                # 16
],
"class_remap": {
    0:  0,   # Bodypanel-Dent        → dent
    1:  2,   # Front-Windscreen      → glass_damage
    2:  3,   # Headlight-Damage      → light_damage
    3:  2,   # Rear-windscreen       → glass_damage
    4:  0,   # RunningBoard-Dent     → dent
    5:  4,   # Sidemirror-Damage     → mirror_damage
    6:  3,   # Signlight-Damage      → light_damage
    7:  3,   # Taillight-Damage      → light_damage
    8:  0,   # bonnet-dent           → dent
    9:  0,   # boot-dent             → dent
    10: 0,   # doorouter-dent        → dent
    11: 0,   # fender-dent           → dent
    12: 0,   # front-bumper-dent     → dent
    13: 0,   # pillar-dent           → dent
    14: 0,   # quaterpanel-dent      → dent
    15: 0,   # rear-bumper-dent      → dent
    16: 0,   # roof-dent             → dent
}
```

---

### Dataset 2: CarDD YOLO (NEW)
Downloaded via Kaggle API to: `data/raw/cardd_yolo/`

**Check the actual directory structure first** — Kaggle zips vary. It may be:
- `data/raw/cardd_yolo/train/images/` + `valid/` + `test/`
- `data/raw/cardd_yolo/images/` (flat)
- `data/raw/cardd_yolo/CarDD-YOLO/images/`

Use `pathlib.Path.rglob('*.jpg')` to discover the actual layout before writing the
download/pool logic for this dataset.

```python
"source_classes": [
    "dent",          # 0
    "scratch",       # 1
    "crack",         # 2
    "glass_damage",  # 3
    "light_damage",  # 4
    "tire_flat",     # 5
],
"class_remap": {
    0: 0,   # dent          → dent
    1: 1,   # scratch       → scratch  ✅ NEW
    2: 0,   # crack         → dent (body surface crack, same repair category)
    3: 2,   # glass_damage  → glass_damage
    4: 3,   # light_damage  → light_damage
    # 5: tire_flat → DROP
}
```

CarDD is the **primary source for scratch** (class 1). It also adds diversity for
dent and glass_damage since it comes from a completely different annotation team
than Sindhu.

---

### Dataset 3: nivethetha (RECOMMENDED ADDITION)
`https://universe.roboflow.com/nivethetha/car-damages-godhu`
`workspace: nivethetha / project: car-damages-godhu / version: 5`

Now valuable for both scratch AND mirror coverage.

```python
"source_classes": [
    "scratch",           # 0
    "dent",              # 1
    "bonnet",            # 2  ← part label, DROP
    "bonnet-dent",       # 3
    "bonnet-scratch",    # 4
    "bumper-dent",       # 5
    "bumper-scratch",    # 6
    "door-crack",        # 7
    "door-dent",         # 8
    "door-scratch",      # 9
    "glass broken",      # 10
    "grill-broken",      # 11 ← DROP
    "headlight-broken",  # 12
    "roof-crushed",      # 13
    "sidemirror-broken", # 14
    "sidemirror-crack",  # 15
    "taillight-broken",  # 16
    "window-broken",     # 17
    "window-crack",      # 18
    "window-scratch",    # 19
    "windshield-broken", # 20
    "windshield-crack",  # 21
    "windshield-scratch",# 22
],
"class_remap": {
    0:  1,   # scratch           → scratch
    1:  0,   # dent              → dent
    3:  0,   # bonnet-dent       → dent
    4:  1,   # bonnet-scratch    → scratch
    5:  0,   # bumper-dent       → dent
    6:  1,   # bumper-scratch    → scratch
    7:  0,   # door-crack        → dent (body crack)
    8:  0,   # door-dent         → dent
    9:  1,   # door-scratch      → scratch
    10: 2,   # glass broken      → glass_damage
    12: 3,   # headlight-broken  → light_damage
    13: 0,   # roof-crushed      → dent
    14: 4,   # sidemirror-broken → mirror_damage
    15: 4,   # sidemirror-crack  → mirror_damage
    16: 3,   # taillight-broken  → light_damage
    17: 2,   # window-broken     → glass_damage
    18: 2,   # window-crack      → glass_damage
    19: 1,   # window-scratch    → scratch
    20: 2,   # windshield-broken → glass_damage
    21: 2,   # windshield-crack  → glass_damage
    22: 1,   # windshield-scratch→ scratch
    # 2: bonnet (part label) → DROP
    # 11: grill-broken → DROP (no category)
}
```

---

### Dataset 4: DamageLocation (RECOMMENDED ADDITION)
`https://universe.roboflow.com/damagedetection-hloj4/damagelocation`
`workspace: damagedetection-hloj4 / project: damagelocation / version: 7`

Now useful because it has a Scratch class.
**Warning:** Many classes here are part labels (not damage labels) — drop them all.
Only use classes that are explicitly damage-typed.

```python
"source_classes": [
    "Back Door",          # 0  ← part label, DROP
    "Back Window",        # 1  ← part label, DROP
    "Broken Headlight",   # 2
    "Broken Windshield",  # 3
    "Bumper",             # 4  ← part label, DROP
    "Bumper Damage",      # 5
    "Car",                # 6  ← whole car, DROP
    "Dent",               # 7
    "Fender",             # 8  ← part label, DROP
    "Fender Damage",      # 9
    "Front Door",         # 10 ← part label, DROP
    "Front Headlight",    # 11 ← part label (not damage), DROP
    "Front Window",       # 12 ← part label, DROP
    "Front Windscreen",   # 13 ← part label (not damage), DROP
    "Rear Bumper",        # 14 ← part label, DROP
    "Rear Bumper Damage", # 15
    "Rear Fender",        # 16 ← part label, DROP
    "Rear Headlight",     # 17 ← part label, DROP
    "Scratch",            # 18
],
"class_remap": {
    2:  3,   # Broken Headlight  → light_damage
    3:  2,   # Broken Windshield → glass_damage
    5:  0,   # Bumper Damage     → dent
    7:  0,   # Dent              → dent
    9:  0,   # Fender Damage     → dent
    15: 0,   # Rear Bumper Damage→ dent
    18: 1,   # Scratch           → scratch
    # all others → DROP (part labels, not damage)
}
```

---

## Files to Update

### 1. `stage1_detection/dataset.py`
- Update `TARGET_CLASS_NAMES` from 4 entries to 5:
  ```python
  TARGET_CLASS_NAMES = ["dent", "scratch", "glass_damage", "light_damage", "mirror_damage"]
  ```
- Update `_D, _G, _L, _M` shorthand constants — add `_S = TARGET_ID["scratch"]`
- Replace the single Sindhu entry in `DATASETS` with all 4 datasets above
- The `download()` function currently only handles Roboflow (sindhu). CarDD is already
  on disk (downloaded via Kaggle). Add logic to skip download for datasets where
  `workspace` is None (Kaggle datasets — already on disk).
- The `prepare()` function should work as-is once DATASETS is updated.

### 2. `stage1_detection/config.yaml`
```yaml
nc: 5
names:
  0: dent
  1: scratch
  2: glass_damage
  3: light_damage
  4: mirror_damage
```

### 3. `stage1_detection/evaluate.py`
Update any hardcoded class name list from 4 entries to 5.
Check that the pass/fail summary uses `TARGET_CLASS_NAMES` dynamically
rather than a hardcoded list.

### 4. `stage1_detection/schemas.py`
`DamageRegion.class_name` is a string — no change needed.
Verify that `DetectionResult` doesn't hardcode class count anywhere.

### 5. `tests/test_stage1.py`
Update any test that checks for exactly 4 classes.

---

## Rebuild Steps (in order)

```bash
# 1. Download Roboflow datasets (sindhu already cached, skips; nivethetha + DamageLocation are new)
python stage1_detection/dataset.py download

# 2. Pool all 4 datasets, remap to 5-class taxonomy, stratified re-split
python stage1_detection/dataset.py prepare

# 3. Validate class distribution across splits — scratch should appear in all 3
python stage1_detection/dataset.py validate

# 4. Retrain with new 5-class config (Colab recommended)
python stage1_detection/train.py

# 5. Evaluate on new test split
python stage1_detection/evaluate.py
```

**Old `data/splits/` must be deleted before running `prepare()`** — it contains the
old 4-class labels and will conflict.

---

## Expected Class Distribution After Pooling

| Class | Sources | Expected coverage |
|---|---|---|
| dent | Sindhu + CarDD + nivethetha + DamageLocation | High — well covered |
| scratch | CarDD + nivethetha + DamageLocation | Medium — 3 sources |
| glass_damage | Sindhu + CarDD + nivethetha + DamageLocation | Medium-high |
| light_damage | Sindhu + CarDD + nivethetha + DamageLocation | Medium-high |
| mirror_damage | Sindhu + nivethetha only | Low — flag if < 200 boxes |

Run `validate` and check the class distribution printout. If mirror_damage is under
200 training boxes, note it but proceed — it's structurally limited by available data.

---

## What "Done" Looks Like

- [x] `dataset.py` has 4 entries in DATASETS, all remapping to 5-class taxonomy
- [x] `dataset.py prepare` runs clean, all 5 classes appear in train/val/test splits
- [x] `config.yaml` has `nc: 5` and correct class names
- [x] `train.py` runs with new config, W&B logs 5-class metrics
- [x] `evaluate.py` reports per-class mAP for all 5 classes
- [x] `tests/test_stage1.py` passes with updated class count
- [x] mAP@50 > 0.50 overall (minimum), scratch class mAP > 0.40 (new class baseline)

---

## Results (yolo11s-v2-5class, trained on Colab T4, 95 epochs)

Test split (728 images, 1339 instances, never used for tuning):

| Class | Images | Instances | P | R | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| all | 728 | 1339 | 0.722 | 0.664 | **0.705** | **0.485** |
| dent | 503 | 789 | 0.656 | 0.475 | 0.542 | 0.273 |
| scratch | 202 | 329 | 0.530 | 0.514 | 0.507 | 0.273 |
| glass_damage | 99 | 99 | 0.939 | 0.879 | 0.922 | 0.772 |
| light_damage | 108 | 112 | 0.757 | 0.752 | 0.787 | 0.571 |
| mirror_damage | 10 | 10 | 0.727 | 0.700 | 0.769 | 0.533 |

vs. v1 (4-class) baseline: mAP@50 0.515 → **0.705**, mAP@50-95 0.256 → **0.485**,
recall 0.459 → **0.664**. The new `scratch` class clears its 0.40 target at 0.507.

ONNX CPU latency: **35.7ms/image** (target <100ms).

**Weakest classes:** `dent` and `scratch` (0.51-0.54 mAP@50) — despite being the
most abundant classes by box count, they're visually ambiguous across the 4 pooled
datasets' annotation styles. `glass_damage`, `light_damage`, and `mirror_damage`
(despite being rarest) all score >0.75, since these damage types are visually
distinctive even with less data.

**Status: Stage 1 v2 complete.** This is the new baseline for Stage 2 crop generation.

## Key Decisions (Do Not Change Without Discussion)

1. **crack → dent** not its own class. Body surface cracks and dents share the same
   repair pathway (body shop, not glass shop). Keeping them merged avoids a weakly-
   covered class.
2. **tire_flat dropped** — one data source, different use case, not collision damage.
3. **CarDD is training data** — the OOD evaluation plan is cancelled. CarDD's value
   is in scratch and dataset diversity, not as a benchmark.
4. **DamageLocation part labels dropped** — only damage-typed classes survive remap.
   Part labels (Front Windscreen, Rear Headlight, etc.) label intact parts and would
   poison training if included.
