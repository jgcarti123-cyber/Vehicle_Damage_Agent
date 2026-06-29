# AEGIS — AI Vehicle Damage Assessment & Repair-Cost Engine

> An end-to-end computer-vision + LLM system that detects vehicle crash damage from a
> single photo, grades its severity at the *vehicle* level, estimates the repair bill
> at Indian workshop rates, and routes uncertain cases to a human assessor.

**Stack:** Python · PyTorch · Ultralytics YOLOv11 · EfficientNet-B0 · OpenAI GPT-4o-mini (Vision) · FastAPI · Pydantic v2 · OpenCV · Vanilla JS/CSS frontend · Weights & Biases

---

## 1. Problem

Motor-insurance claims in India are assessed manually: a surveyor inspects each
damaged car, decides severity, and itemises a repair estimate. This is slow,
subjective, and inconsistent. The goal was a system that:

1. **Detects** every damaged component in a photo.
2. **Grades severity** the way an insurer does — at the *whole-vehicle* level, not
   per scratch (repair cost accumulates across components).
3. **Estimates the repair cost** in INR using realistic, segment-aware workshop rates.
4. **Knows when it is unsure** and escalates those cases to a human instead of
   silently guessing.

---

## 2. System Architecture

A four-stage pipeline, each stage feeding the next:

```
                 ┌──────────────────────────────────────────────────────────┐
   Photo ───────▶│  STAGE 1 — Detection (YOLOv11s, 5 classes)                │
                 │  Finds damage regions → bounding boxes + crops            │
                 └───────────────┬──────────────────────────────────────────┘
                                 ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  STAGE 2 — Severity per crop (EfficientNet-B0, 3 classes) │
                 │  Each crop → mild / moderate / severe + probabilities     │
                 └───────────────┬──────────────────────────────────────────┘
                                 ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  STAGE 3 — Vehicle-level judgement                        │
                 │  (a) Whole-image VLM surveyor (GPT-4o-mini)               │
                 │  (b) CV aggregation of per-crop results                   │
                 │  Reconciled into one verdict + human-review routing       │
                 └───────────────┬──────────────────────────────────────────┘
                                 ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  STAGE 4 — Repair-cost estimation (GPT-4o-mini + pricing) │
                 │  Itemised INR breakdown at Indian workshop rates          │
                 └──────────────────────────────────────────────────────────┘

   Served by a FastAPI backend behind a single-page web app.
```

---

## 3. Stage 1 — Damage Detection (YOLOv11s)

**Model:** `yolo11s` fine-tuned to a unified **5-class** damage taxonomy:
`dent · scratch · glass_damage · light_damage · mirror_damage`.

**The hard part was the data.** No single public dataset covers Indian crash photos
well, so I **pooled four heterogeneous datasets** (Roboflow `sindhu v9`,
Kaggle `CarDD`, Roboflow `nivethetha v1`, Roboflow `damagelocation v7`), each with
its own incompatible label scheme. I built a dataset-prep pipeline (`dataset.py`)
that:

- Downloads each source and applies a **per-dataset class remap** into the unified
  5-class taxonomy (dropping classes that don't generalise, e.g. `tire_flat`).
- Pools all images, namespaces filenames to avoid collisions, and produces a
  **stratified 80/10/10 train/val/test split**.
- Runs **offline augmentation** (albumentations) plus Ultralytics' built-in
  online augmentation (mosaic, mixup, HSV jitter, rotation, scale).

Trained 100 epochs at 640px on Apple Silicon (MPS) with early stopping, with
**Weights & Biases** experiment tracking.

---

## 4. Stage 2 — Severity Classification (EfficientNet-B0)

Each detected crop is classified **mild / moderate / severe** (`total_loss` is
merged into `severe`) by an EfficientNet-B0 fine-tuned from ImageNet weights.

**Labelling without a labelling team:** Hand-labelling thousands of crops by
severity was infeasible, so I used **VLM pseudo-labelling** — GPT-4o-mini labels each
crop (`label_crops.py`), then a **confidence filter** keeps only high-agreement
labels (`labels_v6_filtered.csv`). This is a weak-supervision pipeline that turned an
unlabelled crop pool into a clean training set cheaply.

**Accuracy techniques to handle severe class imbalance** (severe crops are rare):

| Technique | Why |
|---|---|
| **Focal loss** (γ=2) | Down-weights easy examples, focuses on the hard minority (severe) |
| **Inverse-frequency class weighting** | Stops the majority class from dominating the loss |
| **WeightedRandomSampler oversampling** | Each batch sees ~equal class representation |
| **Two-phase fine-tuning** | Freeze backbone for 5 epochs (train head only), then unfreeze with backbone LR = head LR ÷ 10 — avoids wrecking pretrained features early |
| **Per-class accuracy evaluation** | Tracks minority-class recall, not just overall accuracy (which imbalance inflates) |

---

## 5. Stage 3 — Vehicle-Level Judgement (the accuracy core)

A per-crop classifier has a blind spot: it sees one region at a time, so it can't
tell that a "moderate" crop is one of several on a destroyed front-end. Insurers
judge severity at the **vehicle level**. I solved this with a **two-signal ensemble**:

### Signal A — Whole-image VLM surveyor
The full photo is sent to GPT-4o-mini with a senior-assessor prompt that judges
overall severity and flags total-loss indicators (crushed cabin, deployed airbags,
bent frame, displaced wheels). Because it sees the **whole car in context**, it is
the **authoritative judge** for the moderate↔severe call.

### Signal B — CV aggregation
Per-crop results are rolled up with:
- A **damage score** (mild=1, moderate=3, severe=8) for the mild→moderate step.
- A **soft-severe trigger**: if any crop carries ≥30% probability on `severe` (even
  when "moderate" is the argmax), that hedged signal is treated as severe — this
  catches total-loss panels the per-crop model is unsure about.

### Reconciliation
The two signals are combined **worst-of**, with one critical calibration:
**SEVERE is only reached on hard evidence** — an actual severe crop, a soft-severe
crop, or the VLM's total-loss/severe verdict. **A pile of moderate components can
never escalate to severe by count alone**, because that is a repairable car, not a
write-off. (See §7 — this fix removed a real false-positive class.)

### Confidence-based human-in-the-loop routing
Every result is routed by confidence so the system **defers when unsure**:

| Confidence | Route | Meaning |
|---|---|---|
| ≥ 0.85 | `auto_classify` | Accept the AI decision |
| 0.70 – 0.84 | `suggest_human_confirm` | One-click assessor confirmation |
| < 0.70 | `human_review` | Full manual review |

Any severe / total-loss / escalated case is forced to human review — over-calling
only costs a human glance, which is the safe direction for an insurer.

---

## 6. Stage 4 — Repair-Cost Estimation Engine

The cost model is GPT-4o-mini grounded in a **hand-built Indian pricing reference**
baked into the prompt:

- **All-in workshop costs** (part + paint + labour, incl. 18% GST) across
  **6 vehicle segments** (Budget → Luxury) × ~20 components.
- **Spatial grounding:** every detection box is converted to a human-readable
  location + frame-coverage descriptor (`_describe_box()`) so the model knows
  *which* panel and *how much* of the car is affected.
- **Sequential after severity:** the cost call receives the surveyor's verdict, so
  the estimate can never contradict it (e.g. "total loss" + "₹8K bumper").
- **Anti-under-itemisation rules:** one box over a crushed front-end must expand into
  every destroyed component (bumper + bonnet + grille + both headlights + radiator +
  condenser + both fenders + crash frame), plus **mandatory total-loss lines**
  (airbags, engine diagnostic, chassis straightening, full respray).
- **Segment & city aware:** optional make/model/year + repair-city inputs select the
  correct pricing column and labour tier (a Swift bumper ≠ a Fortuner bumper).

---

## 7. How Accuracy Was Achieved — Iterative, Error-Driven Calibration

Beyond model training, accuracy came from **closing the loop on real failure cases**:

| Failure observed | Root cause | Fix | Result |
|---|---|---|---|
| Boxes drawn on **people/bystanders** at very low threshold | Detector firing on clothing/bikes | Calibrated detection confidence to **0.15** (balance: catch all crash damage, reject background) | Clean detections |
| Total-loss car costed as a **single ₹8K bumper** | Cost model had no spatial context and ran *parallel* to severity, so it never saw the total-loss verdict | Added bbox spatial context, made cost **sequential** after severity, injected the surveyor verdict, added anti-under-itemisation rules | Itemised, realistic totals |
| Repair totals **far too low** vs reality | Pricing table was *parts only* (no paint/labour) and missing total-loss lines | Rebuilt table as **all-in** costs; added grille, crash-frame, chassis, paint, airbags, engine lines; total-loss compact floor raised to ₹2.5L | Estimates match real invoices |
| **Moderate cars flagged SEVERE** (false-severe) | CV aggregation summed 3 moderate + 1 mild = score 10, crossed an old SEVERE score threshold, and overrode the VLM's correct "moderate" | **Removed the damage-score path to SEVERE** entirely; severe now requires hard per-crop evidence or the VLM verdict | False-severe class eliminated; verdict now agrees with the VLM |

This error-driven calibration is what moved the system from "demo that mostly works"
to something whose verdicts and costs are defensible to an insurer.

---

## 8. Production Hardening (Security)

I ran a full security pass on the serving layer and fixed:

- **XSS** — all LLM/server-generated text is HTML-escaped before DOM insertion
  (LLM output → `innerHTML` was a prompt-injection-to-XSS chain).
- **Prompt injection** — user-supplied make/model/city are sanitised (newline-stripped,
  length-capped) before entering the LLM prompt.
- **Rate limiting** — token-bucket limiter (10 req/IP/min) protects the paid VLM calls.
- **File-upload validation** — magic-byte checking, not just the client-controlled
  `Content-Type`; 20 MB cap enforced before disk write.
- **Information-leak prevention** — generic client errors, full detail logged server-side.
- **Security headers** — CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy.
- **Secrets hygiene** — all API keys in a gitignored `.env`, never committed.

---

## 9. Engineering Highlights

- **Strictly-typed contracts** between every stage via Pydantic v2 schemas — no
  defensive dict-poking downstream.
- **Graceful degradation** — every external (VLM) call is best-effort; on any failure
  the pipeline falls back to the CV-only path and still returns a result.
- **Single-process model serving** — both models load once at FastAPI startup
  (lifespan), auto-selecting MPS / CUDA / CPU.
- **Self-contained frontend** — no build step; real-time pipeline-stage animation,
  annotated detection overlay, severity verdict card, and itemised cost table.

---

## 10. Resume Bullet Points (copy-ready)

> **AI Vehicle Damage Assessment & Repair-Cost Engine** — *Python, PyTorch, YOLOv11, EfficientNet-B0, GPT-4o-mini Vision, FastAPI*
>
> - Built an end-to-end 2-stage CV pipeline (YOLOv11 detection → EfficientNet-B0
>   severity) that detects vehicle crash damage and grades severity from a single
>   photo, served via a FastAPI backend and a custom web app.
> - Unified **4 incompatible public datasets** into a single 5-class detection
>   taxonomy with per-dataset class remapping and stratified splitting.
> - Eliminated manual labelling by building a **VLM weak-supervision pipeline**
>   (GPT-4o-mini pseudo-labels + confidence filtering) to train the severity model;
>   handled severe class imbalance with focal loss, class weighting, oversampling,
>   and two-phase fine-tuning.
> - Designed a **two-signal ensemble** (per-crop classifier + whole-image VLM
>   surveyor) with worst-of reconciliation and **confidence-based human-in-the-loop
>   routing**, mirroring how insurers assess severity at the vehicle level.
> - Engineered an **LLM repair-cost estimator** grounded in a segment-aware Indian
>   workshop pricing table with spatial bounding-box context, producing itemised
>   INR estimates that match real-world invoices.
> - Drove accuracy through **iterative, error-driven calibration** — eliminated a
>   false-severe class, fixed systematic cost under-estimation, and tuned detection
>   thresholds against real crash photos.
> - Hardened the serving layer against **XSS, prompt injection, and DoS** (rate
>   limiting, magic-byte upload validation, CSP, secrets hygiene).

---

*Note: add your actual metrics from the Weights & Biases runs — Stage 1 mAP@50 /
mAP@50-95, and Stage 2 overall + per-class accuracy on the held-out test split —
to make the impact quantitative.*
