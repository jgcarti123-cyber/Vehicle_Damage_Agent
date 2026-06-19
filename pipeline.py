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
from collections import Counter
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
# Image-level severity aggregation
#
# The per-crop classifier only sees one region at a time — it cannot tell that
# a "moderate" crop is one of several damages on a destroyed front-end. Insurers
# assess severity at the VEHICLE level: repair costs accumulate across damaged
# components. We mirror that with a damage-score that sums per-component severity.
#
# Severity is driven ONLY by the damage score (actual per-component severity,
# accumulated). Area coverage is computed for transparency but deliberately does
# NOT drive the decision — it depends entirely on how zoomed-in the photo is (a
# close-up of two small scratches can fill 60% of the frame), which would wrongly
# escalate purely cosmetic damage. Many real scratches still escalate via the
# score (each mild = 1 point), independent of framing.
# ---------------------------------------------------------------------------

# Points roughly proportional to repair cost per damaged component.
SEVERITY_POINTS = {"mild": 1, "moderate": 3, "severe": 8}

# Total damage score thresholds (sum of per-region points).
SCORE_SEVERE   = 9    # e.g. 3 moderate components, or 1 severe + extras
SCORE_MODERATE = 5    # e.g. 2 moderate, or ~5 mild scratches adding up

# Soft-severe trigger: if the per-crop model puts at least this much probability on
# "severe" for ANY region — even when "moderate" is the argmax — escalate the whole
# vehicle to severe. A crushed total-loss panel often reads as "moderate 57% /
# severe 34%": the 34% is the real signal, and counting detections misses it.
SEVERE_PROB_TRIGGER = 0.30

# Whole-image VLM severity — sees the entire vehicle (not isolated crops), so it
# judges total-loss the way a human assessor does. Best-effort: falls back to the
# crop heuristic if no API key / the call fails.
VLM_SEVERITY_MODEL = "gpt-4o-mini"
VLM_COST_MODEL     = "gpt-4o-mini"
VLM_SEVERITY_PROMPT = """\
You are a senior motor-insurance assessor. Assess the OVERALL damage severity of
the vehicle in this photo at the VEHICLE level — judge the whole car, not one spot.

Severity levels (Indian motor insurance):
- mild:     cosmetic only, fully driveable (scratches, scuffs, small dents).
- moderate: one or more components need replacement/major repair, likely still
            driveable (bumper, headlight/taillight, fender, door panel).
- severe:   structural or safety-critical damage, possible total-loss — the car
            is unsafe or impossible to drive.

Pay special attention to TOTAL-LOSS indicators: crushed/exposed cabin, deployed
airbags, bent frame/chassis, roof or pillar crush, multiple severely crushed
panels, displaced wheels/suspension, fire damage.

Respond with valid JSON only, no other text:
{
  "severity": "<mild|moderate|severe>",
  "is_total_loss": <true|false>,
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence citing the specific visible evidence>"
}"""

VLM_COST_PROMPT_TEMPLATE = """\
You are an experienced Indian motor insurance surveyor and workshop cost estimator.

VEHICLE: {vehicle_info}
REPAIR LOCATION: {location_info}

AI SURVEYOR VERDICT (already assessed at the whole-vehicle level):
{surveyor_verdict}

DETECTED DAMAGE REGIONS (computer-vision bounding boxes — location is given as
row-column of the frame, plus the % of the frame each box covers):
{region_list}

─────────────────────────────────────────────────────────────────────
PARTS PRICE REFERENCE — OEM parts, Indian market 2024, incl. 18% GST
─────────────────────────────────────────────────────────────────────
Segment guide:
  Budget       → Alto K10, Kwid, S-Presso, WagonR (base), i10 Nios (base)
  Compact      → Swift, Baleno, i20, Altroz, Tiago, Polo, Punch (base)
  Compact SUV  → Venue, Sonet, Nexon, Brezza, Magnite, Amaze, Dzire, City
  Mid SUV      → Creta, Seltos, XUV300, Kushaq, Taigun, Harrier (base)
  Premium SUV  → XUV700, Fortuner, Innova Crysta, Thar 4×4, Safari, Endeavour
  Luxury       → Jeep Compass, BMW 3/5, Mercedes C/E, Audi A4/Q5+

Part (OEM replace cost)  | Budget   | Compact  | Cmpct SUV| Mid SUV  | Prem SUV | Luxury
-------------------------|----------|----------|----------|----------|----------|----------
Front bumper             | 3K–6K    | 5K–12K   | 8K–18K   | 12K–25K  | 20K–45K  | 40K–90K
Rear bumper              | 2.5K–5K  | 4K–10K   | 7K–15K   | 10K–22K  | 18K–40K  | 35K–80K
Bonnet / hood            | 8K–14K   | 12K–22K  | 18K–30K  | 25K–45K  | 45K–80K  | 80K–1.5L
Front door (per side)    | 8K–14K   | 12K–22K  | 16K–28K  | 22K–40K  | 40K–75K  | 80K–1.5L
Rear door (per side)     | 7K–12K   | 10K–18K  | 14K–25K  | 20K–35K  | 35K–65K  | 70K–1.2L
Front fender (per side)  | 4K–8K    | 6K–12K   | 8K–16K   | 12K–22K  | 20K–40K  | 40K–80K
Headlight assembly       | 2.5K–5K  | 4K–10K   | 8K–18K   | 15K–30K  | 25K–55K  | 50K–1.2L
Taillight assembly       | 1.5K–4K  | 3K–8K    | 5K–14K   | 10K–22K  | 18K–40K  | 35K–80K
Windshield (front)       | 5K–9K    | 8K–14K   | 10K–18K  | 14K–25K  | 20K–40K  | 40K–90K
Windshield (rear)        | 4K–8K    | 6K–12K   | 8K–16K   | 12K–22K  | 18K–35K  | 35K–80K
Side mirror (per side)   | 1K–2.5K  | 1.5K–4K  | 3K–8K    | 5K–12K   | 8K–20K   | 18K–45K
Alloy wheel (each)       | 3K–5K    | 4K–8K    | 6K–12K   | 8K–18K   | 15K–30K  | 30K–70K
Radiator                 | 4K–8K    | 6K–12K   | 8K–18K   | 12K–25K  | 22K–45K  | 45K–1L
A/C condenser            | 5K–10K   | 8K–16K   | 12K–22K  | 18K–32K  | 28K–55K  | 55K–1.2L

LABOUR RATES (workshop, not dealership):
  Tier-1 (Mumbai, Delhi, Bengaluru, Hyderabad, Chennai, Pune, Kolkata): ₹600–1,100/hr
  Tier-2 (Ahmedabad, Jaipur, Lucknow, Chandigarh, Indore, Kochi, Nagpur, Coimbatore): ₹400–750/hr
  Tier-3 / small towns: ₹250–500/hr

PAINT / REFINISHING (materials + labour, per panel):
  Tier-1: ₹3,500–6,000 | Tier-2: ₹2,500–4,500 | Tier-3: ₹1,800–3,500

REPAIR vs REPLACE guide:
  Scratch only (no dent)        → refinishing only
  Small dent < 5 cm, no cracks  → PDR repair ₹1,500–5,000 flat
  Dent with paint damage         → panel repair + refinish
  Crack / sharp deformation      → replacement
  Structural deformation         → replacement + frame check
  Total loss                     → salvage value = 5–15% of IDV
{unknown_vehicle_note}
─────────────────────────────────────────────────────────────────────

TASK: Study the photo together with the bounding boxes. Itemise EVERY component
that is visibly damaged — use the boxes as a guide, but trust your own reading of
the photo over the box count (the detector often merges or misses parts).

CRITICAL — do NOT under-itemise:
- A bounding box labelled "dent" on a crushed area usually means the WHOLE panel
  plus the parts behind it are destroyed. One box over the front end is NOT "one
  bumper".
- If the front end is crushed, itemise each affected part separately: front bumper,
  bonnet/hood, grille, BOTH headlights, radiator, A/C condenser, BOTH front fenders,
  and add a "Front structure / crash frame" line if metal is bent or the cabin is
  pushed in. Do the same for rear-end or side impacts.
- When the AI surveyor verdict above says TOTAL LOSS or severe structural damage,
  the total MUST reflect a major rebuild (typically ₹1.5L+ for a compact car, often
  exceeding the car's value). A few thousand rupees is wrong for a total loss.
- Merge boxes only when they clearly sit on the SAME single component.

Choose repair/replace/refinish per item and price using the table above. If the
vehicle segment is known use that column; otherwise default to Compact SUV.
Set is_total_loss=true if the surveyor flagged it or the damage is a clear write-off.

Respond with valid JSON only — no other text:
{{
  "items": [
    {{
      "region_ids": [<int>, ...],
      "component": "<specific part e.g. Front bumper, Left front door>",
      "damage_type": "<dent|scratch|crack|broken|deformed|shattered|other>",
      "repair_action": "<repair|replacement|refinishing|mechanical repair|total loss>",
      "cost_min_inr": <integer>,
      "cost_max_inr": <integer>
    }}
  ],
  "total_min_inr": <integer>,
  "total_max_inr": <integer>,
  "is_total_loss": <true|false>,
  "notes": "<one sentence: segment assumed + key caveat>"
}}"""

_openai_client = None
_openai_checked = False


def _sanitize_input(value: str, max_len: int = 80) -> str:
    """Strip newlines and truncate user-supplied text before it enters an LLM prompt."""
    return value.replace("\n", " ").replace("\r", " ").strip()[:max_len]


def _get_openai_client():
    """Lazily build a cached OpenAI client, or None if unavailable."""
    global _openai_client, _openai_checked
    if _openai_checked:
        return _openai_client
    _openai_checked = True
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("[vlm] OPENAI_API_KEY not set — whole-image severity disabled.")
            return None
        from openai import OpenAI
        _openai_client = OpenAI(api_key=key, timeout=20.0)
    except Exception as e:
        print(f"[vlm] OpenAI client unavailable: {e}")
        _openai_client = None
    return _openai_client


def assess_whole_image_vlm(image_path: Path) -> VLMSeverity | None:
    """Send the full image to the VLM for a vehicle-level severity call.

    Returns None on any failure (no key, network error, bad response) so the
    caller can fall back to the crop-based heuristic."""
    client = _get_openai_client()
    if client is None:
        return None
    try:
        import base64, mimetypes
        img_bytes = Path(image_path).read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        resp = client.chat.completions.create(
            model=VLM_SEVERITY_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"}},
                    {"type": "text", "text": VLM_SEVERITY_PROMPT},
                ],
            }],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=200,
        )
        data = json.loads(resp.choices[0].message.content)
        sev = data.get("severity")
        if sev not in SEVERITY_ORDER:
            return None
        is_total = bool(data.get("is_total_loss", False))
        if is_total:
            sev = "severe"  # a write-off is always severe
        return VLMSeverity(
            severity=sev,
            is_total_loss=is_total,
            confidence=float(data.get("confidence", 0.0)),
            reasoning=str(data.get("reasoning", "")),
            model=VLM_SEVERITY_MODEL,
        )
    except Exception as e:
        print(f"[vlm] whole-image assessment failed: {e}")
        return None

def _describe_box(bbox: list[float], image_w: int, image_h: int) -> str:
    """Human-readable location + frame coverage for a detection box, so the cost
    VLM understands where each damage sits and how much of the car it spans."""
    if image_w <= 0 or image_h <= 0:
        return "unknown location"
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / image_w
    cy = (y1 + y2) / 2 / image_h
    col = "left" if cx < 0.4 else ("right" if cx > 0.6 else "centre")
    row = "upper" if cy < 0.4 else ("lower" if cy > 0.6 else "mid")
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / (image_w * image_h)
    return f"{row}-{col}, covers {area:.0%} of frame"


def estimate_repair_cost(
    image_path: Path,
    regions: list["RegionResult"],
    car_make: str = "",
    car_model: str = "",
    car_year: int | None = None,
    city: str = "",
    image_w: int = 0,
    image_h: int = 0,
    vlm_severity: "VLMSeverity | None" = None,
) -> "RepairCostBreakdown | None":
    """Estimate per-component repair costs using gpt-4o-mini with the full image.

    Receives the detection bounding boxes (location + frame coverage) and the
    whole-vehicle surveyor verdict so the estimate reflects the FULL extent of
    damage — not a single merged part. Returns None on failure (best-effort)."""
    client = _get_openai_client()
    if client is None:
        return None
    try:
        import base64, mimetypes
        # Build region list with spatial context from each bounding box.
        lines = []
        for i, r in enumerate(regions, 1):
            sev = r.severity.severity if r.severity else "unknown"
            loc = _describe_box(r.bbox_xyxy, image_w, image_h)
            lines.append(f"  Box {i}: {r.class_name}, severity={sev}, "
                         f"location={loc}, det_conf={r.detection_confidence:.0%}")
        region_list = "\n".join(lines) if lines else "  No boxes detected."

        # Surveyor verdict block — keeps cost consistent with the severity call.
        if vlm_severity is not None:
            tl = "YES — write-off" if vlm_severity.is_total_loss else "no"
            surveyor_verdict = (
                f"  Severity: {vlm_severity.severity.upper()}  |  Total loss: {tl}  |  "
                f"Confidence: {vlm_severity.confidence:.0%}\n"
                f"  Surveyor note: {vlm_severity.reasoning}"
            )
        else:
            surveyor_verdict = "  (no whole-vehicle verdict available — judge from the photo)"

        # Sanitize user-supplied strings before they enter the LLM prompt.
        safe_make  = _sanitize_input(car_make)
        safe_model = _sanitize_input(car_model)
        safe_city  = _sanitize_input(city)

        # Build vehicle info
        parts = [safe_make, safe_model]
        if car_year:
            parts.append(f"({car_year})")
        vehicle_info = " ".join(p for p in parts if p).strip() or \
            "Unknown — defaulting to Compact SUV segment for pricing"

        # Build location / labour tier info
        location_info = safe_city or "Unknown — defaulting to Tier-2 city labour rates"

        # Note when vehicle details are absent
        if not (safe_make or safe_model):
            unknown_vehicle_note = (
                "\nNOTE: No vehicle details provided. Compact SUV segment prices used as "
                "default. Actual cost may differ significantly for budget or luxury vehicles."
            )
        else:
            unknown_vehicle_note = ""

        prompt = VLM_COST_PROMPT_TEMPLATE.format(
            region_list=region_list,
            surveyor_verdict=surveyor_verdict,
            vehicle_info=vehicle_info,
            location_info=location_info,
            unknown_vehicle_note=unknown_vehicle_note,
        )

        # detail="high" — the model needs to actually see crush extent to itemise
        # the parts behind a damaged panel; "low" downsamples too far for that.
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
        mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        resp = client.chat.completions.create(
            model=VLM_COST_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1300,
        )
        data = json.loads(resp.choices[0].message.content)
        items = []
        for it in data.get("items", []):
            items.append(CostEstimateItem(
                region_ids=[int(x) for x in it.get("region_ids", [])],
                component=str(it.get("component", "Unknown")),
                damage_type=str(it.get("damage_type", "unknown")),
                repair_action=str(it.get("repair_action", "repair")),
                cost_min_inr=int(it.get("cost_min_inr", 0)),
                cost_max_inr=int(it.get("cost_max_inr", 0)),
            ))
        return RepairCostBreakdown(
            items=items,
            total_min_inr=int(data.get("total_min_inr", 0)),
            total_max_inr=int(data.get("total_max_inr", 0)),
            is_total_loss=bool(data.get("is_total_loss", False)),
            notes=str(data.get("notes", "")),
            model=VLM_COST_MODEL,
        )
    except Exception as e:
        print(f"[cost] repair cost estimation failed: {e}")
        return None


# Fixed detection confidence — low so ALL damage is caught consistently. Severity
# must not depend on a user-tunable threshold (a high threshold drops low-confidence
# damages, shrinks the score, and makes a totalled car read as moderate).
DETECTION_CONF = 0.15


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


class VLMSeverity(BaseModel):
    """Whole-image severity from a vision LLM that sees the entire vehicle."""
    severity: str = Field(..., description="mild | moderate | severe")
    is_total_loss: bool = Field(..., description="True if the car appears to be a write-off")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    model: str


class CostEstimateItem(BaseModel):
    """Per-component repair cost estimated by the VLM."""
    region_ids: list[int] = Field(default_factory=list, description="Which detected regions this covers")
    component: str = Field(..., description="Car part name, e.g. 'Front bumper'")
    damage_type: str
    repair_action: str = Field(..., description="repair | replacement | refinishing | mechanical repair | total loss")
    cost_min_inr: int
    cost_max_inr: int


class RepairCostBreakdown(BaseModel):
    """Full itemised repair cost estimate from VLM analysis."""
    items: list[CostEstimateItem]
    total_min_inr: int
    total_max_inr: int
    is_total_loss: bool = False
    notes: str
    model: str


class Aggregation(BaseModel):
    """Image-level severity reasoning — how per-crop results roll up to the car."""
    overall: str = Field(..., description="Vehicle-level severity: mild | moderate | severe | unknown")
    damage_score: int = Field(..., description="Sum of per-region severity points")
    coverage: float = Field(..., description="Fraction of image area covered by damage (union)")
    counts: dict[str, int] = Field(..., description="Per-severity region counts")
    worst_individual: str = Field(..., description="Worst single-crop severity")
    escalated: bool = Field(..., description="True if aggregate severity exceeds worst individual crop")
    reason: str = Field(..., description="Human-readable explanation of the overall severity")


class PipelineResult(BaseModel):
    image_path: str
    image_width: int
    image_height: int
    num_damages: int
    regions: list[RegionResult]
    overall_severity: str = Field(..., description="Vehicle-level severity (VLM-authoritative when available)")
    overall_routing: str  = Field(..., description="Most cautious routing across all regions")
    severity_source: str  = Field(..., description="vlm | crop_heuristic — what drove overall_severity")
    aggregation: Aggregation
    vlm_assessment: VLMSeverity | None = Field(
        default=None, description="Whole-image VLM severity (None if VLM unavailable)"
    )
    cost_estimate: "RepairCostBreakdown | None" = Field(
        default=None, description="VLM repair cost breakdown (None if unavailable)"
    )
    detection_time_ms: float
    severity_time_ms: float
    vlm_time_ms: float = 0.0
    cost_time_ms: float = 0.0
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


def _union_coverage(regions: list["RegionResult"], image_w: int, image_h: int) -> float:
    """Fraction of the image area covered by the union of damage boxes.

    Uses a coarse grid rasterisation so overlapping boxes are not double-counted
    (cheap, deterministic, and accurate enough for an escalation signal)."""
    if not regions or image_w <= 0 or image_h <= 0:
        return 0.0
    grid = 100
    cells = [[False] * grid for _ in range(grid)]
    for r in regions:
        x1, y1, x2, y2 = r.bbox_xyxy
        gx1 = max(0, int(x1 / image_w * grid))
        gy1 = max(0, int(y1 / image_h * grid))
        gx2 = min(grid, int(x2 / image_w * grid) + 1)
        gy2 = min(grid, int(y2 / image_h * grid) + 1)
        for gy in range(gy1, gy2):
            for gx in range(gx1, gx2):
                cells[gy][gx] = True
    covered = sum(row.count(True) for row in cells)
    return covered / (grid * grid)


def aggregate_severity(regions: list["RegionResult"], image_w: int, image_h: int) -> Aggregation:
    """Roll per-crop severities up to a vehicle-level severity.

    Escalates beyond the worst single crop when damage accumulates across many
    components (damage_score), mirroring how insurers sum repair costs. Coverage
    is reported for context but does not drive the decision (framing-dependent)."""
    scored = [r for r in regions if r.severity]
    if not scored:
        return Aggregation(overall="unknown", damage_score=0, coverage=0.0, counts={},
                           worst_individual="unknown", escalated=False,
                           reason="No damage regions with severity to assess.")

    sevs   = [r.severity.severity for r in scored]
    counts = dict(Counter(sevs))
    score  = sum(SEVERITY_POINTS.get(s, 0) for s in sevs)
    coverage = _union_coverage(scored, image_w, image_h)
    worst  = worst_severity(sevs)

    # Soft-severe: the highest "severe" probability the model assigned to any crop.
    # This catches total-loss panels the per-crop model hedges on (moderate argmax
    # but heavy severe mass) without depending on how many boxes the detector found.
    max_severe_prob = max(
        (r.severity.probabilities.get("severe", 0.0) for r in scored), default=0.0
    )
    soft_severe = max_severe_prob >= SEVERE_PROB_TRIGGER

    # Severity is driven by per-crop severity, soft-severe probability, and the
    # accumulated damage score. Coverage is reported but never escalates.
    if counts.get("severe", 0) >= 1 or soft_severe or score >= SCORE_SEVERE:
        overall = "severe"
    elif counts.get("moderate", 0) >= 1 or score >= SCORE_MODERATE:
        overall = "moderate"
    else:
        overall = "mild"

    escalated = SEVERITY_ORDER[overall] > SEVERITY_ORDER[worst]

    # Build an audit-friendly reason string.
    parts = []
    if counts.get("severe", 0):
        parts.append(f"{counts['severe']} severe region(s)")
    if counts.get("moderate", 0):
        parts.append(f"{counts['moderate']} moderate region(s)")
    if counts.get("mild", 0):
        parts.append(f"{counts['mild']} mild region(s)")
    detail = ", ".join(parts)
    # Note the dominant driver of the result.
    if soft_severe and counts.get("severe", 0) == 0:
        driver = f"a region reads {max_severe_prob:.0%} likely severe"
    elif score >= SCORE_SEVERE and counts.get("severe", 0) == 0:
        driver = f"damage score {score} across components"
    else:
        driver = f"damage score {score}"
    if escalated:
        reason = (f"{len(scored)} damages ({detail}); {driver} "
                  f"→ escalated to {overall.upper()} (worst single crop was {worst}).")
    else:
        reason = (f"{len(scored)} damages ({detail}); {driver} → {overall.upper()}.")

    return Aggregation(overall=overall, damage_score=score, coverage=round(coverage, 3),
                       counts=counts, worst_individual=worst,
                       escalated=escalated, reason=reason)


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
    detection_conf: float = DETECTION_CONF,
    annotate: bool = True,
    use_vlm: bool = True,
    stage1_name: str = "yolo11s-v2-5class",
    stage2_name: str = "efficientnet-b0-v6",
    car_make: str = "",
    car_model: str = "",
    car_year: int | None = None,
    city: str = "",
) -> PipelineResult:
    t_start = time.perf_counter()

    detection, det_ms = _detect(yolo_model, image_path, out_dir, detection_conf, annotate)
    regions, sev_ms   = _classify_regions(detection, severity_model, device)

    routings = [r.severity.routing for r in regions if r.severity]
    aggregation = aggregate_severity(regions, detection.image_width, detection.image_height)

    # Whole-image VLM calls (both best-effort). Run severity FIRST, then feed its
    # verdict into the cost call so the two never contradict (e.g. surveyor says
    # total-loss while cost shows a single bumper). The cost call also receives the
    # detection boxes (location + coverage) to itemise the full extent of damage.
    vlm_assessment = None
    cost_estimate   = None
    vlm_ms  = 0.0
    cost_ms = 0.0
    if use_vlm:
        t_vlm = time.perf_counter()
        vlm_assessment = assess_whole_image_vlm(image_path)
        vlm_ms = (time.perf_counter() - t_vlm) * 1000.0

        t_cost = time.perf_counter()
        cost_estimate = estimate_repair_cost(
            image_path, regions, car_make, car_model, car_year, city,
            image_w=detection.image_width, image_h=detection.image_height,
            vlm_severity=vlm_assessment,
        )
        cost_ms = (time.perf_counter() - t_cost) * 1000.0

    # Reconcile: take the WORST of {VLM, crop-heuristic}. Never under-call a
    # total loss — over-calling only routes to a human, which is the safe direction.
    if vlm_assessment is not None:
        overall_sev = worst_severity([aggregation.overall, vlm_assessment.severity])
        severity_source = "vlm" if SEVERITY_ORDER.get(vlm_assessment.severity, -1) >= \
            SEVERITY_ORDER.get(aggregation.overall, -1) else "crop_heuristic"
    else:
        overall_sev = aggregation.overall
        severity_source = "crop_heuristic"

    overall_routing = worst_routing(routings) if routings else "human_review"
    # High-stakes calls always get a human: a severe/total-loss result, or any
    # case where the vehicle-level severity exceeded the worst single crop.
    if overall_sev == "severe" or aggregation.escalated or (
        vlm_assessment is not None and vlm_assessment.is_total_loss):
        overall_routing = "human_review"

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
        severity_source=severity_source,
        aggregation=aggregation,
        vlm_assessment=vlm_assessment,
        cost_estimate=cost_estimate,
        detection_time_ms=round(det_ms, 1),
        severity_time_ms=round(sev_ms, 1),
        vlm_time_ms=round(vlm_ms, 1),
        cost_time_ms=round(cost_ms, 1),
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
    print(f"Overall  : {result.overall_severity.upper()}  [{result.overall_routing}]  "
          f"(source: {result.severity_source})")
    if result.vlm_assessment:
        v = result.vlm_assessment
        tl = "  TOTAL-LOSS" if v.is_total_loss else ""
        print(f"VLM      : {v.severity.upper()} {v.confidence:.0%}{tl} — {v.reasoning}")
    print(f"Crops    : {result.aggregation.reason}")
    print(f"Timing   : detection={result.detection_time_ms:.0f}ms  "
          f"severity={result.severity_time_ms:.0f}ms  "
          f"vlm={result.vlm_time_ms:.0f}ms  "
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
    p.add_argument("--conf",     type=float, default=DETECTION_CONF,
                   help="YOLO detection threshold (default low to catch all damage)")
    p.add_argument("--device",   type=str,   default=None)
    p.add_argument("--json",     action="store_true", help="Print JSON output")
    p.add_argument("--no-annotate", action="store_true")
    p.add_argument("--no-vlm", action="store_true",
                   help="Skip the whole-image VLM call (offline / crop-heuristic only)")
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
            use_vlm=not args.no_vlm,
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
