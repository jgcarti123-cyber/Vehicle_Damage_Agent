"""Vehicle Damage Assessment API — FastAPI wrapper for the Stage 1+2 pipeline.

Endpoints:
    GET  /health          — liveness + model-loaded check
    POST /assess          — upload image -> full assessment JSON
    GET  /routing-guide   — explains the three routing tiers

Confidence-based routing (applied per damage region):
    >= 0.85  auto_classify        AI decision, no human needed
    0.70-0.84 suggest_human_confirm  AI suggests, assessor confirms
    < 0.70   human_review         Route to human assessor queue

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import base64
import logging
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Ensure project root is importable.
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stage1_detection"))

from pipeline import (
    Aggregation, CostEstimateItem, PipelineResult, RegionResult,
    RepairCostBreakdown, SeverityAssessment, VLMSeverity, run_image,
)
from stage2_severity.infer import load_model as _load_severity

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STAGE1_WEIGHTS = _ROOT / "runs/detect/vehicle-damage-detection/yolo11s-v2-5class/weights/best.pt"
STAGE2_WEIGHTS = _ROOT / "runs/severity/efficientnet-b0-v6/weights/best.pt"
ALLOWED_TYPES  = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_MB    = 20
STATIC_DIR     = Path(__file__).parent / "static"

# Magic-byte signatures for accepted image formats.
_MAGIC: list[tuple[bytes, bytes | None]] = [
    (b"\xff\xd8\xff", None),          # JPEG
    (b"\x89PNG",      None),          # PNG
    (b"RIFF",         b"WEBP"),       # WEBP  (bytes 0-3 = RIFF, bytes 8-11 = WEBP)
    (b"BM",           None),          # BMP
]

# Simple in-memory rate limiter: max 10 /assess requests per IP per minute.
_rate_window  = 60.0
_rate_limit   = 10
_rate_buckets: dict[str, list[float]] = defaultdict(list)


def _check_magic(data: bytes) -> bool:
    for prefix, extra in _MAGIC:
        if data[:len(prefix)] == prefix:
            if extra is None:
                return True
            if data[8 : 8 + len(extra)] == extra:
                return True
    return False


def _enforce_rate_limit(client_ip: str) -> None:
    now   = time.monotonic()
    times = _rate_buckets[client_ip]
    times[:] = [t for t in times if now - t < _rate_window]
    if len(times) >= _rate_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded — maximum {_rate_limit} requests per minute.",
        )
    times.append(now)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class AssessmentResponse(BaseModel):
    """Full pipeline response returned to the client."""
    image_filename: str
    num_damages: int
    overall_severity: str = Field(..., description="mild | moderate | severe | unknown")
    overall_routing: str  = Field(..., description="auto_classify | suggest_human_confirm | human_review")
    severity_source: str  = Field(..., description="vlm | crop_heuristic")
    aggregation: Aggregation
    vlm_assessment: VLMSeverity | None = None
    cost_estimate: RepairCostBreakdown | None = None
    regions: list[RegionResult]
    detection_time_ms: float
    severity_time_ms: float
    vlm_time_ms: float
    cost_time_ms: float = 0.0
    total_time_ms: float
    stage1_model: str
    stage2_model: str
    annotated_image_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded annotated JPEG (only when include_annotated=true)"
    )

    @classmethod
    def from_pipeline(
        cls,
        result: PipelineResult,
        filename: str,
        annotated_b64: Optional[str] = None,
    ) -> "AssessmentResponse":
        return cls(
            image_filename=filename,
            num_damages=result.num_damages,
            overall_severity=result.overall_severity,
            overall_routing=result.overall_routing,
            severity_source=result.severity_source,
            aggregation=result.aggregation,
            vlm_assessment=result.vlm_assessment,
            cost_estimate=result.cost_estimate,
            regions=result.regions,
            detection_time_ms=result.detection_time_ms,
            severity_time_ms=result.severity_time_ms,
            vlm_time_ms=result.vlm_time_ms,
            cost_time_ms=result.cost_time_ms,
            total_time_ms=result.total_time_ms,
            stage1_model=result.stage1_model,
            stage2_model=result.stage2_model,
            annotated_image_b64=annotated_b64,
        )


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str
    stage1_weights: str
    stage2_weights: str


# ---------------------------------------------------------------------------
# App lifespan — load models once at startup
# ---------------------------------------------------------------------------

_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from ultralytics import YOLO

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[startup] Loading Stage 1 model from {STAGE1_WEIGHTS} ...")
    _models["yolo"] = YOLO(str(STAGE1_WEIGHTS))

    print(f"[startup] Loading Stage 2 model from {STAGE2_WEIGHTS} ...")
    _models["severity"], _models["device"] = _load_severity(STAGE2_WEIGHTS, device)

    print(f"[startup] Both models ready on {device}")
    yield
    _models.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Vehicle Damage Assessment API",
    description=(
        "Detects vehicle damage regions (Stage 1 — YOLOv11) and classifies "
        "each region's severity (Stage 2 — EfficientNet-B0). Returns a structured "
        "JSON report with per-region severity and confidence-based routing decisions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' data: blob:; "
        "script-src 'self' 'unsafe-inline'; "
        "connect-src 'self'"
    )
    return response


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def frontend():
    """Serve the single-page test frontend."""
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """Liveness check — confirms both models are loaded."""
    return HealthResponse(
        status="ok",
        models_loaded=bool(_models),
        device=str(_models.get("device", "not loaded")),
        stage1_weights=STAGE1_WEIGHTS.name,
        stage2_weights=STAGE2_WEIGHTS.name,
    )


@app.get("/routing-guide", tags=["meta"])
def routing_guide():
    """Explains the three routing tiers and their confidence thresholds."""
    return {
        "routing_tiers": [
            {
                "tier": "auto_classify",
                "confidence_range": ">= 0.85",
                "meaning": "AI is highly confident. No human review needed.",
                "action": "Accept AI decision directly.",
            },
            {
                "tier": "suggest_human_confirm",
                "confidence_range": "0.70 – 0.84",
                "meaning": "AI is moderately confident. Show result to assessor for quick confirmation.",
                "action": "Present AI suggestion with photo to assessor for one-click confirm/override.",
            },
            {
                "tier": "human_review",
                "confidence_range": "< 0.70",
                "meaning": "AI is uncertain. Full manual review required.",
                "action": "Route to human assessor queue with photo and AI reasoning.",
            },
        ],
        "overall_routing": "Most cautious tier across all detected regions is used as the overall routing.",
        "overall_severity": "Worst severity (mild < moderate < severe) across all detected regions.",
    }


@app.post("/assess", response_model=AssessmentResponse, tags=["assessment"])
async def assess(
    request: Request,
    file: UploadFile = File(..., description="Vehicle damage photo (JPEG / PNG / WEBP)"),
    include_annotated: bool = Query(default=False,
        description="Include base64-encoded annotated image in the response"),
    car_make: str  = Query(default="", max_length=80, description="Vehicle make, e.g. Maruti Suzuki"),
    car_model: str = Query(default="", max_length=80, description="Vehicle model, e.g. Swift"),
    car_year: Optional[int] = Query(default=None, description="Vehicle manufacturing year, e.g. 2021"),
    city: str      = Query(default="", max_length=80, description="City where repairs will be done, e.g. Mumbai"),
):
    """
    Assess vehicle damage from an uploaded photo.

    Returns:
    - Per-region damage type, severity, confidence, and routing decision
    - Overall severity and routing for the whole image
    - Optionally a base64-encoded annotated image with colored bounding boxes
    """
    client_ip = request.client.host if request.client else "unknown"
    _enforce_rate_limit(client_ip)

    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Retry in a moment.")

    # Validate Content-Type header (client-controlled — magic bytes checked below)
    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type!r}. Allowed: {sorted(ALLOWED_TYPES)}"
        )

    # Read and size-check
    data = await file.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_MB} MB."
        )

    # Verify actual file content via magic bytes (Content-Type is client-controlled)
    if not _check_magic(data):
        raise HTTPException(status_code=400, detail="File content does not match a supported image format.")

    # Write to temp dir, run pipeline, read annotated output
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / f"upload{suffix}"
        tmp_path.write_bytes(data)

        try:
            result = run_image(
                image_path=tmp_path,
                yolo_model=_models["yolo"],
                severity_model=_models["severity"],
                device=_models["device"],
                out_dir=Path(tmpdir) / "out",
                annotate=include_annotated,
                car_make=car_make,
                car_model=car_model,
                car_year=car_year,
                city=city,
            )
        except Exception as e:
            logger.error("Pipeline error for %s: %s", client_ip, e)
            raise HTTPException(status_code=500, detail="Assessment failed. Please try again.")

        annotated_b64 = None
        if include_annotated:
            annotated_path = Path(tmpdir) / "out" / f"upload_pipeline.jpg"
            if annotated_path.exists():
                annotated_b64 = base64.b64encode(annotated_path.read_bytes()).decode("utf-8")

    return AssessmentResponse.from_pipeline(
        result=result,
        filename=file.filename or "upload",
        annotated_b64=annotated_b64,
    )
