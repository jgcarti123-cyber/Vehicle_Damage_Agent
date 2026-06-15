"""Pydantic v2 output schemas for Stage 1 (damage detection).

These are the contract between Stage 1 and downstream stages (Stage 2 severity
classifier, Stage 4 LangGraph agent). Keep them strictly typed so consumers do
not need defensive coding around detection output.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DamageRegion(BaseModel):
    """A single detected damage region within an image."""

    class_id: int = Field(..., ge=0, description="Integer class index from config.yaml")
    class_name: str = Field(..., description='Human-readable class, e.g. "scratch"')
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox_xyxy: list[float] = Field(
        ..., min_length=4, max_length=4, description="[x1, y1, x2, y2] in pixel coords"
    )
    bbox_xywh_norm: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x_center, y_center, w, h] normalised YOLO format (0-1)",
    )
    crop_path: str | None = Field(
        default=None, description="Path to the saved crop fed into Stage 2"
    )


class DetectionResult(BaseModel):
    """Full detection output for one image."""

    image_path: str
    image_width: int = Field(..., gt=0)
    image_height: int = Field(..., gt=0)
    num_damages: int = Field(..., ge=0)
    regions: list[DamageRegion]
    inference_time_ms: float = Field(..., ge=0.0)
    model_version: str = Field(..., description="Model weights identifier / tag")
