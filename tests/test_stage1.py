"""Stage 1 unit tests: schemas, config consistency, class merge, data splits."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

import dataset
from schemas import DamageRegion, DetectionResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "stage1_detection" / "config.yaml"
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"


# --- Schemas ---------------------------------------------------------------
def test_detection_result_valid():
    region = DamageRegion(
        class_id=0,
        class_name="dent",
        confidence=0.91,
        bbox_xyxy=[10.0, 20.0, 100.0, 200.0],
        bbox_xywh_norm=[0.5, 0.5, 0.2, 0.3],
    )
    result = DetectionResult(
        image_path="car.jpg",
        image_width=640,
        image_height=480,
        num_damages=1,
        regions=[region],
        inference_time_ms=42.0,
        model_version="best.pt",
    )
    assert result.num_damages == 1
    assert result.regions[0].crop_path is None
    # Round-trips through JSON.
    assert DetectionResult.model_validate_json(result.model_dump_json()) == result


def test_confidence_out_of_bounds_rejected():
    with pytest.raises(Exception):
        DamageRegion(
            class_id=0, class_name="dent", confidence=1.5,
            bbox_xyxy=[0, 0, 1, 1], bbox_xywh_norm=[0, 0, 1, 1],
        )


def test_bbox_must_have_four_values():
    with pytest.raises(Exception):
        DamageRegion(
            class_id=0, class_name="dent", confidence=0.5,
            bbox_xyxy=[0, 0, 1], bbox_xywh_norm=[0, 0, 1, 1],
        )


# --- Config ----------------------------------------------------------------
def test_config_nc_matches_names():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    assert cfg["nc"] == len(cfg["names"])
    assert list(cfg["names"].keys()) == list(range(cfg["nc"]))


def test_config_classes_match_target_taxonomy():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    assert [cfg["names"][i] for i in range(cfg["nc"])] == dataset.TARGET_CLASS_NAMES


# --- Class merge -----------------------------------------------------------
def test_merge_map_covers_all_source_classes():
    assert set(dataset.CLASS_MERGE_MAP.keys()) == set(range(len(dataset.SOURCE_CLASS_NAMES)))


def test_merge_targets_in_range():
    n = len(dataset.TARGET_CLASS_NAMES)
    assert all(0 <= v < n for v in dataset.CLASS_MERGE_MAP.values())


def test_remap_label_text():
    # source id 12 (front-bumper-dent) -> dent(0); id 2 (Headlight) -> light_damage(2)
    raw = "12 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.05 0.05\n"
    out = dataset._remap_label_text(raw)
    lines = out.splitlines()
    assert lines[0].startswith("0 ")
    assert lines[1].startswith("2 ")


# --- Data splits (skip gracefully if data not prepared) --------------------
def _splits_ready() -> bool:
    return all((DATA_SPLITS / s / "images").exists() and
               any((DATA_SPLITS / s / "images").iterdir()) for s in ("train", "val", "test"))


@pytest.mark.skipif(not _splits_ready(), reason="data/splits not prepared")
def test_every_split_has_all_classes():
    nc = len(dataset.TARGET_CLASS_NAMES)
    for split in ("train", "val", "test"):
        counts: Counter[int] = Counter()
        for lf in (DATA_SPLITS / split / "labels").glob("*.txt"):
            counts.update(int(float(l.split()[0])) for l in lf.read_text().splitlines() if l.split())
        assert set(counts.keys()) == set(range(nc)), f"{split} missing classes: {counts}"


@pytest.mark.skipif(not _splits_ready(), reason="data/splits not prepared")
def test_images_and_labels_are_paired():
    for split in ("train", "val", "test"):
        orphans, valid = dataset._pair_images_labels(
            DATA_SPLITS / split / "images", DATA_SPLITS / split / "labels"
        )
        assert not orphans, f"{split} has {len(orphans)} images without labels"
        assert valid
