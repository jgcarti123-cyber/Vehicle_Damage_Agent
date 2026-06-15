"""Stage 1 ONNX export + CPU latency benchmark.

Production inference uses the ONNX model, not the .pt (see CLAUDE.md). Target:
<100ms inference on CPU.

Examples:
    python stage1_detection/export.py --weights runs/detect/yolo11s-baseline/weights/best.pt
    python stage1_detection/export.py --weights best.pt --benchmark
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def export_onnx(weights: Path, imgsz: int = 640, simplify: bool = True) -> Path:
    """Export YOLO weights to ONNX. Returns the .onnx path."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=simplify)
    print(f"Exported ONNX -> {onnx_path}")
    return Path(onnx_path)


def benchmark_cpu(onnx_path: Path, imgsz: int = 640, runs: int = 50) -> float:
    """Benchmark mean CPU inference latency (ms) over `runs` forward passes."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    for _ in range(5):  # warmup
        sess.run(None, {inp.name: dummy})

    start = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {inp.name: dummy})
    mean_ms = (time.perf_counter() - start) / runs * 1000.0

    target = 100.0
    flag = "OK " if mean_ms < target else "SLOW"
    print(f"[{flag}] CPU latency: {mean_ms:.1f}ms/image over {runs} runs (target <{target}ms)")
    return mean_ms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Stage 1 model to ONNX")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--no-simplify", action="store_true")
    p.add_argument("--benchmark", action="store_true", help="run CPU latency benchmark")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    onnx_path = export_onnx(a.weights, a.imgsz, simplify=not a.no_simplify)
    if a.benchmark:
        benchmark_cpu(onnx_path, a.imgsz)
