"""Stage 2 severity classifier — detailed evaluation on the held-out test split.

Reproduces the exact same stratified split used during training (seed=42),
then prints a confusion matrix + per-class precision / recall / F1.

Examples:
    python stage2_severity/evaluate.py \
        --weights runs/severity/efficientnet-b0-v6/weights/best.pt

    python stage2_severity/evaluate.py \
        --weights runs/severity/efficientnet-b0-v6/weights/best.pt \
        --labels stage2_severity/data/labels_v6_filtered.csv \
        --output runs/severity/efficientnet-b0-v6/eval.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from stage2_severity.train import (
    SEVERITY_CLASSES,
    CLASS_TO_IDX,
    SeverityDataset,
    build_model,
    get_transforms,
    load_and_split,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    return all_labels, all_preds


def confusion_matrix(labels, preds, n):
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(labels, preds):
        cm[t][p] += 1
    return cm


def per_class_metrics(cm, class_names):
    n = len(class_names)
    rows = []
    for i, name in enumerate(class_names):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n)) - tp
        fn = sum(cm[i][j] for j in range(n)) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support   = sum(cm[i])
        rows.append(dict(name=name, precision=precision, recall=recall,
                         f1=f1, support=support))
    return rows


def format_report(cm, metrics, class_names, overall_acc) -> str:
    n = len(class_names)
    lines = []

    # Confusion matrix
    col_w = max(len(c) for c in class_names) + 2
    lines.append("Confusion matrix (rows=actual, cols=predicted):")
    header = " " * (col_w + 2) + "  ".join(f"{c:>{col_w}}" for c in class_names)
    lines.append(header)
    for i, name in enumerate(class_names):
        row_str = f"{name:>{col_w}}  " + "  ".join(f"{cm[i][j]:>{col_w}}" for j in range(n))
        lines.append(row_str)

    # Per-class metrics
    lines.append("")
    lines.append(f"{'Class':12s}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'Support':>7}")
    lines.append("-" * 50)
    f1_scores = []
    for m in metrics:
        lines.append(f"{m['name']:12s}  {m['precision']:9.3f}  {m['recall']:6.3f}  "
                     f"{m['f1']:6.3f}  {m['support']:7d}")
        f1_scores.append(m['f1'])
    lines.append("-" * 50)
    macro_f1 = sum(f1_scores) / len(f1_scores)
    lines.append(f"{'macro avg':12s}  {'':9s}  {'':6s}  {macro_f1:6.3f}")
    lines.append(f"\nOverall accuracy : {overall_acc:.3f}  ({overall_acc*100:.1f}%)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Stage 2 severity classifier evaluation")
    p.add_argument("--weights", type=Path, required=True,
                   help="Path to best.pt weights")
    p.add_argument("--labels", type=Path,
                   default=Path("stage2_severity/data/labels_v6_filtered.csv"))
    p.add_argument("--output", type=Path, default=None,
                   help="Optional path to save the report as a text file")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers",   type=int, default=2)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--device",    type=str, default=None)
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device  : {device}")

    _, _, test_r = load_and_split(args.labels, seed=args.seed)
    print(f"Test set: {len(test_r)} crops")
    for cls in SEVERITY_CLASSES:
        n = sum(1 for r in test_r if r["severity"] == cls)
        print(f"  {cls:10s}: {n}")

    test_ds = SeverityDataset(test_r, get_transforms(train=False))
    loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers)

    model = build_model(num_classes=len(SEVERITY_CLASSES)).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f"Weights : {args.weights}\n")

    labels, preds = run_inference(model, loader, device)
    overall_acc   = sum(l == p for l, p in zip(labels, preds)) / len(labels)

    cm      = confusion_matrix(labels, preds, len(SEVERITY_CLASSES))
    metrics = per_class_metrics(cm, SEVERITY_CLASSES)
    report  = format_report(cm, metrics, SEVERITY_CLASSES, overall_acc)

    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"\nReport saved → {args.output}")


if __name__ == "__main__":
    main()
