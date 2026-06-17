"""Stage 2 severity classifier — EfficientNet-B0 fine-tuned on VLM pseudo-labels.

Classes: mild(0)  moderate(1)  severe(2)   [total_loss merged into severe]

Examples:
    python stage2_severity/train.py
    python stage2_severity/train.py --epochs 40 --batch-size 64 --wandb
    python stage2_severity/train.py --labels stage2_severity/data/labels_filtered.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_CLASSES = ["mild", "moderate", "severe"]
CLASS_TO_IDX = {c: i for i, c in enumerate(SEVERITY_CLASSES)}
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SeverityDataset(Dataset):
    def __init__(self, records: list[dict], transform):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        img = Image.open(r["crop_path"]).convert("RGB")
        return self.transform(img), CLASS_TO_IDX[r["severity"]]


# ---------------------------------------------------------------------------
# Data loading + stratified split
# ---------------------------------------------------------------------------

def load_and_split(
    labels_csv: Path,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load filtered CSV, merge total_loss→severe, return stratified splits."""
    rows: list[dict] = []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sev = r["severity"]
            if sev == "total_loss":
                sev = "severe"
            if sev not in CLASS_TO_IDX:
                continue
            if not Path(r["crop_path"]).exists():
                continue
            rows.append({"crop_path": r["crop_path"], "severity": sev,
                         "damage_type": r["damage_type"]})

    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {c: [] for c in SEVERITY_CLASSES}
    for r in rows:
        by_class[r["severity"]].append(r)

    train_r, val_r, test_r = [], [], []
    for cls_rows in by_class.values():
        rng.shuffle(cls_rows)
        n = len(cls_rows)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        test_r.extend(cls_rows[:n_test])
        val_r.extend(cls_rows[n_test:n_test + n_val])
        train_r.extend(cls_rows[n_test + n_val:])

    return train_r, val_r, test_r


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 3) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017). gamma=0 reduces to weighted cross-entropy."""
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


def compute_class_weights(records: list[dict], device: torch.device) -> torch.Tensor:
    counts = [sum(1 for r in records if r["severity"] == c) for c in SEVERITY_CLASSES]
    total = sum(counts)
    weights = [total / (len(counts) * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_sampler(records: list[dict]) -> WeightedRandomSampler:
    """WeightedRandomSampler so each batch sees ~equal class representation."""
    counts = {c: sum(1 for r in records if r["severity"] == c) for c in SEVERITY_CLASSES}
    sample_weights = [1.0 / counts[r["severity"]] for r in records]
    return WeightedRandomSampler(sample_weights, num_samples=len(records), replacement=True)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    model.train(train)
    total_loss = correct = total = 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train and optimizer:
                optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            if train and optimizer:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (out.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_per_class(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    correct_per = [0] * len(SEVERITY_CLASSES)
    total_per   = [0] * len(SEVERITY_CLASSES)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        for cls_idx in range(len(SEVERITY_CLASSES)):
            mask = labels == cls_idx
            total_per[cls_idx]   += mask.sum().item()
            correct_per[cls_idx] += (preds[mask] == cls_idx).sum().item()
    return {
        SEVERITY_CLASSES[i]: correct_per[i] / total_per[i] if total_per[i] else 0.0
        for i in range(len(SEVERITY_CLASSES))
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Stage 2 severity classifier training")
    p.add_argument("--labels", type=Path,
                   default=Path("stage2_severity/data/labels_filtered.csv"))
    p.add_argument("--output", type=Path,
                   default=Path("runs/severity/efficientnet-b0-v2"))
    p.add_argument("--epochs",        type=int,   default=40)
    p.add_argument("--freeze-epochs", type=int,   default=5,
                   help="Epochs to train head only before unfreezing backbone")
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=3e-4,
                   help="Head lr during freeze phase; backbone lr = lr/10 after unfreeze")
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--focal-gamma",  type=float, default=2.0,
                   help="Focal loss gamma (0 = standard weighted cross-entropy)")
    p.add_argument("--oversample",   action="store_true",
                   help="Use WeightedRandomSampler to balance batches by class")
    p.add_argument("--workers",      type=int,   default=2)
    p.add_argument("--device",       type=str,   default=None)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--wandb",        action="store_true")
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
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Splits
    train_r, val_r, test_r = load_and_split(args.labels, seed=args.seed)
    print(f"\nSplit: train={len(train_r)}  val={len(val_r)}  test={len(test_r)}")
    print(f"{'class':12s} {'train':>6} {'val':>6} {'test':>6}")
    for cls in SEVERITY_CLASSES:
        tc = sum(1 for r in train_r if r["severity"] == cls)
        vc = sum(1 for r in val_r   if r["severity"] == cls)
        ec = sum(1 for r in test_r  if r["severity"] == cls)
        print(f"  {cls:10s} {tc:6d} {vc:6d} {ec:6d}")

    train_ds = SeverityDataset(train_r, get_transforms(train=True))
    val_ds   = SeverityDataset(val_r,   get_transforms(train=False))
    test_ds  = SeverityDataset(test_r,  get_transforms(train=False))

    if args.oversample:
        sampler = make_sampler(train_r)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=(device.type != "cpu"))
        print("Sampler: WeightedRandomSampler (balanced batches)")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=(device.type != "cpu"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=(device.type != "cpu"))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=(device.type != "cpu"))

    # Model + loss
    model = build_model(num_classes=len(SEVERITY_CLASSES)).to(device)
    class_weights = compute_class_weights(train_r, device)
    print(f"\nClass weights: " +
          "  ".join(f"{c}={w:.2f}" for c, w in zip(SEVERITY_CLASSES, class_weights.tolist())))
    if args.focal_gamma > 0:
        criterion = FocalLoss(weight=class_weights, gamma=args.focal_gamma)
        print(f"Loss: FocalLoss(gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Loss: WeightedCrossEntropyLoss")

    # Phase 1: freeze backbone, train head only at higher lr.
    for p_ in model.features.parameters():
        p_.requires_grad = False
    optimizer = torch.optim.AdamW(
        filter(lambda p_: p_.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    print(f"\nPhase 1 ({args.freeze_epochs} epochs): backbone frozen, head lr={args.lr:.0e}")

    # W&B
    if args.wandb:
        import wandb
        from dotenv import load_dotenv
        import os
        load_dotenv()
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "vehicle-damage-detection"),
            name="efficientnet-b0-severity-v2",
            config=vars(args),
        )

    # Output dirs
    weights_dir = args.output / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (args.output / "args.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in vars(args).items())
    )

    # Training loop
    log_rows: list[dict] = []
    best_val_acc = 0.0
    patience_count = 0
    print()

    for epoch in range(1, args.epochs + 1):
        # Phase 2: unfreeze backbone with lower lr after freeze_epochs.
        if epoch == args.freeze_epochs + 1:
            for p_ in model.features.parameters():
                p_.requires_grad = True
            backbone_lr = args.lr / 10
            optimizer = torch.optim.AdamW([
                {"params": model.features.parameters(), "lr": backbone_lr},
                {"params": model.classifier.parameters(), "lr": args.lr / 10},
            ], weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.freeze_epochs, eta_min=1e-6
            )
            print(f"\nPhase 2: backbone unfrozen, lr={backbone_lr:.0e}")

        t0 = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      device, train=False)
        scheduler.step()
        elapsed = time.perf_counter() - t0
        lr_now = scheduler.get_last_lr()[0]

        row = dict(epoch=epoch, train_loss=round(train_loss, 4), train_acc=round(train_acc, 4),
                   val_loss=round(val_loss, 4), val_acc=round(val_acc, 4), lr=lr_now)
        log_rows.append(row)
        marker = ""
        print(f"[{epoch:02d}/{args.epochs}] "
              f"loss={train_loss:.4f}/{val_loss:.4f}  "
              f"acc={train_acc:.3f}/{val_acc:.3f}  "
              f"lr={lr_now:.2e}  {elapsed:.1f}s{marker}")

        if args.wandb:
            import wandb
            wandb.log(row)

        torch.save(model.state_dict(), weights_dir / "last.pt")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            torch.save(model.state_dict(), weights_dir / "best.pt")
            print(f"  -> best val_acc={best_val_acc:.3f}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping (patience={args.patience})")
                break

    # Save training log
    with open(args.output / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch","train_loss","train_acc","val_loss","val_acc","lr"])
        writer.writeheader()
        writer.writerows(log_rows)

    # Final per-class accuracy on test set
    print(f"\nLoading best weights for test evaluation ...")
    model.load_state_dict(torch.load(weights_dir / "best.pt", map_location=device))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, train=False)
    per_class = eval_per_class(model, test_loader, device)

    print(f"\n--- Test results ---")
    print(f"Overall accuracy : {test_acc:.3f}")
    for cls, acc in per_class.items():
        print(f"  {cls:10s}: {acc:.3f}")

    print(f"\nBest val_acc : {best_val_acc:.3f}")
    print(f"Weights      : {weights_dir / 'best.pt'}")
    print(f"Next         : python stage2_severity/evaluate.py "
          f"--weights {weights_dir / 'best.pt'}")

    if args.wandb:
        import wandb
        wandb.log({"test_acc": test_acc, **{f"test_{k}": v for k, v in per_class.items()}})
        wandb.finish()


if __name__ == "__main__":
    main()
