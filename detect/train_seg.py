#!/usr/bin/env python3
"""
Train segmentation model (TransformerUNet) with active learning.

Pipeline:
  1. Convert YOLO polygon labels → segmentation masks
  2. Start with a small labeled pool (initial_ratio)
  3. Train model on labeled pool
  4. Score unlabeled pool by prediction uncertainty (entropy)
  5. Add most uncertain samples to labeled pool
  6. Repeat until all data is used or max rounds reached

Usage:
    python detect/train_seg.py --data_dir ./data
    python detect/train_seg.py --data_dir ./data --epochs 50 --al_rounds 5
    python detect/train_seg.py --data_dir ./data --img_size 256 --batch_size 8
"""

import os
import sys
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# Ensure detect/ is on path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from model import build_model


# ---------------------------------------------------------------------------
#  YOLO polygon → segmentation mask conversion
# ---------------------------------------------------------------------------
def parse_yolo_polygon(label_path, img_h, img_w):
    """Parse YOLO polygon label file → list of (class_id, polygon_points).

    YOLO polygon format: class_id x1 y1 x2 y2 ... xn yn
    where coordinates are normalized [0, 1].
    """
    polygons = []
    if not os.path.exists(label_path):
        return polygons

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class_id + at least 3 points
                continue
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            # Convert normalized coords to pixel coords
            points = []
            for i in range(0, len(coords), 2):
                x = coords[i] * img_w
                y = coords[i + 1] * img_h
                points.append([x, y])
            polygons.append((class_id, np.array(points, dtype=np.int32)))
    return polygons


def polygons_to_mask(polygons, img_h, img_w, num_classes=3):
    """Convert list of (class_id, polygon) → segmentation mask.

    Returns mask of shape (H, W) where:
      0 = background
      1 = ArUco  (YOLO class 0)
      2 = Vase   (YOLO class 1)
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for class_id, points in polygons:
        # Map YOLO class to mask class: YOLO 0 → mask 1, YOLO 1 → mask 2
        mask_class = class_id + 1
        cv2.fillPoly(mask, [points], mask_class)
    return mask


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------
class SegDataset(Dataset):
    """Dataset that loads images + YOLO polygon labels → (image, mask) pairs."""

    def __init__(self, image_dir, label_dir, img_size=256, augment=False):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment

        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        self.image_files = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in img_exts
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            # Return blank if image can't be read
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            mask = np.zeros((self.img_size, self.img_size), dtype=np.int64)
            return self._to_tensor(img), torch.from_numpy(mask)

        h, w = img.shape[:2]

        # Parse polygons at original resolution, then create mask
        polygons = parse_yolo_polygon(str(label_path), h, w)
        mask = polygons_to_mask(polygons, h, w)

        # Augmentation (before resize for better quality)
        if self.augment:
            img, mask = self._augment(img, mask)

        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        return self._to_tensor(img), torch.from_numpy(mask.astype(np.int64))

    def _to_tensor(self, img):
        """BGR uint8 (H, W, 3) → RGB float (3, H, W) normalized to [0, 1]."""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img.transpose(2, 0, 1))

    def _augment(self, img, mask):
        """Simple augmentations: flip, rotate, color jitter."""
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        # Random vertical flip
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        # Random 90° rotation
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()
        # Color jitter (brightness/contrast)
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.randint(-20, 20)     # brightness
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        return img, mask


# ---------------------------------------------------------------------------
#  Active learning: uncertainty scoring
# ---------------------------------------------------------------------------
def compute_uncertainty(model, dataset, indices, device, img_size=256, batch_size=8):
    """Compute per-sample uncertainty as mean pixel entropy of predictions.

    Higher entropy = more uncertain = more informative for active learning.
    """
    model.eval()
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    uncertainties = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            probs = model.predict_proba(imgs)  # (B, C, H, W)
            # Per-pixel entropy: -sum(p * log(p))
            log_probs = torch.log(probs + 1e-8)
            entropy = -(probs * log_probs).sum(dim=1)  # (B, H, W)
            # Mean entropy per sample
            mean_entropy = entropy.mean(dim=(1, 2))  # (B,)
            uncertainties.extend(mean_entropy.cpu().tolist())

    return uncertainties


def active_learning_query(model, dataset, labeled_idx, unlabeled_idx,
                          device, query_size, img_size=256, batch_size=8):
    """Select the most uncertain samples from unlabeled pool.

    Returns list of indices to add to labeled pool.
    """
    if len(unlabeled_idx) == 0:
        return []

    # Score all unlabeled samples
    uncertainties = compute_uncertainty(
        model, dataset, unlabeled_idx, device, img_size, batch_size
    )

    # Select top-k most uncertain
    idx_uncertainty = list(zip(unlabeled_idx, uncertainties))
    idx_uncertainty.sort(key=lambda x: x[1], reverse=True)

    n_query = min(query_size, len(idx_uncertainty))
    selected = [idx for idx, _ in idx_uncertainty[:n_query]]

    if len(uncertainties) > 0:
        avg_unc = sum(uncertainties) / len(uncertainties)
        top_unc = idx_uncertainty[0][1] if idx_uncertainty else 0
        print(f"    Uncertainty — avg: {avg_unc:.4f}, max: {top_unc:.4f}")

    return selected


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------
def compute_class_weights(dataset, indices, num_classes=3, device="cpu"):
    """Compute inverse-frequency class weights for handling class imbalance."""
    counts = torch.zeros(num_classes)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=16, shuffle=False, num_workers=0)
    for _, masks in loader:
        for c in range(num_classes):
            counts[c] += (masks == c).sum().float()

    # Inverse frequency weighting, clamped
    total = counts.sum()
    weights = total / (num_classes * counts + 1e-6)
    weights = weights / weights.min()  # normalize so min weight = 1
    weights = weights.clamp(max=10.0)
    return weights.to(device)


def dice_loss(pred, target, num_classes=3, smooth=1.0):
    """Soft Dice loss for all classes."""
    pred_soft = F.softmax(pred, dim=1)  # (B, C, H, W)
    loss = 0.0
    for c in range(num_classes):
        pred_c = pred_soft[:, c]
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        loss += 1.0 - (2.0 * intersection + smooth) / (union + smooth)
    return loss / num_classes


def train_one_epoch(model, loader, optimizer, criterion, device, num_classes=3):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)

        # Combined loss: CE + Dice
        ce_loss = criterion(logits, masks)
        d_loss = dice_loss(logits, masks, num_classes)
        loss = ce_loss + d_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, num_classes=3):
    """Evaluate model. Returns mean IoU and per-class IoU."""
    model.eval()
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)
        preds = logits.argmax(dim=1)

        for c in range(num_classes):
            pred_c = (preds == c)
            target_c = (masks == c)
            intersection[c] += (pred_c & target_c).sum().float()
            union[c] += (pred_c | target_c).sum().float()

    iou = intersection / (union + 1e-6)
    mean_iou = iou[1:].mean()  # exclude background
    return mean_iou.item(), iou.cpu().tolist()


def train_round(model, dataset, labeled_idx, val_loader, device, args, round_num):
    """Train model on current labeled pool for args.epochs epochs."""
    print(f"\n  --- Training Round {round_num} ({len(labeled_idx)} samples) ---")

    # DataLoader for labeled pool
    labeled_subset = Subset(dataset, labeled_idx)
    train_loader = DataLoader(
        labeled_subset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(labeled_idx) > args.batch_size,
        num_workers=0,
    )

    # Class weights for imbalanced data
    weights = compute_class_weights(dataset, labeled_idx, num_classes=3, device=device)
    print(f"  Class weights: bg={weights[0]:.2f}, aruco={weights[1]:.2f}, vase={weights[2]:.2f}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_miou = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            miou, class_iou = evaluate(model, val_loader, device)
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{args.epochs} — loss: {loss:.4f}, "
                  f"mIoU: {miou:.4f} [aruco: {class_iou[1]:.3f}, vase: {class_iou[2]:.3f}], "
                  f"lr: {lr:.6f}")

            if miou > best_miou:
                best_miou = miou
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_miou


# ---------------------------------------------------------------------------
#  Main: active learning loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train segmentation model with active learning")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Root data directory with train/valid/test splits")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Input image size (default: 256)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Epochs per active learning round (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--al_rounds", type=int, default=5,
                        help="Number of active learning rounds (default: 5)")
    parser.add_argument("--initial_ratio", type=float, default=0.3,
                        help="Initial labeled pool ratio (default: 0.3)")
    parser.add_argument("--query_ratio", type=float, default=0.15,
                        help="Fraction of unlabeled pool to query per round (default: 0.15)")
    parser.add_argument("--base_ch", type=int, default=32,
                        help="Base channels for UNet (default: 32)")
    parser.add_argument("--t_heads", type=int, default=4,
                        help="Transformer attention heads (default: 4)")
    parser.add_argument("--t_layers", type=int, default=2,
                        help="Transformer encoder layers (default: 2)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Checkpoint directory (default: detect/checkpoints)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Paths
    data_dir = Path(args.data_dir)
    train_img_dir = data_dir / "train" / "images"
    train_lbl_dir = data_dir / "train" / "labels"
    val_img_dir = data_dir / "valid" / "images"
    val_lbl_dir = data_dir / "valid" / "labels"

    if not train_img_dir.exists():
        print(f"ERROR: Train images not found at {train_img_dir}")
        sys.exit(1)

    # Checkpoint dir
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(_SCRIPT_DIR, "checkpoints")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Datasets
    train_dataset = SegDataset(train_img_dir, train_lbl_dir, args.img_size, augment=True)
    val_dataset = SegDataset(val_img_dir, val_lbl_dir, args.img_size, augment=False)

    print(f"Train images: {len(train_dataset)}")
    print(f"Val images:   {len(val_dataset)}")

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Build model
    num_classes = 3  # bg + ArUco + Vase
    model = build_model(
        num_classes=num_classes,
        img_size=args.img_size,
        base_ch=args.base_ch,
        t_heads=args.t_heads,
        t_layers=args.t_layers,
        t_ff=args.base_ch * 16 * 2,  # 2x bottleneck channels
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # ── Active Learning Setup ──
    n_total = len(train_dataset)
    all_indices = list(range(n_total))
    random.shuffle(all_indices)

    # Initial labeled pool
    n_initial = max(2, int(n_total * args.initial_ratio))
    labeled_idx = all_indices[:n_initial]
    unlabeled_idx = all_indices[n_initial:]

    query_size = max(1, int(n_total * args.query_ratio))

    print(f"\n{'='*60}")
    print(f"Active Learning Configuration")
    print(f"{'='*60}")
    print(f"  Total samples     : {n_total}")
    print(f"  Initial pool      : {n_initial} ({args.initial_ratio*100:.0f}%)")
    print(f"  Query size/round  : {query_size}")
    print(f"  AL rounds         : {args.al_rounds}")
    print(f"  Epochs per round  : {args.epochs}")

    best_overall_miou = 0.0
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_seg_model.pt")

    for round_num in range(1, args.al_rounds + 1):
        print(f"\n{'='*60}")
        print(f"ACTIVE LEARNING ROUND {round_num}/{args.al_rounds}")
        print(f"  Labeled: {len(labeled_idx)}, Unlabeled: {len(unlabeled_idx)}")
        print(f"{'='*60}")

        # Re-initialize model each round for clean training
        # (prevents overfitting to earlier smaller pools)
        model = build_model(
            num_classes=num_classes,
            img_size=args.img_size,
            base_ch=args.base_ch,
            t_heads=args.t_heads,
            t_layers=args.t_layers,
            t_ff=args.base_ch * 16 * 2,
        ).to(device)

        # Train on current labeled pool
        miou = train_round(model, train_dataset, labeled_idx, val_loader, device, args, round_num)

        # Save if best
        if miou > best_overall_miou:
            best_overall_miou = miou
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "img_size": args.img_size,
                "base_ch": args.base_ch,
                "t_heads": args.t_heads,
                "t_layers": args.t_layers,
                "miou": miou,
                "round": round_num,
                "labeled_count": len(labeled_idx),
            }
            torch.save(checkpoint, best_checkpoint_path)
            print(f"  ★ New best mIoU: {miou:.4f} — saved to {best_checkpoint_path}")

        # Also save round checkpoint
        round_path = os.path.join(args.checkpoint_dir, f"round_{round_num}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "img_size": args.img_size,
            "base_ch": args.base_ch,
            "t_heads": args.t_heads,
            "t_layers": args.t_layers,
            "miou": miou,
            "round": round_num,
            "labeled_count": len(labeled_idx),
        }, round_path)

        # Query most uncertain samples from unlabeled pool
        if round_num < args.al_rounds and len(unlabeled_idx) > 0:
            print(f"\n  Querying {query_size} most uncertain samples...")
            new_indices = active_learning_query(
                model, train_dataset, labeled_idx, unlabeled_idx,
                device, query_size, args.img_size, args.batch_size
            )
            labeled_idx.extend(new_indices)
            unlabeled_idx = [i for i in unlabeled_idx if i not in set(new_indices)]
            print(f"  Pool updated: labeled={len(labeled_idx)}, unlabeled={len(unlabeled_idx)}")

    # ── Final Summary ──
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"  Best mIoU       : {best_overall_miou:.4f}")
    print(f"  Best checkpoint  : {best_checkpoint_path}")
    print(f"  Final pool size  : {len(labeled_idx)}/{n_total}")

    # Final evaluation on test set if available
    test_img_dir = data_dir / "test" / "images"
    test_lbl_dir = data_dir / "test" / "labels"
    if test_img_dir.exists():
        test_dataset = SegDataset(test_img_dir, test_lbl_dir, args.img_size, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Load best model
        ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

        test_miou, test_iou = evaluate(model, test_loader, device)
        print(f"\n  Test mIoU: {test_miou:.4f}")
        print(f"    ArUco: {test_iou[1]:.4f}")
        print(f"    Vase:  {test_iou[2]:.4f}")


if __name__ == "__main__":
    main()
