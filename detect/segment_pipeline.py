#!/usr/bin/env python3
"""
Segmentation inference pipeline — called by run.py for --segment mode.

Provides:
    load_seg_model(checkpoint_path, device) → (model, num_classes)
    segment_image(model, img, img_size, device) → full_mask (H, W) with class IDs
    mask_for_target(full_mask, target) → binary_mask (H, W) with 0/1
"""

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from model import build_model


def load_seg_model(checkpoint_path, device="cpu"):
    """Load trained TransformerUNet from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device: Device string ("cuda", "cpu", "mps").

    Returns:
        (model, num_classes) tuple.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Segmentation checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    num_classes = ckpt.get("num_classes", 3)
    img_size = ckpt.get("img_size", 256)
    base_ch = ckpt.get("base_ch", 32)
    t_heads = ckpt.get("t_heads", 4)
    t_layers = ckpt.get("t_layers", 2)

    model = build_model(
        num_classes=num_classes,
        img_size=img_size,
        base_ch=base_ch,
        t_heads=t_heads,
        t_layers=t_layers,
        t_ff=base_ch * 16 * 2,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"  Loaded segmentation model: {num_classes} classes, "
          f"base_ch={base_ch}, img_size={img_size}, "
          f"mIoU={ckpt.get('miou', 'N/A')}")

    return model, num_classes


@torch.no_grad()
def segment_image(model, img, img_size=256, device="cpu"):
    """Run segmentation on a single BGR image.

    Args:
        model: Loaded TransformerUNet model.
        img: BGR numpy image (H, W, 3), uint8.
        img_size: Model input size (default 256).
        device: Device string.

    Returns:
        full_mask: (H, W) numpy array with class IDs:
            0 = background, 1 = ArUco, 2 = Vase
    """
    orig_h, orig_w = img.shape[:2]

    # Preprocess: resize, RGB, normalize, to tensor
    resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # Inference
    logits = model(tensor)  # (1, C, img_size, img_size)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Resize back to original resolution
    if (orig_h, orig_w) != (img_size, img_size):
        pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return pred


def mask_for_target(full_mask, target="vase"):
    """Convert multi-class mask to binary mask for the specified target.

    Args:
        full_mask: (H, W) with class IDs (0=bg, 1=ArUco, 2=Vase).
        target: "vase", "aruco", or "both".

    Returns:
        binary_mask: (H, W) with 0/1 values.
    """
    if target == "vase":
        return (full_mask == 2).astype(np.uint8)
    elif target == "aruco":
        return (full_mask == 1).astype(np.uint8)
    elif target == "both":
        return (full_mask > 0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown target: {target}. Use 'vase', 'aruco', or 'both'.")
