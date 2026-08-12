"""Object detection via Grounding DINO + SAM for leg/cube seed points.

Uses Grounding DINO Base (zero-shot) to find bounding boxes for "leg" and
"cube with black and white pattern", then SAM refines to masks. Center 5%
of each mask produces reliable 3D seed points for cluster labeling.

Runs on GPU BEFORE VGGT inference to avoid VRAM contention.
"""
import os

import numpy as np
from PIL import Image


_CACHE_DIR = os.path.expanduser("~/.cache/opencode/models")


def _get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _load_grounding_dino(device):
    """Load Grounding DINO Base model."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=_CACHE_DIR)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id, cache_dir=_CACHE_DIR).to(device)
    model.eval()
    return processor, model


def _load_sam(device):
    """Load SAM ViT-B from segment-anything."""
    import torch
    from segment_anything import sam_model_registry, SamPredictor

    checkpoint_path = os.path.join(_CACHE_DIR, "sam_vit_b_01ec64.pth")
    if not os.path.exists(checkpoint_path):
        _download_sam_checkpoint(checkpoint_path)

    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()
    return SamPredictor(sam)


def _download_sam_checkpoint(path):
    """Download SAM ViT-B checkpoint if missing."""
    import urllib.request

    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  Downloading SAM checkpoint to {path} ...")
    urllib.request.urlretrieve(url, path)
    print("  Done.")


def detect_objects(images_518, text_prompts=None, frames=None):
    """Detect objects in VGGT-preprocessed images, return labeled 3D seed coordinates.

    Args:
        images_518:  (S, 3, 518, 518) float32 numpy array in [0, 1] range.
        text_prompts: list of string prompts for Grounding DINO.
        frames: which frame indices to run detection on (default: first, middle, last).

    Returns:
        seeds: list of dicts with keys {"label": str, "xyz": (3,) float32}.
               label is one of "leg" or "cube".
    """
    if text_prompts is None:
        text_prompts = ["leg", "cube with black and white pattern"]

    S = images_518.shape[0]
    if frames is None:
        frames = [0, S // 2, S - 1]
    frames = [f for f in frames if 0 <= f < S]
    frames = list(dict.fromkeys(frames))

    import torch

    device = _get_device()
    print(f"  Detection on device: {device}")

    gd_processor, gd_model = _load_grounding_dino(device)
    sam_predictor = _load_sam(device)

    seeds = []

    for frame_idx in frames:
        img_np = images_518[frame_idx].transpose(1, 2, 0)
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        # Grounding DINO detection
        inputs = gd_processor(images=img_pil, text=". ".join(text_prompts),
                              return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = gd_model(**inputs)

        results = gd_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.25,
            text_threshold=0.25,
            target_sizes=[img_pil.size[::-1]],
        )[0]

        boxes = results.get("boxes", [])
        labels = results.get("labels", [])

        if len(boxes) == 0:
            print(f"  Frame {frame_idx}: no detections")
            continue

        for box, label in zip(boxes, labels):
            box_np = box.cpu().numpy()
            x1, y1, x2, y2 = box_np

            # SAM refinement
            sam_predictor.set_image(np.array(img_pil))
            input_box = np.array([[x1, y1, x2, y2]])
            masks, scores, _ = sam_predictor.predict(
                box=input_box,
                multimask_output=False,
            )
            mask = masks[0]

            # Center 5% crop
            ys, xs = np.where(mask)
            if len(ys) < 10:
                continue
            cy, cx = int(ys.mean()), int(xs.mean())
            h, w = mask.shape
            ch, cw = max(1, int(h * 0.05)), max(1, int(w * 0.05))
            y0, y1_crop = max(0, cy - ch // 2), min(h, cy + ch // 2)
            x0, x1_crop = max(0, cx - cw // 2), min(w, cx + cw // 2)
            center_mask = mask[y0:y1_crop, x0:x1_crop]

            cy_adj, cx_adj = np.where(center_mask)
            if len(cy_adj) == 0:
                continue

            mapped_label = "leg" if "leg" in label.lower() else "cube"

            for dy, dx in zip(cy_adj, cx_adj):
                seeds.append({
                    "label": mapped_label,
                    "frame": frame_idx,
                    "u": int(x0 + dx),
                    "v": int(y0 + dy),
                })

        print(f"  Frame {frame_idx}: {len(boxes)} detections → "
              f"{len([s for s in seeds if s['frame'] == frame_idx])} seeds")

    del gd_model, sam_predictor
    torch.cuda.empty_cache()

    print(f"  Total seeds: {len(seeds)} "
          f"(leg={sum(1 for s in seeds if s['label']=='leg')}, "
          f"cube={sum(1 for s in seeds if s['label']=='cube')})")
    return seeds


def seeds_to_xyz_labels(seeds, world_points):
    """Convert pixel-coordinate seeds to 3D world-space labeled points.

    Args:
        seeds: list from detect_objects() with frame, u, v keys.
        world_points: (S, H, W, 3) float32 numpy array.

    Returns:
        xyz: (N, 3) float32 array of 3D seed points.
        labels: (N,) int32 array. 0=unknown, 1=leg, 2=cube.
    """
    xyz_list = []
    label_list = []

    label_map = {"leg": 1, "cube": 2}

    for seed in seeds:
        frame = seed["frame"]
        u = seed["u"]
        v = seed["v"]
        lbl_str = seed["label"]

        if 0 <= frame < world_points.shape[0] and 0 <= v < world_points.shape[1] and 0 <= u < world_points.shape[2]:
            pt = world_points[frame, v, u]
            xyz_list.append(pt)
            label_list.append(label_map.get(lbl_str, 0))

    if len(xyz_list) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    return np.array(xyz_list, dtype=np.float32), np.array(label_list, dtype=np.int32)
