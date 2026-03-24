"""
SAM-based scene segmentation for the volume measurement pipeline.

Segments input images into 3 classes:
  - box:   white reference cube with ArUco markers (known real-world size)
  - leg:   measurement target — the biggest object that is NOT the box
  - floor: table/ground surface (discarded from reconstruction)

Strategy:
  1. Detect ArUco markers (DICT_5X5_50) → bounding box of the reference cube
  2. SAM prompted with that bbox → precise box mask
  3. SAM auto masks → largest mask that doesn't overlap the box → target object
  4. Everything else → floor / background (implicitly discarded)
"""

import os
import numpy as np
import cv2
import open3d as o3d

# ---------------------------------------------------------------------------
# SAM checkpoint management
# ---------------------------------------------------------------------------

SAM_CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
SAM_CHECKPOINT_FILES = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}


def download_sam_checkpoint(model_type="vit_b", save_dir="checkpoints"):
    """Download SAM checkpoint if not already present."""
    os.makedirs(save_dir, exist_ok=True)
    filename = SAM_CHECKPOINT_FILES[model_type]
    filepath = os.path.join(save_dir, filename)
    if os.path.exists(filepath):
        print(f"SAM checkpoint found: {filepath}")
        return filepath
    url = SAM_CHECKPOINT_URLS[model_type]
    print(f"Downloading SAM checkpoint ({model_type}) …")
    import urllib.request
    urllib.request.urlretrieve(url, filepath)
    print(f"Saved to {filepath}")
    return filepath


def load_sam_model(model_type="vit_b", checkpoint_path=None, device="cuda"):
    """Load and return a SAM model ready for inference."""
    from segment_anything import sam_model_registry
    if checkpoint_path is None:
        checkpoint_path = download_sam_checkpoint(model_type)
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.eval()
    print(f"SAM model ({model_type}) loaded on {device}")
    return sam


# ---------------------------------------------------------------------------
# ArUco-based reference-box detection
# ---------------------------------------------------------------------------

# Dictionaries to try (ordered by likelihood for this project)
_ARUCO_DICTS = [
    ("DICT_5X5_50",  cv2.aruco.DICT_5X5_50),
    ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
    ("DICT_4X4_50",  cv2.aruco.DICT_4X4_50),
    ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
    ("DICT_6X6_50",  cv2.aruco.DICT_6X6_50),
    ("DICT_ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
    ("DICT_APRILTAG_36h11", cv2.aruco.DICT_APRILTAG_36h11),
]


def detect_aruco_box(image_bgr):
    """
    Detect ArUco markers and return a bounding box enclosing the reference cube.

    Tries multiple ArUco dictionaries until markers are found.

    Args:
        image_bgr: (H, W, 3) BGR image (as loaded by cv2.imread).

    Returns:
        (x1, y1, x2, y2) bounding box with margin, or None if no markers found.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    for name, dict_id in _ARUCO_DICTS:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            all_pts = np.concatenate(corners, axis=1).squeeze()
            x_min, y_min = all_pts.min(axis=0)
            x_max, y_max = all_pts.max(axis=0)

            # Expand bbox to cover the full cube face (markers don't cover edges)
            pad = max(x_max - x_min, y_max - y_min) * 0.35
            x1 = max(0, int(x_min - pad))
            y1 = max(0, int(y_min - pad))
            x2 = min(W, int(x_max + pad))
            y2 = min(H, int(y_max + pad))

            print(f"  ArUco detected ({name}): {len(ids)} markers, "
                  f"IDs={ids.flatten().tolist()}, bbox=[{x1},{y1},{x2},{y2}]")
            return (x1, y1, x2, y2)

    print("  Warning: no ArUco markers detected in this frame")
    return None


# ---------------------------------------------------------------------------
# SAM helpers
# ---------------------------------------------------------------------------

def _resize_for_sam(image_rgb, max_side=1024):
    """Resize image so the longest side is at most max_side. Return (resized, scale)."""
    H, W = image_rgb.shape[:2]
    longest = max(H, W)
    if longest <= max_side:
        return image_rgb, 1.0
    scale = max_side / longest
    new_W, new_H = int(W * scale), int(H * scale)
    resized = cv2.resize(image_rgb, (new_W, new_H), interpolation=cv2.INTER_AREA)
    return resized, scale


def _sam_mask_from_bbox(image_rgb, sam_model, bbox):
    """Run SAM with a bounding-box prompt and return the best mask at original resolution."""
    from segment_anything import SamPredictor

    H_orig, W_orig = image_rgb.shape[:2]
    img_small, scale = _resize_for_sam(image_rgb)

    predictor = SamPredictor(sam_model)
    predictor.set_image(img_small)

    # Scale bbox to match resized image
    scaled_bbox = np.array(bbox, dtype=np.float32) * scale
    masks, scores, _ = predictor.predict(box=scaled_bbox, multimask_output=True)
    best = int(scores.argmax())

    # Resize mask back to original resolution
    mask_small = masks[best]
    if scale != 1.0:
        mask_orig = cv2.resize(mask_small.astype(np.uint8), (W_orig, H_orig),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        mask_orig = mask_small

    return mask_orig, float(scores[best])


def _sam_auto_masks(image_rgb, sam_model):
    """Generate automatic masks at reduced resolution, return masks scaled to original size."""
    from segment_anything import SamAutomaticMaskGenerator

    H_orig, W_orig = image_rgb.shape[:2]
    img_small, scale = _resize_for_sam(image_rgb)

    gen = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=500,
    )
    masks = gen.generate(img_small)

    # Upscale masks back to original resolution
    if scale != 1.0:
        for m in masks:
            seg_small = m["segmentation"]
            m["segmentation"] = cv2.resize(
                seg_small.astype(np.uint8), (W_orig, H_orig),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            # Recompute area at original resolution
            m["area"] = int(m["segmentation"].sum())

    masks.sort(key=lambda m: m["area"], reverse=True)
    return masks


# ---------------------------------------------------------------------------
# High-level segmentation entry point
# ---------------------------------------------------------------------------

def segment_frame(image_path, sam_model, world_points_frame=None, conf_frame=None):
    """
    Segment an image into box / leg (target) / floor.

    Pipeline:
      1. ArUco detection → bbox of reference box
      2. SAM prompted with bbox → box mask
      3. SAM auto masks → pick largest non-box mask → target object ("leg")
      4. Return classified masks dict

    Args:
        image_path:  path to the image on disk.
        sam_model:   loaded SAM model.
        world_points_frame: (H, W, 3) VGGT world points (unused in ArUco mode,
                     kept for API compatibility).
        conf_frame:  (H, W) confidence map (optional, unused in ArUco mode).

    Returns:
        dict  {"box": {"mask":…, "score":…}, "leg": {…}, "floor": None}
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]

    print(f"Segmenting {os.path.basename(image_path)} ({W}x{H}) …")

    # ── 1. Detect reference box via ArUco ──────────────────────────────
    aruco_bbox = detect_aruco_box(image_bgr)

    if aruco_bbox is None:
        print("  Falling back to pure SAM auto-segmentation (no ArUco)")
        return _fallback_segment(image_rgb, sam_model, world_points_frame, conf_frame)

    # ── 2. SAM prompted segmentation for the box ──────────────────────
    print("  Running SAM (box prompt) …")
    box_mask, box_score = _sam_mask_from_bbox(image_rgb, sam_model, aruco_bbox)
    print(f"  Box mask: {int(box_mask.sum()):,} pixels, score={box_score:.3f}")

    # ── 3. SAM auto masks → largest non-box object = target ───────────
    print("  Running SAM auto masks …")
    auto_masks = _sam_auto_masks(image_rgb, sam_model)
    print(f"  SAM produced {len(auto_masks)} auto masks")

    # Find the biggest auto mask that does NOT overlap heavily with the box
    target_mask = None
    target_score = 0.0
    for m in auto_masks:
        seg = m["segmentation"]
        overlap = float((seg & box_mask).sum()) / (box_mask.sum() + 1e-8)
        # Skip masks that overlap >30% with the box
        if overlap > 0.30:
            continue
        # Skip masks that are too small (likely noise)
        if m["area"] < 0.01 * H * W:
            continue
        target_mask = seg
        target_score = float(m.get("predicted_iou", m.get("stability_score", 0.9)))
        break  # already sorted by area, so first valid = largest

    if target_mask is None:
        print("  Warning: no target object found — using everything minus box as target")
        target_mask = ~box_mask

    print(f"  Target mask: {int(target_mask.sum()):,} pixels")

    result = {
        "box": {"mask": box_mask, "score": box_score, "label": "box"},
        "leg": {"mask": target_mask, "score": target_score, "label": "leg"},
        "floor": None,  # floor points are implicitly everything not box/leg
    }
    return result


def _fallback_segment(image_rgb, sam_model, world_points_frame=None, conf_frame=None):
    """Fallback when ArUco markers are not detected — uses old heuristic approach."""
    auto_masks = _sam_auto_masks(image_rgb, sam_model)
    print(f"  Fallback: SAM produced {len(auto_masks)} masks")

    if world_points_frame is not None:
        return _classify_masks_3d(auto_masks, world_points_frame, conf_frame)
    return _classify_masks_2d(auto_masks, image_rgb.shape)


# ---------------------------------------------------------------------------
# Legacy classifiers (used only when ArUco is unavailable)
# ---------------------------------------------------------------------------

def _classify_masks_3d(masks, world_points_frame, conf_frame=None, conf_threshold=0.0):
    """Classify SAM auto-masks using VGGT 3D geometry."""
    H_v, W_v = world_points_frame.shape[:2]
    candidates = []
    for i, mi in enumerate(masks):
        mo = mi["segmentation"]
        Ho, Wo = mo.shape
        mv = cv2.resize(mo.astype(np.uint8), (W_v, H_v),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
        valid = mv if conf_frame is None else mv & (conf_frame > conf_threshold)
        pts = world_points_frame[valid]
        if len(pts) < 10:
            continue
        extent = pts.max(axis=0) - pts.min(axis=0)
        dims = np.sort(extent)
        centroid = pts.mean(axis=0)
        try:
            _, s, _ = np.linalg.svd(pts - centroid, full_matrices=False)
            planarity = 1.0 - s[2] / (s[1] + 1e-8)
        except Exception:
            planarity = 0.0
        ys, _ = np.where(mo)
        bottom = float((ys > Ho * 0.5).sum()) / (len(ys) + 1e-8)
        candidates.append(dict(index=i, mask_orig=mo, dims=dims,
                               planarity=planarity, bottom_ratio=bottom,
                               area=mi["area"], centroid_3d=centroid))

    if len(candidates) < 2:
        return None

    result = {"floor": None, "box": None, "leg": None}
    used = set()
    ma = candidates[0]["area"]

    # Floor
    fc = sorted(candidates, key=lambda c: c["planarity"]*0.4 + c["bottom_ratio"]*0.3
                + (c["area"]/ma)*0.3, reverse=True)
    result["floor"] = {"mask": fc[0]["mask_orig"], "score": 1.0, "label": "floor"}
    used.add(fc[0]["index"])

    # Box — most cube-like
    rem = [c for c in candidates if c["index"] not in used]
    bc = sorted(rem, key=lambda c: (c["dims"]/(c["dims"].max()+1e-8)).min(), reverse=True)
    if bc:
        result["box"] = {"mask": bc[0]["mask_orig"], "score": 1.0, "label": "box"}
        used.add(bc[0]["index"])

    # Leg — largest remaining
    rem = [c for c in candidates if c["index"] not in used]
    if rem:
        rem.sort(key=lambda c: c["area"], reverse=True)
        result["leg"] = {"mask": rem[0]["mask_orig"], "score": 1.0, "label": "leg"}

    return result


def _classify_masks_2d(masks, image_shape):
    """Fallback classifier using only 2D heuristics."""
    H, W = image_shape[:2]
    result = {"floor": None, "box": None, "leg": None}
    cands = []
    for i, mi in enumerate(masks):
        m = mi["segmentation"]
        ys, _ = np.where(m)
        if len(ys) == 0:
            continue
        bw, bh = mi["bbox"][2], mi["bbox"][3]
        cands.append(dict(index=i, mask=m, area=mi["area"],
                          aspect=max(bw, bh)/(min(bw, bh)+1e-6),
                          bottom_ratio=float((ys>H*0.5).sum())/len(ys)))
    if not cands:
        return result
    used = set()
    fc = sorted(cands, key=lambda c: c["bottom_ratio"]*0.5+(c["area"]/cands[0]["area"])*0.5, reverse=True)
    result["floor"] = {"mask": fc[0]["mask"], "score": 1.0, "label": "floor"}
    used.add(fc[0]["index"])
    rem = [c for c in cands if c["index"] not in used]
    if rem:
        rem.sort(key=lambda c: c["aspect"])
        result["box"] = {"mask": rem[0]["mask"], "score": 1.0, "label": "box"}
        used.add(rem[0]["index"])
    rem = [c for c in cands if c["index"] not in used]
    if rem:
        rem.sort(key=lambda c: c["area"], reverse=True)
        result["leg"] = {"mask": rem[0]["mask"], "score": 1.0, "label": "leg"}
    return result


# ---------------------------------------------------------------------------
# Map 2D masks → 3D point clouds
# ---------------------------------------------------------------------------

def extract_points_by_mask(world_points, images, conf, mask_dict, frame_idx=0):
    """
    Split the VGGT point cloud into per-class subsets using 2D masks.

    For the segmented frame the mask is applied directly.
    For other frames points inside the 3D bounding box are included.

    Args:
        world_points: (S, H, W, 3)
        images:       (S, 3, H, W) or (S, H, W, 3)
        conf:         (S, H, W) confidence
        mask_dict:    classified masks from segment_frame()
        frame_idx:    frame whose masks are used

    Returns:
        dict  {class_name: {"points": (N,3), "colors": (N,3)}}
    """
    S, H_v, W_v = world_points.shape[:3]
    imgs = np.transpose(images, (0, 2, 3, 1)) if (images.ndim == 4 and images.shape[1] == 3) else images

    result = {}
    for cls in ("box", "leg", "floor"):
        entry = mask_dict.get(cls)
        if entry is None:
            result[cls] = {"points": np.zeros((0, 3)), "colors": np.zeros((0, 3))}
            continue

        mask_orig = entry["mask"]
        mask_v = cv2.resize(mask_orig.astype(np.uint8), (W_v, H_v),
                            interpolation=cv2.INTER_NEAREST).astype(bool)

        all_pts, all_cols = [], []

        # Primary frame — direct mask
        s = frame_idx
        valid = mask_v.copy()
        if conf is not None:
            pctl = np.percentile(conf[s][mask_v], 10) if mask_v.any() else 0
            valid = mask_v & (conf[s] > pctl)
        pts = world_points[s][valid]
        cols = imgs[s][valid]
        if len(pts) > 0:
            all_pts.append(pts)
            all_cols.append(cols)

        # Other frames — 3D bounding-box proximity
        if len(pts) > 0 and S > 1:
            margin = 0.05
            bb_min = pts.min(axis=0) - margin
            bb_max = pts.max(axis=0) + margin
            for s2 in range(S):
                if s2 == frame_idx:
                    continue
                fp = world_points[s2]
                fc = imgs[s2]
                fconf = conf[s2] if conf is not None else np.ones((H_v, W_v))
                inside = (
                    (fp[..., 0] >= bb_min[0]) & (fp[..., 0] <= bb_max[0])
                    & (fp[..., 1] >= bb_min[1]) & (fp[..., 1] <= bb_max[1])
                    & (fp[..., 2] >= bb_min[2]) & (fp[..., 2] <= bb_max[2])
                    & (fconf > 1e-5)
                )
                p2, c2 = fp[inside], fc[inside]
                if len(p2) > 0:
                    all_pts.append(p2)
                    all_cols.append(c2)

        if all_pts:
            result[cls] = {"points": np.concatenate(all_pts),
                           "colors": np.concatenate(all_cols)}
        else:
            result[cls] = {"points": np.zeros((0, 3)), "colors": np.zeros((0, 3))}
    return result


# ---------------------------------------------------------------------------
# PLY I/O
# ---------------------------------------------------------------------------

def save_class_ply(points, colors, output_path):
    """Save a coloured point cloud to PLY."""
    if len(points) == 0:
        print(f"  Warning: 0 points — skipping {output_path}")
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    c = colors.astype(np.float64)
    if c.max() > 1.0:
        c = c / 255.0
    pcd.colors = o3d.utility.Vector3dVector(np.clip(c, 0, 1))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"  Saved {len(points)} pts → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    "floor": np.array([100, 100, 255], dtype=np.float32),
    "box":   np.array([255, 200,   0], dtype=np.float32),
    "leg":   np.array([  0, 255, 100], dtype=np.float32),
}


def create_segmentation_overlay(image_rgb, mask_dict, alpha=0.45):
    """Return RGB image with translucent colour overlays per class."""
    overlay = image_rgb.astype(np.float32).copy()
    H, W = image_rgb.shape[:2]
    for cls, colour in CLASS_COLORS.items():
        entry = mask_dict.get(cls)
        if entry is None:
            continue
        mask = entry["mask"]
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay[mask] = overlay[mask] * (1 - alpha) + colour * alpha
    return overlay.astype(np.uint8)
