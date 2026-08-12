"""Stage 2 — confidence-filter the VGGT pointmap and export it as PLY.

One output: points.ply. Ghost reduction happens in Stage 3, applied per
identified cluster, so no pre-filtered cloud is produced here.
"""
import os

import numpy as np
import trimesh

from pipeline.core.filters import adaptive_confidence_filter


def _extract_base_cloud(predictions, args):
    """Confidence-filtered points + colours. No SOR (Stage 3 applies it)."""
    if args.prediction_mode == "pointmap":
        world_points = predictions["world_points"]
        conf_raw = predictions["world_points_conf"]
    else:
        world_points = predictions["world_points_from_depth"]
        conf_raw = predictions["depth_conf"]

    imgs_np = predictions["images"]
    if imgs_np.ndim == 4 and imgs_np.shape[1] == 3:
        colors_4d = imgs_np.transpose(0, 2, 3, 1)
    elif imgs_np.ndim == 4 and imgs_np.shape[3] == 3:
        colors_4d = imgs_np
    else:
        raise ValueError(f"Unexpected images shape: {imgs_np.shape}")

    S, H, W_shape = world_points.shape[:3]

    conf_flat = conf_raw.reshape(-1)
    conf_thresh = adaptive_confidence_filter(conf_flat, args.conf_thres)
    conf_mask = (conf_raw >= conf_thresh) & (conf_raw > 1e-5)

    # Multi-view consistency — the only filter that removes ghost sheets, which
    # are parallel duplicates the geometric filters downstream cannot see.
    from pipeline.config import MULTIVIEW_MIN_VIEWS, MULTIVIEW_REL_THRESHOLD
    min_views = getattr(args, "multiview_min_views", MULTIVIEW_MIN_VIEWS)
    if min_views and min_views > 0 and predictions.get("depth") is not None:
        from pipeline.multiview import multiview_mask
        mv = multiview_mask(world_points, predictions["depth"],
                            predictions["extrinsic"], predictions["intrinsic"],
                            min_views=min_views,
                            rel_threshold=MULTIVIEW_REL_THRESHOLD)
        before = int(conf_mask.sum())
        conf_mask &= mv
        print(f"    combined with confidence: {before:,} → {int(conf_mask.sum()):,}")

    if getattr(args, "mask_black_bg", False):
        brightness = colors_4d.reshape(-1, 3).astype(np.float32).mean(axis=1)
        bg_mask = (brightness > 15.0).reshape(S, H, W_shape)
        conf_mask &= bg_mask
    if getattr(args, "mask_white_bg", False):
        brightness = colors_4d.reshape(-1, 3).astype(np.float32).mean(axis=1)
        bg_mask = (brightness < 240.0).reshape(S, H, W_shape)
        conf_mask &= bg_mask

    points_flat = world_points[conf_mask].reshape(-1, 3).astype(np.float32)
    colors_flat = (colors_4d[conf_mask].reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)

    print(f"  Base cloud: {conf_mask.sum():,} points "
          f"(kept {100 * conf_mask.mean():.1f}% after confidence)")

    return points_flat, colors_flat, world_points, conf_raw, conf_mask, imgs_np


def export_ply(predictions, output_dir, args):
    """Confidence + multi-view filter, then spatial outlier removal."""
    from pipeline.core.filters import remove_spatial_outliers

    print()
    print("=" * 60)
    print("STAGE 2: Exporting PLY point cloud")
    print("=" * 60)
    print(f"  Mode: {'pointmap regression' if args.prediction_mode == 'pointmap' else 'depth-based unprojection'}")

    points_out, colors_out, _wp, conf_raw, conf_mask, _imgs = \
        _extract_base_cloud(predictions, args)
    conf_out = conf_raw[conf_mask].reshape(-1).astype(np.float32)

    points_out, colors_out, _conf_out = remove_spatial_outliers(
        points_out, colors_out, conf_out)

    print(f"  Final point count: {points_out.shape[0]:,}")

    ply_path = os.path.join(output_dir, "points.ply")
    pc = trimesh.PointCloud(points_out, colors=colors_out)
    pc.export(ply_path)
    print(f"  Exported: {ply_path}")

    return ply_path
