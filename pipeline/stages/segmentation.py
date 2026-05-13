"""Stage 3b — Marker-based leg surface segmentation.

Runs after cleaning (Stage 3) and before reconstruction (Stage 4).
Takes the obj.ply output from clean, detects colored markers, and cuts
the leg surface horizontally between marker positions.
"""

import os

import numpy as np
import open3d as o3d

from pipeline.core.segmentation import segment_point_cloud


def segment_leg_stage(object_paths, output_dir, height_axis="z", seed=42):
    """Apply marker-based leg segmentation to the obj.ply in object_paths.

    Args:
        object_paths: List of PLY paths from the clean stage (box.ply, obj.ply).
        output_dir: Base output directory for the pipeline.
        height_axis: Axis for height-based cut ("z" by default, vertical after leveling).
        seed: Random seed (unused, kept for API consistency).

    Returns:
        new_paths: List of PLY paths with obj.ply replaced by segmented_obj.ply.
                   If no obj.ply is found or segmentation produces too few points,
                   returns the original list unchanged.
    """
    print()
    print("=" * 60)
    print("STAGE 3b: Marker-based leg segmentation")
    print("=" * 60)

    clean_dir = os.path.join(output_dir, "clean_objects")
    seg_dir = os.path.join(output_dir, "segmented")
    os.makedirs(seg_dir, exist_ok=True)

    new_paths = list(object_paths)
    obj_ply = None
    obj_idx = None

    for i, p in enumerate(object_paths):
        if os.path.basename(p).lower().startswith("obj"):
            obj_ply = p
            obj_idx = i
            break

    if obj_ply is None:
        print("  No obj.ply found in clean outputs — skipping segmentation")
        return new_paths

    print(f"  Input: {obj_ply}")

    try:
        pcd = o3d.io.read_point_cloud(obj_ply)
        if len(pcd.points) == 0:
            print("  Empty point cloud — skipping")
            return new_paths

        segmented_pcd, summary = segment_point_cloud(pcd, height_axis=height_axis, verbose=True)

    except Exception as e:
        print(f"  ERROR during segmentation: {e}")
        return new_paths

    # --- Guard: don't save if we kept everything ---
    cut_info = summary.get("cut", {})
    if cut_info.get("case") == 0:
        reason = cut_info.get("reason", "no_markers")
        print(f"  No cut applied ({reason}) — keeping original obj.ply for reconstruction")
        return new_paths

    # --- Guard: too few points remaining ---
    n_kept = summary.get("n_kept", 0)
    if n_kept < 100:
        print(f"  Only {n_kept} points after cut (< 100) — keeping original obj.ply")
        return new_paths

    # --- Save segmented PLY ---
    seg_path = os.path.join(seg_dir, "segmented_obj.ply")
    o3d.io.write_point_cloud(seg_path, segmented_pcd, write_ascii=False)

    seg_pts = np.asarray(segmented_pcd.points)
    print(f"  Saved: {seg_path} ({n_kept:,} pts) "
          f"X[{seg_pts[:,0].min():.3f},{seg_pts[:,0].max():.3f}] "
          f"Y[{seg_pts[:,1].min():.3f},{seg_pts[:,1].max():.3f}] "
          f"Z[{seg_pts[:,2].min():.3f},{seg_pts[:,2].max():.3f}]")

    new_paths[obj_idx] = seg_path
    return new_paths
