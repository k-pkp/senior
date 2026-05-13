"""Stage 7 — Compute real-world volumes using the ArUco reference cube.

Formula:
    mesh_bbox_vol_ref = X_ref * Y_ref * Z_ref     # product of reference extents
    k = real_size^3 / mesh_bbox_vol_ref           # volume scale factor
    real_X = mesh_X * k^(1/3)
    real_volume = mesh_volume * k

Example:
    Ref ArUco is 14x14x14 cm cube → real_volume = 14^3 = 2744 cm^3.
    Ref mesh bbox = 40 x 50 x 30 → bbox_vol = 60000.
    k = 2744 / 60000 = 0.04573.
"""
import os
import numpy as np
import trimesh

from pipeline.config import REFERENCE_OBJECT_INDEX, REFERENCE_REAL_SIZE_CM


def _measure_mesh(path, idx):
    """Load a mesh and return its volume/extent info, or None if missing."""
    if not os.path.exists(path):
        print(f"  [{idx}] {path}: missing, skipping")
        return None

    mesh = trimesh.load(path, force="mesh", process=False)
    # Oriented bbox: extents reflect true object dimensions even if the mesh
    # is tilted in world frame. AABB would over-report for any rotated mesh.
    obb_extents = np.asarray(mesh.bounding_box_oriented.extents)
    extents = np.sort(obb_extents)[::-1]   # longest, mid, short
    max_extent = float(extents[0])

    if mesh.is_watertight:
        volume = float(abs(mesh.volume))
        method = "watertight"
    else:
        try:
            volume = float(abs(mesh.convex_hull.volume))
            method = "convex_hull (not watertight)"
        except Exception:
            volume = 0.0
            method = "FAILED"

    bbox_vol = float(extents[0] * extents[1] * extents[2])
    return {
        "idx": idx, "name": os.path.basename(path), "volume": volume,
        "max_extent": max_extent, "extents": extents, "bbox_vol": bbox_vol,
        "method": method,
    }


def compute_volumes(object_mesh_paths):
    """Compute real-world volume of each object using ArUco reference for scale."""
    print()
    print("=" * 60)
    print(f"STAGE 7: Computing real-world volumes "
          f"(ref: object_{REFERENCE_OBJECT_INDEX} = {REFERENCE_REAL_SIZE_CM}cm ArUco)")
    print("=" * 60)

    if len(object_mesh_paths) <= REFERENCE_OBJECT_INDEX:
        print(f"  Not enough meshes (need object_{REFERENCE_OBJECT_INDEX} as reference). Skipping.")
        return

    infos = []
    for i, path in enumerate(object_mesh_paths):
        info = _measure_mesh(path, i)
        if info is None:
            continue
        infos.append(info)
        ext = info["extents"]
        print(f"  [{i}] {info['name']}: "
              f"mesh_vol={info['volume']:.6f}, bbox_vol={info['bbox_vol']:.6f} "
              f"(extents {ext[0]:.4f}x{ext[1]:.4f}x{ext[2]:.4f}) ({info['method']})")

    ref = next((x for x in infos if x["idx"] == REFERENCE_OBJECT_INDEX), None)
    if ref is None or ref["bbox_vol"] <= 0:
        print(f"  Reference object_{REFERENCE_OBJECT_INDEX} not available. Skipping.")
        return

    real_ref_vol = REFERENCE_REAL_SIZE_CM ** 3        # e.g. 14^3 = 2744 cm^3
    k = real_ref_vol / ref["bbox_vol"]
    cube_root_k = k ** (1.0 / 3.0)

    print("\n  Scale factor:")
    print(f"    ref bbox_vol = {ref['extents'][0]:.4f} * {ref['extents'][1]:.4f} * "
          f"{ref['extents'][2]:.4f} = {ref['bbox_vol']:.6f}")
    print(f"    real_ref_vol = {REFERENCE_REAL_SIZE_CM}^3 = {real_ref_vol:.2f} cm^3")
    print(f"    k            = real_ref_vol / mesh_bbox_vol = "
          f"{real_ref_vol:.2f} / {ref['bbox_vol']:.6f} = {k:.6f}")
    print(f"    k^(1/3)      = {cube_root_k:.6f}")

    # Linear dims = OBB extents * k^(1/3).
    # Volume = actual mesh volume (watertight / convex hull) * k, NOT bbox*k.
    print("\n  Real-world dimensions and volumes:")
    print(f"  {'IDX':>4} {'NAME':<20} {'L x W x H (cm)':<24} {'VOLUME (cm^3)':>14}")
    print("  " + "-" * 68)
    for info in infos:
        ext_cm = info["extents"] * cube_root_k
        real_vol = info["volume"] * k
        size_str = f"{ext_cm[0]:6.2f} x {ext_cm[1]:6.2f} x {ext_cm[2]:6.2f}"
        marker = "  <- REF" if info["idx"] == REFERENCE_OBJECT_INDEX else ""
        print(f"  {info['idx']:>4} {info['name']:<20} {size_str:<24} "
              f"{real_vol:>14.2f}{marker}")
