#!/usr/bin/env python3
"""Compute real-world volumes of meshes using a reference cube.

Usage:
    python volume.py mesh1.ply mesh2.ply [...]

Formula:
    mesh_bbox_vol_ref = X_ref * Y_ref * Z_ref    (product of reference bbox extents)
    real_ref_vol      = real_size^3              (real volume of the reference cube)
    k                 = real_ref_vol / mesh_bbox_vol_ref    (volume scale factor)

    For any object:
        real_X = mesh_X * k^(1/3)
        real_Y = mesh_Y * k^(1/3)
        real_Z = mesh_Z * k^(1/3)
        real_volume = mesh_volume * k

Example:
    ArUco is a 14x14x14 cm cube → real_ref_vol = 14^3 = 2744 cm^3.
    ArUco mesh bbox = 40 x 50 x 30 → mesh_bbox_vol_ref = 60000.
    k = 2744 / 60000 = 0.04573.
    For an object with mesh bbox (X,Y,Z) = (40,50,30):
        real_X = 40 * 0.04573^(1/3) = 40 * 0.3576 = 14.305 cm
        real_Y = 50 * 0.3576 = 17.881 cm
        real_Z = 30 * 0.3576 = 10.729 cm
        real_volume = 40*50*30 * 0.04573 = 2744 cm^3
"""
import sys
import os
import argparse
import trimesh


def compute_mesh_info(path):
    """Load mesh, return dict with volume, extents, bbox_vol, watertight info."""
    mesh = trimesh.load(path, force="mesh")
    extents = mesh.bounds[1] - mesh.bounds[0]
    bbox_vol = float(extents[0] * extents[1] * extents[2])

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

    return {
        "path": path,
        "name": os.path.basename(path),
        "volume": volume,
        "extents": extents,
        "bbox_vol": bbox_vol,
        "watertight": mesh.is_watertight,
        "method": method,
    }


def main():
    p = argparse.ArgumentParser(
        description="Compute real-world volumes from meshes using a reference cube",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("meshes", nargs="+", help="Mesh files (.ply, .stl, .obj, ...)")
    p.add_argument("--ref-index", type=int, default=None,
                   help="Index of reference mesh (0-based). Asked interactively if omitted.")
    p.add_argument("--ref-size", type=float, default=14,
                   help="Real edge length of reference cube (e.g., 14 for a 14x14x14cm cube).")
    p.add_argument("--unit", type=str, default="cm", help="Unit label for output (default: cm)")
    args = p.parse_args()

    # Load meshes
    print(f"\n{'='*70}")
    print("MESH INFO (in mesh units)")
    print(f"{'='*70}")
    infos = []
    for i, path in enumerate(args.meshes):
        if not os.path.exists(path):
            print(f"  [{i}] {path}: NOT FOUND, skipping")
            continue
        info = compute_mesh_info(path)
        infos.append(info)
        ext = info["extents"]
        print(f"\n  [{i}] {info['name']}")
        print(f"      Volume       : {info['volume']:.6f} (units^3)")
        print(f"      Bbox extents : {ext[0]:.4f} x {ext[1]:.4f} x {ext[2]:.4f}")
        print(f"      Bbox volume  : {info['bbox_vol']:.6f} (units^3)")
        print(f"      Watertight   : {info['watertight']} ({info['method']})")

    if not infos:
        print("\nNo valid meshes found.")
        sys.exit(1)

    # Pick reference
    if args.ref_index is None:
        print(f"\n{'='*70}")
        try:
            idx = int(input(f"Reference mesh index [0-{len(infos)-1}]: ").strip())
        except (ValueError, EOFError):
            print("Invalid index. Aborting.")
            sys.exit(1)
    else:
        idx = args.ref_index

    if idx < 0 or idx >= len(infos):
        print(f"Invalid index {idx}. Must be 0-{len(infos)-1}.")
        sys.exit(1)

    ref = infos[idx]
    print(f"\nReference: [{idx}] {ref['name']}")

    # Real reference edge length
    real_size = args.ref_size
    if real_size is None:
        try:
            real_size = float(input(f"Real edge length of reference cube ({args.unit}): ").strip())
        except (ValueError, EOFError):
            print("Invalid value. Aborting.")
            sys.exit(1)

    # k = real_size^3 / mesh_bbox_vol_ref
    real_ref_vol = real_size ** 3
    k = real_ref_vol / ref["bbox_vol"]
    cube_root_k = k ** (1.0 / 3.0)

    print(f"\n{'='*70}")
    print("SCALE FACTOR")
    print(f"{'='*70}")
    print(f"  ref bbox_vol = {ref['extents'][0]:.4f} * {ref['extents'][1]:.4f} * "
          f"{ref['extents'][2]:.4f} = {ref['bbox_vol']:.6f}")
    print(f"  real_ref_vol = {real_size}^3 = {real_ref_vol:.2f} {args.unit}^3")
    print(f"  k            = real_ref_vol / mesh_bbox_vol = {k:.6f}")
    print(f"  k^(1/3)      = {cube_root_k:.6f}")

    # Report real-world dimensions and volumes.
    # Volume = real_X * real_Y * real_Z = mesh_bbox_vol * k
    print(f"\n{'='*70}")
    print(f"REAL-WORLD DIMENSIONS AND VOLUMES ({args.unit}, {args.unit}^3)")
    print(f"{'='*70}")
    print(f"{'IDX':>4} {'NAME':<28} {'REAL SIZE':<26} {'VOLUME':>14}")
    print("-" * 80)
    for i, info in enumerate(infos):
        real_ext = info["extents"] * cube_root_k
        real_vol = float(real_ext[0] * real_ext[1] * real_ext[2])
        size_str = f"{real_ext[0]:6.2f} x {real_ext[1]:6.2f} x {real_ext[2]:6.2f}"
        marker = "  <- REF" if i == idx else ""
        print(f"{i:>4} {info['name']:<28} {size_str:<26} {real_vol:>14.2f}{marker}")

    print()


if __name__ == "__main__":
    main()
