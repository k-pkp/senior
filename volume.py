#!/usr/bin/env python3
"""Compute real-world volumes of watertight meshes using a reference object.

Usage:
    python volume.py mesh1.ply mesh2.ply [...]

The script:
  1. Computes the raw volume of each watertight mesh (in mesh units).
  2. Asks you which mesh is the reference (known real-world size).
  3. Asks for the reference's real linear size (e.g., 14 for a 14x14x14cm cube)
     OR real volume directly.
  4. Computes scale factor k = mesh_dim / real_dim (linear).
  5. Reports real-world volumes for all other meshes:
        real_volume = mesh_volume / k^3

Example:
    ArUco marker is a 14x14x14 cm cube.
    The mesh of the ArUco measures ~30 cm along its largest extent.
    Then k = 30/14 = 2.143 (mesh is 2.143x larger than reality).
    Other object mesh computes to 50 cm^3.
    Real volume = 50 / k^3 = 50 / 9.84 = 5.08 cm^3.
"""
import sys
import os
import argparse
import numpy as np
import trimesh


def compute_mesh_info(path):
    """Load mesh, return (volume, max_extent, watertight, mesh_obj).

    For non-watertight meshes, falls back to convex hull volume.
    """
    mesh = trimesh.load(path, force='mesh')
    bbox = mesh.bounds
    extents = bbox[1] - bbox[0]
    max_extent = float(np.max(extents))

    if mesh.is_watertight:
        volume = float(abs(mesh.volume))
        method = "watertight"
    else:
        try:
            volume = float(abs(mesh.convex_hull.volume))
            method = "convex_hull (mesh not watertight)"
        except Exception:
            volume = 0.0
            method = "FAILED"

    return {
        "path": path,
        "name": os.path.basename(path),
        "volume": volume,
        "max_extent": max_extent,
        "extents": extents,
        "watertight": mesh.is_watertight,
        "method": method,
    }


def main():
    p = argparse.ArgumentParser(
        description="Compute real-world volumes from watertight meshes using a reference object",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("meshes", nargs="+", help="Watertight mesh files (.ply, .stl, .obj, etc.)")
    p.add_argument("--ref-index", type=int, default=None,
                   help="Index of reference mesh (0-based). If omitted, asked interactively.")
    p.add_argument("--ref-size", type=float, default=None,
                   help="Real linear size of reference (e.g., 14 for a 14cm cube). "
                        "Compared against mesh's max extent.")
    p.add_argument("--ref-volume", type=float, default=None,
                   help="Real volume of reference (e.g., 2744 for a 14^3 cm cube). "
                        "Use either --ref-size OR --ref-volume.")
    p.add_argument("--unit", type=str, default="cm",
                   help="Unit label for output (default: cm)")
    args = p.parse_args()

    # Load all meshes
    print(f"\n{'='*70}")
    print("MESH VOLUMES (raw, in mesh units)")
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
        print(f"      Max extent   : {info['max_extent']:.6f} (units)")
        print(f"      Bbox extents : ({ext[0]:.4f}, {ext[1]:.4f}, {ext[2]:.4f})")
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
    print(f"  mesh volume      = {ref['volume']:.6f} units^3")
    print(f"  mesh max extent  = {ref['max_extent']:.6f} units")

    # Get real-world reference value
    real_size = args.ref_size
    real_volume = args.ref_volume

    if real_size is None and real_volume is None:
        print(f"\nProvide ONE of:")
        print(f"  (1) Real linear size of reference in {args.unit} (e.g., 14 for a 14x14x14 cube)")
        print(f"  (2) Real volume of reference in {args.unit}^3 (e.g., 2744)")
        choice = input("Enter '1' or '2': ").strip()
        if choice == "1":
            real_size = float(input(f"Real linear size ({args.unit}): ").strip())
        elif choice == "2":
            real_volume = float(input(f"Real volume ({args.unit}^3): ").strip())
        else:
            print("Invalid choice. Aborting.")
            sys.exit(1)

    # Compute scale factor k (mesh_dim / real_dim)
    if real_size is not None:
        k = ref["max_extent"] / real_size
        method_note = f"k = mesh_max_extent / real_size = {ref['max_extent']:.4f} / {real_size} = {k:.6f}"
    else:
        # k from volume: k^3 = mesh_vol / real_vol  ->  k = (mesh_vol/real_vol)^(1/3)
        k = (ref["volume"] / real_volume) ** (1.0 / 3.0)
        method_note = f"k = (mesh_vol / real_vol)^(1/3) = ({ref['volume']:.4f} / {real_volume})^(1/3) = {k:.6f}"

    k3 = k ** 3
    print(f"\n{'='*70}")
    print("SCALE FACTOR")
    print(f"{'='*70}")
    print(f"  {method_note}")
    print(f"  k     = {k:.6f}  (mesh is {k:.4f}x larger than reality, linearly)")
    print(f"  k^3   = {k3:.6f}  (volume scale factor)")

    # Compute real volumes for all meshes
    print(f"\n{'='*70}")
    print(f"REAL-WORLD VOLUMES (in {args.unit}^3)")
    print(f"{'='*70}")
    print(f"{'IDX':>4} {'NAME':<35} {'MESH VOL':>14} {'REAL VOL':>14}  {'DIM':>10}")
    print("-" * 80)
    for i, info in enumerate(infos):
        real_vol = info["volume"] / k3
        # Equivalent cube edge as a sanity check
        cube_edge = real_vol ** (1.0 / 3.0) if real_vol > 0 else 0
        marker = "  <- REF" if i == idx else ""
        print(f"{i:>4} {info['name']:<35} {info['volume']:>14.4f} "
              f"{real_vol:>14.4f}  {cube_edge:>8.3f}{args.unit}{marker}")

    print(f"\nReal mesh size for each object (extents in {args.unit}):")
    for i, info in enumerate(infos):
        ext = info["extents"] / k
        marker = " <- REF" if i == idx else ""
        print(f"  [{i}] {info['name']:<35}  "
              f"{ext[0]:>7.3f} x {ext[1]:>7.3f} x {ext[2]:>7.3f} {args.unit}{marker}")

    print()


if __name__ == "__main__":
    main()
