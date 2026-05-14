#!/usr/bin/env python3
"""Standalone test runner: Poisson reconstruction (Stage 4) + Watertight repair (Stage 5).

Usage:
    # Auto-detect .ply files in test/inputs/
    python test/run_test.py

    # Explicit paths
    python test/run_test.py box.ply obj.ply

    # Custom inputs folder
    python test/run_test.py --inputs-dir outputs/my_run/clean_objects

    # Reconstruction only (skip watertight)
    python test/run_test.py --no-watertight

Args:
    ply_paths        Optional explicit PLY paths (overrides auto-detection).
    --inputs-dir     Directory to scan for .ply files (default: test/inputs).
    --output-dir     Base directory for output meshes (default: test/test_mesh).
    --no-watertight  Skip watertight repair (reconstruction only).
    --seed           Random seed (default: 42).
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.stages.reconstruct import reconstruct_multiple_objects
from pipeline.stages.watertight import make_watertight_meshes


def main():
    parser = argparse.ArgumentParser(
        description="Poisson reconstruction + watertight repair test"
    )
    parser.add_argument(
        "ply_paths", nargs="*",
        help="One or more PLY point cloud files to reconstruct (optional)",
    )
    parser.add_argument(
        "--inputs-dir", default=os.path.join("test", "inputs"),
        help="Directory to scan for .ply files (default: test/inputs)",
    )
    parser.add_argument(
        "--output-dir", default=os.path.join("test", "test_mesh"),
        help="Base output directory (default: test/test_mesh)",
    )
    parser.add_argument(
        "--no-watertight", action="store_true",
        help="Skip watertight repair (reconstruction only)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    if args.ply_paths:
        ply_paths = list(args.ply_paths)
    else:
        ply_paths = sorted(glob.glob(os.path.join(args.inputs_dir, "*.ply")))
        if not ply_paths:
            print(f"ERROR: no .ply files found in {args.inputs_dir}")
            sys.exit(1)

    for p in ply_paths:
        if not os.path.exists(p):
            print(f"ERROR: file not found: {p}")
            sys.exit(1)

    recon_dir = os.path.join(args.output_dir, "recon")
    wt_dir = os.path.join(args.output_dir, "watertight")

    print()
    print("=" * 60)
    print("TEST: Reconstruction + Watertight")
    print("=" * 60)
    print(f"  Inputs:   {len(ply_paths)} file(s)")
    for p in ply_paths:
        print(f"    - {p}")
    print(f"  Output:   {args.output_dir}")
    print(f"  Seed:     {args.seed}")
    print(f"  Watertight: {'NO (skipped)' if args.no_watertight else 'YES'}")

    # ── Stage 4: Poisson Reconstruction ──
    scene_recon_ply, scene_recon_stl, recon_paths = reconstruct_multiple_objects(
        input_paths=ply_paths,
        output_folder=recon_dir,
        base_name="test_recon",
        seed=args.seed,
    )

    print()
    print(f"  Recon scene PLY:  {scene_recon_ply}")
    print(f"  Recon scene STL:  {scene_recon_stl}")
    print(f"  Recon per-object:")
    for p in recon_paths:
        size_kb = os.path.getsize(p) / 1024
        print(f"    - {p}  ({size_kb:.1f} KB)")

    if args.no_watertight:
        print("\n  Skipping watertight repair (--no-watertight).")
        return

    # ── Stage 5: Watertight Repair ──
    scene_wt_ply, scene_wt_stl, wt_paths = make_watertight_meshes(
        recon_paths=recon_paths,
        output_folder=wt_dir,
        base_name="test",
    )

    print()
    print(f"  Watertight scene PLY:  {scene_wt_ply}")
    print(f"  Watertight scene STL:  {scene_wt_stl}")
    print(f"  Watertight per-object:")
    for p in wt_paths:
        size_kb = os.path.getsize(p) / 1024
        print(f"    - {p}  ({size_kb:.1f} KB)")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Recon meshes:  {recon_dir}")
    print(f"  Watertight:    {wt_dir}")


if __name__ == "__main__":
    main()
