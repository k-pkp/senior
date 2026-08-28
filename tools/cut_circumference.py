#!/usr/bin/env python3
"""Circumference of the limb at the cutting plane, on a finished run.

Stage 6 prints this already; this is the standalone form, for re-measuring an
existing run without re-running it, sweeping the slab thickness, or pointing at
a different cloud. The fit lives in `pipeline/core/crosssection.py` — this file
is only the command line around it.

Usage:
    python3 tools/cut_circumference.py                      # reads ./output
    python3 tools/cut_circumference.py --run work/est_test  # a stagerun dir
    python3 tools/cut_circumference.py --slab-mm 6 --json
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.core.crosssection import (      # noqa: E402
    SLAB_HALF_MM, fit_slice, load_cut_geometry,
)


def scale_from_volumes_csv(path):
    """cm per mesh unit, from the reference row Stage 6 already wrote.

    Read rather than recomputed, so this reports on exactly the scale in
    `volumes.csv` whichever Stage 6 derivation produced it. Two schemas are in
    circulation (see docs/repo_review.md); both are handled.
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    ref = [r for r in rows if str(r.get("is_ref", "")).lower() == "true"]
    if not ref:
        raise ValueError(f"no reference row in {path}")
    r = ref[0]
    if r.get("size_x_cm") and r.get("ext_x"):
        return float(r["size_x_cm"]) / float(r["ext_x"])
    if r.get("obb_a") and r.get("height_cm"):
        return float(r["height_cm"]) / float(r["obb_a"])
    # Both schemas always carry these two.
    return (float(r["real_vol_cm3"]) / float(r["volume"])) ** (1.0 / 3.0)


def _resolve(run_dir, *parts):
    """Handle both layouts: run.py's for_debug/NN_x and stagerun's NN_x."""
    a = os.path.join(run_dir, "for_debug", *parts)
    return a if os.path.exists(a) else os.path.join(run_dir, *parts)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="output", help="run directory (default: output)")
    ap.add_argument("--slab-mm", type=float, default=SLAB_HALF_MM,
                    help=f"half-thickness of the slice, real mm (default: {SLAB_HALF_MM:g})")
    ap.add_argument("--scale-cm-per-unit", type=float, default=None,
                    help="override the scale instead of reading volumes.csv")
    ap.add_argument("--cloud", default=None,
                    help="override the point cloud (default: leg_open.ply)")
    ap.add_argument("--json", action="store_true", help="also print JSON")
    args = ap.parse_args()

    clean_dir = _resolve(args.run, "03_clean")
    scale = args.scale_cm_per_unit
    if scale is None:
        vcsv = _resolve(args.run, "06_volume", "volumes.csv")
        if not os.path.exists(vcsv):
            sys.exit(f"no volumes.csv under {args.run} — pass --scale-cm-per-unit")
        scale = scale_from_volumes_csv(vcsv)

    pts, markers = load_cut_geometry(clean_dir)
    if pts is None:
        sys.exit(1)

    if args.cloud:
        import open3d as o3d
        pts = np.asarray(o3d.io.read_point_cloud(args.cloud).points, dtype=np.float64)
        if len(pts) == 0:
            sys.exit(f"empty cloud: {args.cloud}")

    print(f"run     {args.run}")
    print(f"scale   {scale:.4f} cm per mesh unit")
    print(f"slab    +/- {args.slab_mm:.1f} mm about the plane")

    results = []
    for i, mk in enumerate(markers):
        print(f"\nplane {i}  centroid {np.round(mk['centroid'], 4).tolist()}")
        try:
            r = fit_slice(pts, mk["centroid"], mk["normal"], scale, args.slab_mm)
        except Exception as exc:
            print(f"  FAILED — {type(exc).__name__}: {exc}")
            continue
        print(f"  slab points  {r['n_slab']:,}")
        print(f"  semi-axes    a = {r['a_cm']:.3f} cm   b = {r['b_cm']:.3f} cm   "
              f"(a/b = {r['a_cm'] / r['b_cm']:.3f})")
        print(f"  tilt in plane  {r['tilt_deg']:.1f} deg")
        print(f"  residual     RMS {r['resid_rms_mm']:.2f} mm")
        print(f"  coverage     {r['coverage'] * 100:.0f}% of the ring, "
              f"largest gap {r['max_gap_deg']:.1f} deg"
              f"{'   ** PARTIAL ARC — the fit is extrapolating **' if r['partial_arc'] else ''}")
        print(f"  cross-check  median-radius polygon "
              f"{r['polygon_circumference_cm']:.2f} cm "
              f"({(r['circumference_cm'] - r['polygon_circumference_cm']) / r['polygon_circumference_cm'] * 100:+.2f}% vs the ellipse)")
        print(f"  CIRCUMFERENCE (Ramanujan II)  {r['circumference_cm']:.2f} cm")
        r["plane"] = i
        results.append(r)

    if args.json:
        print("\n" + json.dumps(results, indent=2))
    if not results:
        sys.exit(1)


if __name__ == "__main__":
    main()
