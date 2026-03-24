#!/usr/bin/env python3
"""
CLI pipeline for leg volume measurement using VGGT + SAM.

Pipeline
--------
1. VGGT inference      → 3D point cloud from input images
2. SAM segmentation    → separate scene into box / leg / floor
3. Box processing      → clean, reconstruct, compute scale factor
4. Leg processing      → clean, reconstruct, compute real-world volume

Usage
-----
    python run_pipeline.py --images ./photos/ --cube-size 14.0
    python run_pipeline.py --images ./photos/ --cube-size 14.0 \\
        --sam-model vit_h --output-dir ./results/
"""

import os
import sys
import argparse
import glob
import time
import numpy as np
import torch
import cv2

# Make sure local packages are importable
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "vggt"))
sys.path.insert(0, os.path.join(_ROOT, "clean"))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from clean_ply import clean_point_cloud
from recons import reconstruct_mesh
from segment_sam import (
    load_sam_model,
    segment_frame,
    extract_points_by_mask,
    save_class_ply,
    create_segmentation_overlay,
)

import open3d as o3d


# ── CLI arguments ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Leg volume measurement: VGGT → SAM → scale → volume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--images", required=True,
                   help="Directory containing input images")
    p.add_argument("--cube-size", type=float, required=True,
                   help="Real reference-cube side length in cm")
    p.add_argument("--output-dir", default="pipeline_output",
                   help="Output directory (default: pipeline_output)")
    p.add_argument("--sam-model", default="vit_b",
                   choices=["vit_b", "vit_l", "vit_h"],
                   help="SAM backbone size (default: vit_b)")
    p.add_argument("--sam-checkpoint", default=None,
                   help="Path to SAM .pth checkpoint (auto-downloads if omitted)")
    p.add_argument("--conf-threshold", type=float, default=60.0,
                   help="Confidence percentile threshold (default: 60)")
    p.add_argument("--seg-frame", type=int, default=0,
                   help="Frame index used for SAM segmentation (default: 0)")
    p.add_argument("--device", default=None,
                   help="Device (default: cuda if available, else cpu)")
    p.add_argument("--prediction-mode", default="Pointmap Branch",
                   choices=["Pointmap Branch", "Depthmap and Camera Branch"],
                   help="Which VGGT output branch to use")
    return p.parse_args()


# ── Step 1: VGGT ──────────────────────────────────────────────────────────

def run_vggt_inference(image_dir, device):
    """Load VGGT, run inference on *image_dir*, return (predictions, image_paths)."""
    print("\n" + "=" * 60)
    print("STEP 1 / 4 : VGGT 3D Reconstruction")
    print("=" * 60)

    print("Loading VGGT-1B model …")
    model = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(url))
    model.eval().to(device)

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    image_names = sorted(
        f for f in glob.glob(os.path.join(image_dir, "*"))
        if f.lower().endswith(exts)
    )
    print(f"Found {len(image_names)} images in {image_dir}")
    if not image_names:
        raise ValueError(f"No images found in {image_dir}")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Tensor shape: {images.shape}")

    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    print("Running inference …")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)

    # Pose → extrinsics / intrinsics
    ext, intr = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = ext
    predictions["intrinsic"] = intr

    # Tensors → numpy
    for k in list(predictions.keys()):
        if isinstance(predictions[k], torch.Tensor):
            predictions[k] = predictions[k].cpu().numpy().squeeze(0)
    predictions["pose_enc_list"] = None

    # Depth → world points
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"],
    )

    torch.cuda.empty_cache()
    print("VGGT inference complete.\n")
    return predictions, image_names


# ── Step 2: SAM segmentation ──────────────────────────────────────────────

def run_segmentation(predictions, image_names, sam_model,
                     seg_frame=0, prediction_mode="Pointmap Branch"):
    """Segment the scene into box / leg / floor."""
    print("=" * 60)
    print("STEP 2 / 4 : SAM Segmentation")
    print("=" * 60)

    # Choose world-point source
    if "Pointmap" in prediction_mode:
        wp = predictions.get("world_points", predictions["world_points_from_depth"])
        conf = predictions.get("world_points_conf",
                               predictions.get("depth_conf"))
    else:
        wp = predictions["world_points_from_depth"]
        conf = predictions.get("depth_conf")

    seg_frame = min(seg_frame, len(image_names) - 1)
    image_path = image_names[seg_frame]
    wp_frame = wp[seg_frame]
    conf_frame = conf[seg_frame] if conf is not None else None

    mask_dict = segment_frame(image_path, sam_model, wp_frame, conf_frame)
    if mask_dict is None:
        raise RuntimeError(
            "SAM could not produce enough masks. "
            "Try a different --seg-frame or provide manual bounding boxes."
        )

    for cls in ("floor", "box", "leg"):
        if mask_dict.get(cls) is not None:
            n = int(mask_dict[cls]["mask"].sum())
            print(f"  {cls:6s}: {n:>8,} pixels  (score {mask_dict[cls]['score']:.3f})")
        else:
            print(f"  {cls:6s}: NOT DETECTED")

    # Map masks → 3D point clouds
    print("\nProjecting masks to 3D …")
    images = predictions["images"]
    class_points = extract_points_by_mask(wp, images, conf, mask_dict,
                                          frame_idx=seg_frame)
    for cls in ("box", "leg", "floor"):
        print(f"  {cls:6s}: {len(class_points[cls]['points']):>8,} 3D points")

    print()
    return mask_dict, class_points


# ── Step 3: Box → scale factor ────────────────────────────────────────────

def process_box_for_scale(class_points, output_dir, real_cube_size_cm):
    """Clean + reconstruct the box, then derive the mesh→real scale factor."""
    print("=" * 60)
    print("STEP 3 / 4 : Box → Scale Factor")
    print("=" * 60)

    box_dir = os.path.join(output_dir, "box")
    os.makedirs(box_dir, exist_ok=True)

    # Save raw box PLY
    box_ply = os.path.join(box_dir, "box_raw.ply")
    save_class_ply(
        class_points["box"]["points"],
        class_points["box"]["colors"],
        box_ply,
    )
    if class_points["box"]["points"].shape[0] < 100:
        raise RuntimeError("Too few box points for reliable reconstruction.")

    # Clean
    print("\nCleaning box point cloud …")
    clean_dir = os.path.join(box_dir, "clean")
    cleaned_box = clean_point_cloud(
        input_path=box_ply,
        output_folder=clean_dir,
        output_name="box_clean.ply",
        skip_plane_removal=True,
    )

    # Reconstruct
    print("\nReconstructing box mesh …")
    mesh_dir = os.path.join(box_dir, "mesh")
    box_mesh_ply, box_mesh_stl = reconstruct_mesh(
        input_path=cleaned_box,
        output_folder=mesh_dir,
        base_name="box",
    )

    # Measure box in mesh units
    box_pcd = o3d.io.read_point_cloud(cleaned_box)
    extent = box_pcd.get_axis_aligned_bounding_box().get_extent()
    estimated_side = float(np.median(extent))

    real_m = real_cube_size_cm / 100.0
    scale_factor = real_m / estimated_side

    print(f"\n  Box extent (mesh units): [{extent[0]:.5f}, {extent[1]:.5f}, {extent[2]:.5f}]")
    print(f"  Estimated side:          {estimated_side:.6f} mesh units")
    print(f"  Real cube size:          {real_cube_size_cm} cm  ({real_m} m)")
    print(f"  Scale factor:            {scale_factor:.6f}  "
          f"(1 mesh unit ≈ {real_m / scale_factor:.4f} m)\n")

    return scale_factor, box_mesh_ply, estimated_side


# ── Step 4: Leg → volume ──────────────────────────────────────────────────

def process_leg_for_volume(class_points, output_dir, scale_factor):
    """Clean + reconstruct the leg mesh, then compute real-world volume."""
    print("=" * 60)
    print("STEP 4 / 4 : Leg → Volume")
    print("=" * 60)

    leg_dir = os.path.join(output_dir, "leg")
    os.makedirs(leg_dir, exist_ok=True)

    leg_ply = os.path.join(leg_dir, "leg_raw.ply")
    save_class_ply(
        class_points["leg"]["points"],
        class_points["leg"]["colors"],
        leg_ply,
    )
    if class_points["leg"]["points"].shape[0] < 100:
        raise RuntimeError("Too few leg points for reliable reconstruction.")

    # Clean
    print("\nCleaning leg point cloud …")
    clean_dir = os.path.join(leg_dir, "clean")
    cleaned_leg = clean_point_cloud(
        input_path=leg_ply,
        output_folder=clean_dir,
        output_name="leg_clean.ply",
        skip_plane_removal=True,
    )

    # Reconstruct
    print("\nReconstructing leg mesh …")
    mesh_dir = os.path.join(leg_dir, "mesh")
    leg_mesh_ply, leg_mesh_stl = reconstruct_mesh(
        input_path=cleaned_leg,
        output_folder=mesh_dir,
        base_name="leg",
    )

    # Volume
    mesh = o3d.io.read_triangle_mesh(leg_mesh_ply)
    if not mesh.is_watertight():
        print("  Mesh not watertight → convex-hull fallback")
        pcd_tmp = mesh.sample_points_uniformly(number_of_points=50_000)
        mesh, _ = pcd_tmp.compute_convex_hull()

    raw_vol = mesh.get_volume()
    mesh.scale(scale_factor, center=mesh.get_center())
    real_m3 = mesh.get_volume()
    real_cm3 = real_m3 * 1e6

    print(f"\n  Raw volume:       {raw_vol:.6f} mesh-units³")
    print(f"  Scale factor:     {scale_factor:.6f}")
    print(f"  Real volume:      {real_m3:.6f} m³")
    print(f"  Real volume:      {real_cm3:.2f} cm³")
    print(f"  Watertight:       {mesh.is_watertight()}\n")

    return {
        "raw_volume": float(raw_vol),
        "scale_factor": float(scale_factor),
        "real_volume_m3": float(real_m3),
        "real_volume_cm3": float(real_cm3),
        "is_watertight": bool(mesh.is_watertight()),
        "mesh_ply": leg_mesh_ply,
        "mesh_stl": leg_mesh_stl,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()

    # 1) VGGT
    predictions, image_names = run_vggt_inference(args.images, device)
    pred_path = os.path.join(args.output_dir, "predictions.npz")
    np.savez(
        pred_path,
        **{k: v for k, v in predictions.items() if isinstance(v, np.ndarray)},
    )
    print(f"Predictions saved → {pred_path}")

    # 2) SAM
    print("\nLoading SAM model …")
    sam = load_sam_model(args.sam_model, args.sam_checkpoint, device)
    mask_dict, class_points = run_segmentation(
        predictions, image_names, sam,
        seg_frame=args.seg_frame,
        prediction_mode=args.prediction_mode,
    )

    # Save segmentation overlay
    seg_img = cv2.cvtColor(cv2.imread(image_names[min(args.seg_frame, len(image_names) - 1)]),
                           cv2.COLOR_BGR2RGB)
    overlay = create_segmentation_overlay(seg_img, mask_dict)
    overlay_path = os.path.join(args.output_dir, "segmentation_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Overlay saved → {overlay_path}")

    # Free SAM memory
    del sam
    torch.cuda.empty_cache()

    # 3) Box → scale
    scale_factor, box_mesh, box_side = process_box_for_scale(
        class_points, args.output_dir, args.cube_size,
    )

    # 4) Leg → volume
    vol = process_leg_for_volume(class_points, args.output_dir, scale_factor)

    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Time:             {elapsed:.1f} s")
    print(f"  Reference cube:   {args.cube_size} cm  "
          f"(measured {box_side:.4f} mesh units)")
    print(f"  Scale factor:     {scale_factor:.6f}")
    print(f"  Leg volume:       {vol['real_volume_cm3']:.2f} cm³  "
          f"({vol['real_volume_m3']:.6f} m³)")
    print(f"  Watertight:       {vol['is_watertight']}")
    print()
    print("  Output files:")
    print(f"    Predictions   : {pred_path}")
    print(f"    Seg overlay   : {overlay_path}")
    print(f"    Box mesh      : {box_mesh}")
    print(f"    Leg mesh (PLY): {vol['mesh_ply']}")
    print(f"    Leg mesh (STL): {vol['mesh_stl']}")

    # Persist summary
    summary = os.path.join(args.output_dir, "results.txt")
    with open(summary, "w") as f:
        f.write(f"cube_size_cm: {args.cube_size}\n")
        f.write(f"cube_mesh_units: {box_side:.6f}\n")
        f.write(f"scale_factor: {scale_factor:.6f}\n")
        f.write(f"leg_raw_volume: {vol['raw_volume']:.6f}\n")
        f.write(f"leg_volume_cm3: {vol['real_volume_cm3']:.2f}\n")
        f.write(f"leg_volume_m3: {vol['real_volume_m3']:.6f}\n")
        f.write(f"watertight: {vol['is_watertight']}\n")
    print(f"    Summary       : {summary}")


if __name__ == "__main__":
    main()
