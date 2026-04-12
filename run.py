#!/usr/bin/env python3
"""
VGGT Run Script — automated terminal inference → PLY → clean → reconstruct → evaluate.

Usage:
    python run.py                                          # uses ./baam/ as input
    python run.py --image_folder ./baam/
    python run.py --image_folder ./baam/ --output_dir output/
    python run.py --image_folder ./baam/ --skip_mesh        # PLY only, skip clean+reconstruct
    python run.py --image_folder ./baam/ --evaluate          # auto-screenshot with viewer.py

Supports CUDA, MPS (Apple Silicon), and CPU backends automatically.
"""

import os
import sys
import glob
import time
import shutil
import argparse

import numpy as np
import torch
import trimesh

# Ensure project root is on path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "clean"))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.device import get_device, is_mps, autocast_on, aggressive_cleanup


def parse_args():
    p = argparse.ArgumentParser(
        description="VGGT — run full pipeline: inference → PLY → clean → mesh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                                  # default: ./baam/ input
  python run.py --image_folder ./baam/ --evaluate
  python run.py --image_folder examples/kitchen/images/ --conf_thres 30
  python run.py --skip_mesh                      # PLY only
""",
    )
    p.add_argument("--image_folder", type=str, default="./baam/",
                   help="Path to folder containing input images (default: ./baam/)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory for output files (default: ./output/)")
    p.add_argument("--conf_thres", type=float, default=45.0,
                   help="Confidence threshold (percentile): filter bottom N%% of points (default: 45)")
    p.add_argument("--prediction_mode", type=str, default="pointmap",
                   choices=["pointmap", "depth"],
                   help="'pointmap' uses direct 3D point regression; 'depth' unprojects depth maps")
    p.add_argument("--mask_black_bg", action="store_true",
                   help="Mask out near-black background pixels")
    p.add_argument("--mask_white_bg", action="store_true",
                   help="Mask out near-white background pixels")
    p.add_argument("--skip_mesh", action="store_true",
                   help="Skip clean + reconstruct stages (PLY export only)")
    p.add_argument("--num_objects", type=int, default=2,
                   help="Number of objects to extract during cleaning (default: 2)")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Max frames to process (auto-set to 7 on MPS to avoid OOM)")
    p.add_argument("--evaluate", action="store_true",
                   help="Auto-capture screenshots of outputs with viewer.py")
    p.add_argument("--no-watertight", action="store_true",
                   help="Skip watertight repair (export only Poisson reconstruction)")
    return p.parse_args()


# ---------------------------------------------------------------------------
#  Stage 1: Model inference → predictions dict (numpy)
# ---------------------------------------------------------------------------
def _select_frames(image_names, max_frames):
    """Uniformly subsample frames if there are more than max_frames, keeping first and last."""
    n = len(image_names)
    if max_frames is None or n <= max_frames:
        return image_names
    # Always include first and last; pick the rest evenly
    indices = np.linspace(0, n - 1, max_frames, dtype=int)
    indices = sorted(set(indices))
    selected = [image_names[i] for i in indices]
    return selected


def run_inference(image_folder, device, max_frames=None):
    """Load model, run inference, return predictions dict (numpy) and timings."""
    t0 = time.time()
    print("=" * 60)
    print("STAGE 1: Loading model and running inference")
    print("=" * 60)

    # Flush any leftover MPS cache from previous runs
    aggressive_cleanup(device)

    # Auto-limit frames on MPS to avoid OOM
    # 9 frames @ 518×518 → global attention over 12k tokens → ~4.9GB just for attn scores
    # 7 frames → ~3.0GB → fits in 30GB MPS with model + activations
    if max_frames is None and is_mps(device):
        max_frames = 6
        print(f"  MPS detected: auto-limiting to {max_frames} frames (override with --max_frames)")

    # Load model
    print("  Loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, map_location="cpu"))
    model.eval()
    model = model.to(device)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Load images
    image_names = sorted(glob.glob(os.path.join(image_folder, "*")))
    image_names = [p for p in image_names
                   if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))]
    if not image_names:
        print(f"ERROR: No images found in {image_folder}")
        sys.exit(1)

    original_count = len(image_names)
    image_names = _select_frames(image_names, max_frames)
    if len(image_names) < original_count:
        print(f"  Found {original_count} images → selected {len(image_names)} (uniformly spaced)")
    else:
        print(f"  Found {len(image_names)} images")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"  Preprocessed shape: {images.shape}")

    # Inference
    t1 = time.time()
    print("  Running inference...")
    with torch.no_grad():
        with autocast_on(device):
            predictions = model(images)

    # Pose encoding → extrinsic/intrinsic
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert all tensors to numpy
    for key in list(predictions.keys()):
        v = predictions[key]
        if isinstance(v, torch.Tensor):
            predictions[key] = v.cpu().float().numpy().squeeze(0)
        elif isinstance(v, list):
            predictions[key] = None
    predictions["pose_enc_list"] = None

    # Depth-based world points
    depth_map = predictions["depth"]
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"]
    )

    inference_time = time.time() - t1
    print(f"  Inference done in {inference_time:.1f}s")

    del model
    aggressive_cleanup(device)

    return predictions, inference_time


# ---------------------------------------------------------------------------
#  Stage 2: Export PLY point cloud
# ---------------------------------------------------------------------------
def _adaptive_confidence_filter(conf_flat, target_percentile):
    """Smart confidence filtering that adapts to the data distribution.

    Problem: some datasets (e.g. vase) have 85% of points at conf=1.0,
    so a 50th percentile filter keeps tons of garbage low-confidence points.
    Solution: use the HIGHER of percentile-based and absolute thresholds.
    """
    percentile_val = np.percentile(conf_flat, target_percentile)

    # Analyze distribution to pick a smart absolute minimum
    # If most points are at the minimum (conf ≈ 1.0), raise the bar
    conf_min = conf_flat.min()
    frac_at_min = (conf_flat <= conf_min + 0.01).mean()

    if frac_at_min > 0.5:
        # Most points are at minimum confidence → use a higher absolute threshold
        # Keep only truly confident points (top of the distribution)
        abs_threshold = np.percentile(conf_flat[conf_flat > conf_min + 0.01], 25) if (conf_flat > conf_min + 0.01).any() else conf_min + 0.1
        threshold_val = max(percentile_val, abs_threshold)
        print(f"  Adaptive filter: {frac_at_min*100:.0f}% of points at min conf={conf_min:.2f}")
        print(f"    Percentile threshold: {percentile_val:.4f}")
        print(f"    Absolute threshold:   {abs_threshold:.4f}")
        print(f"    Using: {threshold_val:.4f}")
    else:
        threshold_val = percentile_val
        print(f"  Confidence threshold: {threshold_val:.4f} ({target_percentile:.0f}th percentile)")

    return threshold_val


def _remove_spatial_outliers(points, colors, conf, k=20, std_ratio=2.5):
    """Remove spatial outliers using Open3D statistical outlier removal."""
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        _, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std_ratio)
        inlier_idx = np.asarray(inlier_idx)
        removed = len(points) - len(inlier_idx)
        if removed > 0:
            print(f"  Spatial outlier removal: {len(points):,} → {len(inlier_idx):,} (removed {removed:,})")
        return points[inlier_idx], colors[inlier_idx], conf[inlier_idx]
    except Exception:
        return points, colors, conf


def export_ply(predictions, output_dir, args):
    """Filter and export point cloud as PLY with adaptive confidence and outlier removal."""
    print()
    print("=" * 60)
    print("STAGE 2: Exporting PLY point cloud")
    print("=" * 60)

    if args.prediction_mode == "pointmap":
        world_points = predictions["world_points"]         # (S, H, W, 3)
        conf = predictions["world_points_conf"]             # (S, H, W)
        print("  Mode: pointmap regression")
    else:
        world_points = predictions["world_points_from_depth"]
        conf = predictions["depth_conf"]
        print("  Mode: depth-based unprojection")

    # Images: (S, 3, H, W) → (S, H, W, 3)
    imgs_np = predictions["images"]
    colors = imgs_np.transpose(0, 2, 3, 1)

    points_flat = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    # Adaptive confidence filter
    threshold_val = _adaptive_confidence_filter(conf_flat, args.conf_thres)
    mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

    # Color masks
    if args.mask_black_bg:
        brightness = colors_flat.astype(np.float32).mean(axis=1)
        mask &= brightness > 15.0
    if args.mask_white_bg:
        brightness = colors_flat.astype(np.float32).mean(axis=1)
        mask &= brightness < 240.0

    points_out = points_flat[mask]
    colors_out = colors_flat[mask]
    conf_out = conf_flat[mask]

    print(f"  After filtering: {points_flat.shape[0]:,} → {points_out.shape[0]:,}")

    # Remove spatial outliers (points flying far from the cluster)
    points_out, colors_out, conf_out = _remove_spatial_outliers(points_out, colors_out, conf_out)

    print(f"  Final point count: {points_out.shape[0]:,}")

    ply_path = os.path.join(output_dir, "points.ply")
    pc = trimesh.PointCloud(points_out, colors=colors_out)
    pc.export(ply_path)
    print(f"  Exported: {ply_path}")

    return ply_path


# ---------------------------------------------------------------------------
#  Stage 3: Clean point cloud and extract objects
# ---------------------------------------------------------------------------
def clean_and_extract(ply_path, output_dir, num_objects=2):
    """Clean point cloud, extract top-k objects."""
    print()
    print("=" * 60)
    print("STAGE 3: Cleaning point cloud and extracting objects")
    print("=" * 60)

    try:
        from clean_ply import clean_and_extract_objects
    except ImportError:
        print("  WARNING: clean_ply module not found. Skipping cleaning stage.")
        print("  Make sure ./clean/clean_ply.py exists.")
        return None

    clean_output_dir = os.path.join(output_dir, "clean_objects")
    os.makedirs(clean_output_dir, exist_ok=True)

    try:
        object_paths = clean_and_extract_objects(
            input_path=ply_path,
            output_folder=clean_output_dir,
            k=num_objects,
            visualize=False,
        )
        print(f"  Extracted {len(object_paths)} objects:")
        for p in object_paths:
            print(f"    → {p}")
        return object_paths
    except Exception as e:
        print(f"  ERROR during cleaning: {e}")
        return None


# ---------------------------------------------------------------------------
#  Stage 4: Reconstruct mesh from cleaned objects
# ---------------------------------------------------------------------------
def reconstruct_mesh_stage(object_paths, output_dir):
    """STAGE 4: Reconstruct each object point cloud into a (non-watertight) mesh.

    Returns (scene_recon_path, recon_mesh_paths).
    """
    print()
    print("=" * 60)
    print("STAGE 4: Reconstructing mesh (Poisson)")
    print("=" * 60)

    try:
        from recons import reconstruct_multiple_objects
    except ImportError:
        print("  WARNING: recons module not found. Skipping reconstruction.")
        return None, []

    mesh_output_dir = os.path.join(output_dir, "mesh")
    os.makedirs(mesh_output_dir, exist_ok=True)

    try:
        scene_recon, _, recon_paths = reconstruct_multiple_objects(
            input_paths=object_paths,
            output_folder=mesh_output_dir,
            base_name="scene_recon",
        )
        print(f"  Scene recon mesh: {scene_recon}")
        for p in recon_paths:
            print(f"  Object recon mesh: {p}")
        return scene_recon, recon_paths
    except Exception as e:
        print(f"  ERROR during reconstruction: {e}")
        return None, []
    
# ---------------------------------------------------------------------------
#  Stage 5: Making meshes watertight (PyMeshFix)
# ---------------------------------------------------------------------------


def watertight_stage(recon_paths, output_dir):
    """STAGE 5: Fill each reconstructed mesh to be watertight (PyMeshFix + color transfer).

    Returns (scene_watertight_path, watertight_mesh_paths).
    """
    print()
    print("=" * 60)
    print("STAGE 5: Making meshes watertight (PyMeshFix)")
    print("=" * 60)

    try:
        from recons import make_watertight_meshes
    except ImportError:
        print("  WARNING: recons module not found. Skipping watertight step.")
        return None, []

    mesh_output_dir = os.path.join(output_dir, "mesh")

    try:
        scene_wt, _, wt_paths = make_watertight_meshes(
            recon_paths=recon_paths,
            output_folder=mesh_output_dir,
            base_name="scene",
        )
        print(f"  Scene watertight mesh: {scene_wt}")
        for p in wt_paths:
            print(f"  Object watertight mesh: {p}")
        return scene_wt, wt_paths
    except Exception as e:
        print(f"  ERROR during watertight repair: {e}")
        return None, []


# ---------------------------------------------------------------------------
#  Stage 6: Evaluate — auto-screenshot using viewer.py
# ---------------------------------------------------------------------------
def _capture_multi_view(viewer_script, file_path, output_dir, label, extra_args=None):
    """Capture multi-perspective screenshots of a 3D file using viewer.py --multi-view."""
    import subprocess
    cmd = [
        sys.executable, viewer_script,
        file_path,
        "--multi-view",
        "--screenshot", output_dir,
        "--bg-color", "0.05", "0.05", "0.05",
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode == 0:
        count = result.stdout.strip().count("Saved:")
        print(f"    ✓ {label}: {count} views → {output_dir}")
        return True
    else:
        print(f"    ✗ {label} failed: {result.stderr[:200] if result.stderr else 'unknown error'}")
        return False


def evaluate_with_viewer(output_dir, ply_path, mesh_path=None, object_mesh_paths=None):
    """Capture multi-perspective screenshots of PLY, scene mesh, and each object mesh."""
    print()
    print("=" * 60)
    print("STAGE 6: Evaluation — multi-perspective screenshots")
    print("=" * 60)

    viewer_script = os.path.join(_SCRIPT_DIR, "viewer.py")
    if not os.path.exists(viewer_script):
        print("  WARNING: viewer.py not found. Skipping evaluation.")
        return

    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # Multi-view screenshots for point cloud
    if ply_path and os.path.exists(ply_path):
        print(f"  Point cloud (6 views)...")
        _capture_multi_view(viewer_script, ply_path,
                            os.path.join(eval_dir, "pointcloud"),
                            "Point cloud", ["--point-size", "2.0"])

    # Multi-view screenshots for scene mesh
    if mesh_path and os.path.exists(mesh_path):
        print(f"  Scene mesh (6 views)...")
        _capture_multi_view(viewer_script, mesh_path,
                            os.path.join(eval_dir, "scene"),
                            "Scene mesh")

    # Multi-view screenshots for each individual object mesh
    if object_mesh_paths:
        for i, obj_path in enumerate(object_mesh_paths):
            if os.path.exists(obj_path):
                print(f"  Object {i} mesh (6 views)...")
                _capture_multi_view(viewer_script, obj_path,
                                    os.path.join(eval_dir, f"object_{i}"),
                                    f"Object {i}")

    # Print file info for all outputs
    print()
    print("  --- Output Summary ---")
    all_outputs = [("Point Cloud", ply_path), ("Scene Mesh", mesh_path)]
    if object_mesh_paths:
        for i, p in enumerate(object_mesh_paths):
            all_outputs.append((f"Object {i} Mesh", p))
    for label, path in all_outputs:
        if path and os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {label}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"  {label}: (not generated)")


# ---------------------------------------------------------------------------
#  Stage 7: Compute real-world volumes using ArUco reference
# ---------------------------------------------------------------------------
# Hard-coded: Object 1 is the ArUco marker, a 14x14x14 cm cube.
# Object 0 is the unknown object (e.g., vase) whose volume we want to measure.
REFERENCE_OBJECT_INDEX = 1        # which object is the ArUco reference
REFERENCE_REAL_SIZE_CM = 14.0     # real linear size of reference in cm


def compute_volumes(object_mesh_paths):
    """Compute real-world volume of each object using ArUco reference for scale.

    Uses object at REFERENCE_OBJECT_INDEX as the reference (known 14x14x14 cm cube).
    Formula:
        mesh_bbox_vol_ref = X_ref * Y_ref * Z_ref   (product of reference extents)
        k = real_size^3 / mesh_bbox_vol_ref         (volume scale factor)
        real_X = mesh_X * k^(1/3)
        real_volume = mesh_volume * k
    Example:
        Ref ArUco is 14x14x14 cm cube → real_volume = 14^3 = 2744 cm^3.
        Ref mesh bbox = 40 x 50 x 30 → bbox_vol = 60000.
        k = 2744 / 60000 = 0.04573.
        For object i with mesh bbox (X,Y,Z):
          real_X = X * k^(1/3), real_Y = Y * k^(1/3), real_Z = Z * k^(1/3).
        real_volume_i = mesh_volume_i * k.
    """
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
        if not os.path.exists(path):
            print(f"  [{i}] {path}: missing, skipping")
            continue
        mesh = trimesh.load(path, force="mesh")
        extents = mesh.bounds[1] - mesh.bounds[0]
        max_extent = float(np.max(extents))
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
        infos.append({
            "idx": i, "name": os.path.basename(path), "volume": volume,
            "max_extent": max_extent, "extents": extents, "bbox_vol": bbox_vol,
            "method": method,
        })
        print(f"  [{i}] {infos[-1]['name']}: "
              f"mesh_vol={volume:.6f}, bbox_vol={bbox_vol:.6f} "
              f"(extents {extents[0]:.4f}x{extents[1]:.4f}x{extents[2]:.4f}) ({method})")

    # Compute scale factor from reference object
    ref = next((x for x in infos if x["idx"] == REFERENCE_OBJECT_INDEX), None)
    if ref is None or ref["bbox_vol"] <= 0:
        print(f"  Reference object_{REFERENCE_OBJECT_INDEX} not available. Skipping.")
        return

    real_ref_vol = REFERENCE_REAL_SIZE_CM ** 3   # e.g., 14^3 = 2744 cm^3
    k = real_ref_vol / ref["bbox_vol"]            # volume scale factor
    cube_root_k = k ** (1.0 / 3.0)                # linear scale factor

    print(f"\n  Scale factor:")
    print(f"    ref bbox_vol = {ref['extents'][0]:.4f} * {ref['extents'][1]:.4f} * "
          f"{ref['extents'][2]:.4f} = {ref['bbox_vol']:.6f}")
    print(f"    real_ref_vol = {REFERENCE_REAL_SIZE_CM}^3 = {real_ref_vol:.2f} cm^3")
    print(f"    k            = real_ref_vol / mesh_bbox_vol = "
          f"{real_ref_vol:.2f} / {ref['bbox_vol']:.6f} = {k:.6f}")
    print(f"    k^(1/3)      = {cube_root_k:.6f}")

    # Report real-world values (volume = real_X * real_Y * real_Z = bbox_vol * k)
    print(f"\n  Real-world dimensions and volumes:")
    print(f"  {'IDX':>4} {'NAME':<20} {'SIZE (cm)':<24} {'VOLUME (cm^3)':>14}")
    print("  " + "-" * 68)
    for info in infos:
        ext_cm = info["extents"] * cube_root_k
        real_vol = float(ext_cm[0] * ext_cm[1] * ext_cm[2])
        size_str = f"{ext_cm[0]:6.2f} x {ext_cm[1]:6.2f} x {ext_cm[2]:6.2f}"
        marker = "  <- REF" if info["idx"] == REFERENCE_OBJECT_INDEX else ""
        print(f"  {info['idx']:>4} {info['name']:<20} {size_str:<24} "
              f"{real_vol:>14.2f}{marker}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = get_device()
    total_t0 = time.time()

    print(f"╔{'═' * 58}╗")
    print(f"║  VGGT Full Pipeline                                      ║")
    print(f"╠{'═' * 58}╣")
    print(f"║  Device        : {device:<40}║")
    print(f"║  Input         : {args.image_folder:<40}║")
    print(f"║  Pred. mode    : {args.prediction_mode:<40}║")
    print(f"║  Conf. thresh  : {args.conf_thres:<40}║")
    print(f"║  Skip mesh     : {str(args.skip_mesh):<40}║")
    print(f"║  Watertight    : {str(not args.no_watertight):<40}║")
    print(f"║  Evaluate      : {str(args.evaluate):<40}║")
    print(f"╚{'═' * 58}╝")

    # Validate input folder
    if not os.path.isdir(args.image_folder):
        print(f"\nERROR: Input folder not found: {args.image_folder}")
        sys.exit(1)

    # Resolve output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(_SCRIPT_DIR, "output")
    os.makedirs(args.output_dir, exist_ok=True)

    # Also set up a target_dir with images/ subfolder (for demo_gradio compatibility)
    target_dir = os.path.join(args.output_dir, "target")
    target_images_dir = os.path.join(target_dir, "images")
    os.makedirs(target_images_dir, exist_ok=True)

    # Copy images into target_dir/images/ structure
    image_files = sorted(glob.glob(os.path.join(args.image_folder, "*")))
    image_files = [p for p in image_files
                   if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))]
    for src in image_files:
        dst = os.path.join(target_images_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # ── Stage 1: Inference ──
    predictions, inference_time = run_inference(args.image_folder, device, args.max_frames)

    # Save predictions (compatible with demo_gradio)
    npz_path = os.path.join(target_dir, "predictions.npz")
    save_dict = {k: v for k, v in predictions.items() if v is not None}
    np.savez_compressed(npz_path, **save_dict)
    print(f"  Saved predictions: {npz_path}")

    # Also save in output root
    npz_path2 = os.path.join(args.output_dir, "predictions.npz")
    shutil.copy2(npz_path, npz_path2)

    # ── Stage 2: Export PLY ──
    ply_path = export_ply(predictions, args.output_dir, args)

    # ── Stage 3: Clean + Extract ──
    scene_recon_path = None
    recon_mesh_paths = []
    scene_wt_path = None
    wt_mesh_paths = []

    if not args.skip_mesh:
        object_paths = clean_and_extract(ply_path, args.output_dir, args.num_objects)
        if object_paths:
            # ── Stage 4: Reconstruct (Poisson, non-watertight) ──
            scene_recon_path, recon_mesh_paths = reconstruct_mesh_stage(
                object_paths, args.output_dir)

            # ── Stage 5: Make watertight (default; skip with --no-watertight) ──
            if recon_mesh_paths and not args.no_watertight:
                scene_wt_path, wt_mesh_paths = watertight_stage(
                    recon_mesh_paths, args.output_dir)
    else:
        print("\n  (Skipping mesh stages — --skip_mesh was set)")

    # Use watertight meshes for evaluation and volume if available, else recon meshes
    eval_scene = scene_wt_path or scene_recon_path
    eval_objects = wt_mesh_paths or recon_mesh_paths

    # ── Stage 6: Evaluate ──
    if args.evaluate:
        evaluate_with_viewer(args.output_dir, ply_path, eval_scene, eval_objects)

    # ── Stage 7: Compute real-world volumes (uses ArUco as reference) ──
    if eval_objects:
        compute_volumes(eval_objects)

    # ── Summary ──
    total_time = time.time() - total_t0
    print()
    print(f"╔{'═' * 58}╗")
    print(f"║  Pipeline Complete                                       ║")
    print(f"╠{'═' * 58}╣")
    print(f"║  Total time    : {total_time:>6.1f}s{' ' * 33}║")
    print(f"║  Inference     : {inference_time:>6.1f}s{' ' * 33}║")
    print(f"╠{'═' * 58}╣")
    print(f"║  Outputs:                                                ║")
    print(f"║    PLY         : {ply_path:<40}║")
    if scene_recon_path:
        print(f"║    Scene recon : {scene_recon_path:<40}║")
    for i, p in enumerate(recon_mesh_paths):
        print(f"║    Recon {i}     : {p:<40}║")
    if scene_wt_path:
        print(f"║    Scene wt    : {scene_wt_path:<40}║")
    for i, p in enumerate(wt_mesh_paths):
        print(f"║    Wt {i}        : {p:<40}║")
    print(f"║    Predictions : {npz_path2:<40}║")
    print(f"║    Target dir  : {target_dir:<40}║")
    print(f"╚{'═' * 58}╝")
    print()
    print("To view results interactively:")
    print(f"  python viewer.py {ply_path}")
    if scene_recon_path:
        print(f"  python viewer.py {scene_recon_path}  # recon scene")
    for i, p in enumerate(recon_mesh_paths):
        print(f"  python viewer.py {p}  # recon object {i}")
    if scene_wt_path:
        print(f"  python viewer.py {scene_wt_path}  # watertight scene")
    for i, p in enumerate(wt_mesh_paths):
        print(f"  python viewer.py {p}  # watertight object {i}")
    print()
    print("To use with demo_gradio.py:")
    print(f"  The predictions are saved at: {target_dir}/predictions.npz")
    print(f"  Images are at: {target_images_dir}/")


if __name__ == "__main__":
    main()
