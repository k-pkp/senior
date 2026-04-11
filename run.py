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
Integrates the full demo_gradio pipeline: inference → export PLY → clean → reconstruct mesh.
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
        description="VGGT — run full pipeline: segment → inference → PLY → clean → mesh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                                  # default: ./baam/ input
  python run.py --image_folder ./data/train/images/ --segment
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
    # Segmentation options
    p.add_argument("--segment", action="store_true",
                   help="Pre-segment images to isolate target object before reconstruction")
    p.add_argument("--seg_target", type=str, default="vase",
                   choices=["vase", "aruco", "both"],
                   help="Which object(s) to reconstruct: vase, aruco, or both (default: vase)")
    p.add_argument("--seg_checkpoint", type=str,
                   default=os.path.join(_SCRIPT_DIR, "detect", "checkpoints", "best_seg_model.pt"),
                   help="Path to segmentation model checkpoint")
    p.add_argument("--seg_img_size", type=int, default=256,
                   help="Segmentation model input size (default: 256)")
    return p.parse_args()


# ---------------------------------------------------------------------------
#  Stage 0: Segmentation — isolate target object in images
# ---------------------------------------------------------------------------
def generate_seg_masks(image_folder, output_dir, args, device):
    """Generate multi-class segmentation masks using trained Transformer-CNN model.

    Saves full multi-class masks (0=bg, 1=ArUco, 2=Vase) so both objects
    are reconstructed together and separated at the mesh stage.

    Returns path to mask directory.
    """
    print()
    print("=" * 60)
    print("STAGE 0: Generating multi-class segmentation masks")
    print("=" * 60)

    sys.path.insert(0, os.path.join(_SCRIPT_DIR, "detect"))
    from segment_pipeline import load_seg_model, segment_image
    import cv2

    mask_dir = os.path.join(output_dir, "seg_masks")
    os.makedirs(mask_dir, exist_ok=True)

    device_str = str(device)
    model, num_classes = load_seg_model(args.seg_checkpoint, device_str)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in img_exts
    ])

    # Also save visual overlays on real images
    vis_dir = os.path.join(output_dir, "masks")
    os.makedirs(vis_dir, exist_ok=True)

    count = 0
    for fname in image_files:
        img_path = os.path.join(image_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        # Save full multi-class mask: 0=bg, 1=ArUco, 2=Vase
        full_mask = segment_image(model, img, img_size=args.seg_img_size, device=device_str)
        mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + ".npy")
        np.save(mask_path, full_mask.astype(np.uint8))

        # Generate overlay visualization on real image
        h, w = img.shape[:2]
        mask_resized = cv2.resize(full_mask.astype(np.uint8), (w, h),
                                  interpolation=cv2.INTER_NEAREST)
        overlay = img.copy()
        # Dim background
        overlay[mask_resized == 0] = (overlay[mask_resized == 0] * 0.3).astype(np.uint8)
        # Color overlays: ArUco=blue, Vase=green
        aruco_mask = mask_resized == 1
        vase_mask = mask_resized == 2
        overlay[aruco_mask] = (overlay[aruco_mask] * 0.5 + np.array([200, 100, 0]) * 0.5).astype(np.uint8)
        overlay[vase_mask] = (overlay[vase_mask] * 0.5 + np.array([0, 200, 100]) * 0.5).astype(np.uint8)
        # Draw contours
        for cls_id, color in [(1, (255, 150, 0)), (2, (0, 255, 100))]:
            binary = (mask_resized == cls_id).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

        vis_path = os.path.join(vis_dir, os.path.splitext(fname)[0] + ".jpg")
        cv2.imwrite(vis_path, overlay)
        count += 1

    print(f"  Generated {count} multi-class masks → {mask_dir}")
    print(f"  Overlay visualizations → {vis_dir}")
    print(f"  Classes: 0=background, 1=ArUco (blue), 2=Vase (green)")
    del model
    return mask_dir, count


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


def export_ply(predictions, output_dir, args, seg_masks=None):
    """Filter and export point cloud as PLY with adaptive confidence and outlier removal.

    If seg_masks is provided (multi-class: 0=bg, 1=ArUco, 2=Vase),
    keeps all object pixels and saves per-point class labels for mesh separation.
    """
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

    S, H, W, _ = world_points.shape
    points_flat = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    # Adaptive confidence filter
    threshold_val = _adaptive_confidence_filter(conf_flat, args.conf_thres)
    mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

    # Apply segmentation masks: keep both object classes, store per-pixel class IDs
    class_labels_flat = None
    if seg_masks is not None:
        import cv2
        seg_class_flat = np.zeros(S * H * W, dtype=np.uint8)
        for i in range(min(S, len(seg_masks))):
            m = seg_masks[i]  # (orig_H, orig_W) with values 0, 1, 2
            m_resized = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            seg_class_flat[i * H * W : (i + 1) * H * W] = m_resized.reshape(-1)
        mask &= (seg_class_flat > 0)  # keep both ArUco (1) and Vase (2)
        class_labels_flat = seg_class_flat
        n_aruco = (seg_class_flat == 1).sum()
        n_vase = (seg_class_flat == 2).sum()
        n_total = S * H * W
        print(f"  Segmentation: ArUco={n_aruco:,}, Vase={n_vase:,}, "
              f"total kept={(n_aruco+n_vase):,}/{n_total:,} ({100*(n_aruco+n_vase)/n_total:.1f}%)")

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
    # Track which indices survive for class label alignment
    pre_outlier_count = len(points_out)
    points_out, colors_out, conf_out = _remove_spatial_outliers(points_out, colors_out, conf_out)

    print(f"  Final point count: {points_out.shape[0]:,}")

    ply_path = os.path.join(output_dir, "points.ply")
    pc = trimesh.PointCloud(points_out, colors=colors_out)
    pc.export(ply_path)
    print(f"  Exported: {ply_path}")

    # Save per-point class labels for mesh separation
    if class_labels_flat is not None:
        labels_out = class_labels_flat[mask]
        # If outlier removal changed the count, re-align via nearest neighbor
        if len(labels_out) != len(points_out):
            from scipy.spatial import KDTree
            pre_points = points_flat[mask]
            tree = KDTree(pre_points)
            _, idx = tree.query(points_out)
            labels_out = labels_out[idx]
        labels_path = os.path.join(output_dir, "point_labels.npz")
        np.savez_compressed(labels_path, points=points_out, labels=labels_out,
                            colors=colors_out)
        n_a = (labels_out == 1).sum()
        n_v = (labels_out == 2).sum()
        print(f"  Class labels saved: ArUco={n_a:,}, Vase={n_v:,} → {labels_path}")

    return ply_path


# ---------------------------------------------------------------------------
#  Stage 3: Clean point cloud and extract objects
# ---------------------------------------------------------------------------
def clean_and_extract(ply_path, output_dir, num_objects=2, skip_plane=False,
                      merge_clusters=False):
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
            skip_plane=skip_plane,
            merge_clusters=merge_clusters,
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
    """Reconstruct mesh from cleaned object point clouds."""
    print()
    print("=" * 60)
    print("STAGE 4: Reconstructing mesh from objects")
    print("=" * 60)

    try:
        from recons import reconstruct_multiple_objects
    except ImportError:
        print("  WARNING: recons module not found. Skipping reconstruction.")
        print("  Make sure ./clean/recons.py exists.")
        return None

    mesh_output_dir = os.path.join(output_dir, "mesh")
    os.makedirs(mesh_output_dir, exist_ok=True)

    try:
        mesh_path, _ = reconstruct_multiple_objects(
            input_paths=object_paths,
            output_folder=mesh_output_dir,
            base_name="scene",
        )
        print(f"  Reconstructed mesh: {mesh_path}")
        return mesh_path
    except Exception as e:
        print(f"  ERROR during reconstruction: {e}")
        return None


# ---------------------------------------------------------------------------
#  Stage 4.5: Separate mesh into per-class meshes (Vase / ArUco)
# ---------------------------------------------------------------------------
def separate_mesh(mesh_path, labels_path, output_dir):
    """Separate reconstructed mesh into per-class meshes using point class labels.

    Uses KDTree to assign each mesh vertex to the nearest class-labeled point,
    then splits faces by majority vote of vertex labels.
    """
    print()
    print("=" * 60)
    print("STAGE 4.5: Separating mesh into Vase and ArUco")
    print("=" * 60)

    import open3d as o3d
    from scipy.spatial import KDTree

    if not os.path.exists(labels_path):
        print(f"  WARNING: Point labels not found at {labels_path}. Skipping separation.")
        return {}

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    has_colors = mesh.has_vertex_colors()
    vertex_colors = np.asarray(mesh.vertex_colors) if has_colors else None

    print(f"  Combined mesh: {len(vertices):,} vertices, {len(triangles):,} faces")

    # Load class-labeled points
    data = np.load(labels_path)
    labeled_points = data["points"]
    labels = data["labels"]
    print(f"  Labeled points: {len(labeled_points):,} "
          f"(ArUco={int((labels==1).sum()):,}, Vase={int((labels==2).sum()):,})")

    # Build KD-tree and assign each vertex to nearest labeled point's class
    tree = KDTree(labeled_points)
    dists, indices = tree.query(vertices)
    vertex_labels = labels[indices]

    sep_dir = os.path.join(output_dir, "separated")
    os.makedirs(sep_dir, exist_ok=True)

    class_names = {1: "aruco", 2: "vase"}
    results = {}

    for cls_id, cls_name in class_names.items():
        # Majority vote: face belongs to class if >=2 of 3 vertices have that class
        face_vertex_labels = vertex_labels[triangles]  # (F, 3)
        face_mask = (face_vertex_labels == cls_id).sum(axis=1) >= 2

        cls_faces = triangles[face_mask]
        if len(cls_faces) == 0:
            print(f"  {cls_name}: no faces — skipped")
            continue

        # Re-index vertices used by these faces
        unique_verts, inverse = np.unique(cls_faces.ravel(), return_inverse=True)
        new_verts = vertices[unique_verts]
        new_faces = inverse.reshape(-1, 3)

        cls_mesh = o3d.geometry.TriangleMesh()
        cls_mesh.vertices = o3d.utility.Vector3dVector(new_verts)
        cls_mesh.triangles = o3d.utility.Vector3iVector(new_faces)
        cls_mesh.compute_vertex_normals()

        if has_colors:
            cls_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors[unique_verts])

        # Remove small disconnected components (boundary artifacts)
        components = cls_mesh.cluster_connected_triangles()
        cluster_ids = np.asarray(components[0])
        cluster_areas = np.asarray(components[2])
        if len(cluster_areas) > 1:
            largest = np.argmax(cluster_areas)
            keep_mask = cluster_ids == largest
            cls_mesh.remove_triangles_by_mask(~keep_mask)
            cls_mesh.remove_unreferenced_vertices()
            removed_components = len(cluster_areas) - 1
            print(f"  {cls_name}: removed {removed_components} small components")

        out_path = os.path.join(sep_dir, f"{cls_name}.ply")
        o3d.io.write_triangle_mesh(out_path, cls_mesh)
        results[cls_name] = out_path

        final_verts = len(np.asarray(cls_mesh.vertices))
        final_faces = len(np.asarray(cls_mesh.triangles))
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  {cls_name}: {final_faces:,} faces, {final_verts:,} vertices "
              f"→ {out_path} ({size_mb:.1f} MB)")

    return results


# ---------------------------------------------------------------------------
#  Stage 5: Evaluate — auto-screenshot using viewer.py
# ---------------------------------------------------------------------------
def _capture_screenshot(viewer_script, file_path, screenshot_path, extra_args=None):
    """Helper to capture a screenshot of a 3D file using viewer.py."""
    import subprocess
    cmd = [
        sys.executable, viewer_script,
        file_path,
        "--auto-screenshot",
        "--screenshot", screenshot_path,
        "--bg-color", "0.05", "0.05", "0.05",
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode == 0:
        print(f"    ✓ Saved: {screenshot_path}")
        return True
    else:
        print(f"    ✗ Failed: {result.stderr[:200] if result.stderr else 'unknown error'}")
        return False


def evaluate_with_viewer(output_dir, ply_path, mesh_path=None, separated_meshes=None):
    """Capture screenshots of PLY, combined mesh, and separated meshes."""
    print()
    print("=" * 60)
    print("STAGE 5: Evaluation — capturing screenshots")
    print("=" * 60)

    viewer_script = os.path.join(_SCRIPT_DIR, "viewer.py")
    if not os.path.exists(viewer_script):
        print("  WARNING: viewer.py not found. Skipping evaluation.")
        return

    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # Screenshot point cloud
    if ply_path and os.path.exists(ply_path):
        print(f"  Capturing point cloud screenshot...")
        _capture_screenshot(viewer_script, ply_path,
                            os.path.join(eval_dir, "pointcloud.png"),
                            ["--point-size", "2.0"])

    # Screenshot combined mesh
    if mesh_path and os.path.exists(mesh_path):
        print(f"  Capturing combined mesh screenshot...")
        _capture_screenshot(viewer_script, mesh_path,
                            os.path.join(eval_dir, "mesh_combined.png"))

    # Screenshot separated meshes
    if separated_meshes:
        for cls_name, cls_path in separated_meshes.items():
            if os.path.exists(cls_path):
                print(f"  Capturing {cls_name} mesh screenshot...")
                _capture_screenshot(viewer_script, cls_path,
                                    os.path.join(eval_dir, f"mesh_{cls_name}.png"))

    # Print file info for all outputs
    print()
    print("  --- Output Summary ---")
    all_outputs = [("Point Cloud", ply_path), ("Combined Mesh", mesh_path)]
    if separated_meshes:
        for cls_name, cls_path in separated_meshes.items():
            all_outputs.append((f"Mesh ({cls_name})", cls_path))
    for label, path in all_outputs:
        if path and os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {label}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"  {label}: (not generated)")


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
    print(f"║  Segment       : {str(args.segment):<40}║")
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

    # ── Stage 0: Segmentation masks (optional) ──
    mask_dir = None
    if args.segment:
        if not os.path.exists(args.seg_checkpoint):
            print(f"\nERROR: Segmentation checkpoint not found: {args.seg_checkpoint}")
            print("Train the model first: python detect/train_seg.py --data_dir ./data")
            sys.exit(1)
        mask_dir, _ = generate_seg_masks(args.image_folder, args.output_dir, args, device)

    # ── Stage 1: Inference (on ORIGINAL images for better correspondences) ──
    predictions, inference_time = run_inference(args.image_folder, device, args.max_frames)

    # Save predictions (compatible with demo_gradio)
    npz_path = os.path.join(target_dir, "predictions.npz")
    save_dict = {k: v for k, v in predictions.items() if v is not None}
    np.savez_compressed(npz_path, **save_dict)
    print(f"  Saved predictions: {npz_path}")

    # Also save in output root
    npz_path2 = os.path.join(args.output_dir, "predictions.npz")
    shutil.copy2(npz_path, npz_path2)

    # ── Stage 2: Export PLY (with segmentation mask filtering if enabled) ──
    seg_masks_for_ply = None
    if args.segment:
        # Load multi-class masks (.npy) in the same order as the selected frames
        image_names = sorted(glob.glob(os.path.join(args.image_folder, "*")))
        image_names = [p for p in image_names
                       if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))]
        image_names = _select_frames(image_names, args.max_frames)
        seg_masks_for_ply = []
        for img_path in image_names:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(mask_dir, stem + ".npy")
            if os.path.exists(mask_path):
                m = np.load(mask_path)  # multi-class: 0=bg, 1=ArUco, 2=Vase
                seg_masks_for_ply.append(m)
            else:
                # No mask found — include all points (mark as bg=0, will be excluded)
                seg_masks_for_ply.append(np.zeros((518, 518), dtype=np.uint8))
        print(f"  Loaded {len(seg_masks_for_ply)} multi-class masks for point filtering")

    ply_path = export_ply(predictions, args.output_dir, args, seg_masks=seg_masks_for_ply)

    # ── Stage 3 & 4: Clean + Reconstruct combined mesh (both objects) ──
    mesh_path = None
    separated_meshes = {}
    if not args.skip_mesh:
        # Merge all clusters into one combined cloud for joint reconstruction
        object_paths = clean_and_extract(ply_path, args.output_dir, args.num_objects,
                                          skip_plane=args.segment,
                                          merge_clusters=args.segment)
        if object_paths:
            mesh_path = reconstruct_mesh_stage(object_paths, args.output_dir)

        # ── Stage 4.5: Separate combined mesh into Vase / ArUco ──
        labels_path = os.path.join(args.output_dir, "point_labels.npz")
        if mesh_path and args.segment and os.path.exists(labels_path):
            separated_meshes = separate_mesh(mesh_path, labels_path, args.output_dir)
    else:
        print("\n  (Skipping mesh stages — --skip_mesh was set)")

    # ── Stage 5: Evaluate ──
    if args.evaluate:
        evaluate_with_viewer(args.output_dir, ply_path, mesh_path, separated_meshes)

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
    if mesh_path:
        print(f"║    Combined    : {mesh_path:<40}║")
    for cls_name, cls_path in separated_meshes.items():
        print(f"║    {cls_name:<13}: {cls_path:<40}║")
    print(f"║    Predictions : {npz_path2:<40}║")
    print(f"║    Target dir  : {target_dir:<40}║")
    print(f"╚{'═' * 58}╝")
    print()
    print("To view results interactively:")
    print(f"  python viewer.py {ply_path}")
    if mesh_path:
        print(f"  python viewer.py {mesh_path}")
    for cls_name, cls_path in separated_meshes.items():
        print(f"  python viewer.py {cls_path}  # {cls_name}")
    print()
    print("To use with demo_gradio.py:")
    print(f"  The predictions are saved at: {target_dir}/predictions.npz")
    print(f"  Images are at: {target_images_dir}/")


if __name__ == "__main__":
    main()
