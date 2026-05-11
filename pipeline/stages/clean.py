"""Stage 3 — Clean the point cloud and separate it into per-object PLYs.

Steps:
    1. RANSAC-based leveling so the ground plane is horizontal (Z up).
    2. Adaptive statistical-outlier removal.
    3. Voxel downsampling if very dense.
    4. Smart laser-cut floor removal (only if a horizontal bottom plane exists).
    5. DBSCAN clustering + ArUco-aware ranking → top-k objects.
    6. Save object_0.ply (target) and object_1.ply (ArUco reference).
"""
import os

import numpy as np
import open3d as o3d

from pipeline.core.plane import (
    auto_ransac_threshold,
    detect_plane_ransac_deterministic,
    get_rotation_to_z_axis,
)
from pipeline.core.cluster import compute_eps, detect_top_k_objects


def clean_and_extract_objects(
    input_path,
    output_folder="output_objects",
    k=2,
    visualize=False,
    skip_plane=False,
    merge_clusters=False,
    seed=42,
    ransac_factor=3,
):
    os.makedirs(output_folder, exist_ok=True)

    pcd = o3d.io.read_point_cloud(input_path)
    if len(pcd.points) == 0:
        raise ValueError("Empty point cloud")

    initial_count = len(pcd.points)
    print(f"Loaded point cloud: {initial_count:,} points")

    # ---- STEP 1: LEVELING (FLIP) FIRST ----
    if not skip_plane:
        try:
            distance_threshold = auto_ransac_threshold(pcd, base_factor=ransac_factor)
            plane_model, _ = detect_plane_ransac_deterministic(
                pcd, distance_threshold=distance_threshold, num_iterations=1000, seed=seed
            )

            normal = plane_model[:3]
            print("Leveling scan based on detected plane...")
            R = get_rotation_to_z_axis(normal)
            pcd.rotate(R, center=(0, 0, 0))

            # --- SMART UPSIDE-DOWN CHECK ---
            _, inliers_rot = detect_plane_ransac_deterministic(
                pcd, distance_threshold=distance_threshold, num_iterations=500, seed=seed
            )
            pts_rot = np.asarray(pcd.points)
            plane_z = np.mean(pts_rot[inliers_rot, 2])

            mask = np.ones(len(pts_rot), dtype=bool)
            mask[inliers_rot] = False
            non_plane_z = pts_rot[mask, 2]

            if len(non_plane_z) > 0 and np.mean(non_plane_z) < plane_z:
                print("-> Detected upside-down orientation! Flipping 180 degrees...")
                flip_R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
                pcd.rotate(flip_R, center=(0, 0, 0))

        except Exception as e:
            print(f"Leveling failed: {e}")

    # ---- STEP 2: GENTLER OUTLIER REMOVAL ----
    if initial_count < 50000:
        std_ratio = 3.5
        nb_neighbors = 10
    elif initial_count < 200000:
        std_ratio = 3.0
        nb_neighbors = 15
    else:
        std_ratio = 2.5
        nb_neighbors = 20

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"Outlier removal: {initial_count:,} → {len(pcd.points):,} (std_ratio={std_ratio})")

    # ---- STEP 3: ADAPTIVE DOWNSAMPLING ----
    if len(pcd.points) > 100000:
        voxel_size = 0.002
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Downsampled to {len(pcd.points):,} (voxel={voxel_size})")

    # ---- STEP 4: SMART LASER CUT PLANE REMOVAL ----
    if not skip_plane:
        try:
            distance_threshold = auto_ransac_threshold(pcd, base_factor=ransac_factor)
            distance_threshold = min(distance_threshold, 0.015)

            plane_model, inliers = detect_plane_ransac_deterministic(
                pcd, distance_threshold=distance_threshold, num_iterations=1000, seed=seed
            )

            pts = np.asarray(pcd.points)
            plane_ratio = len(inliers) / len(pts)

            # --- AUTO-PLANE GEOMETRY CHECK ---
            normal = plane_model[:3]

            # 1. Horizontal? Dot product with Z-axis ≈ ±1
            is_horizontal = abs(np.dot(normal, [0, 0, 1])) > 0.85

            # 2. At the bottom of the scene?
            z_vals = pts[:, 2]
            z_min, z_max = np.min(z_vals), np.max(z_vals)
            plane_z = np.median(pts[inliers, 2])
            is_at_bottom = (plane_z - z_min) < 0.25 * (z_max - z_min)

            if is_horizontal and is_at_bottom and plane_ratio > 0.05:
                margin = 0.008
                keep_mask = pts[:, 2] > (plane_z + margin)
                pcd = pcd.select_by_index(np.where(keep_mask)[0])
                print(f"-> Smart Laser-cut floor at Z={plane_z:.4f} "
                      f"({plane_ratio*100:.0f}% of points removed)")
            else:
                reason = []
                if not is_horizontal:
                    reason.append("not horizontal")
                if not is_at_bottom:
                    reason.append("not at the bottom")
                if plane_ratio <= 0.05:
                    reason.append("too small (<5%)")
                print(f"Plane skipped: {', '.join(reason)} ({plane_ratio*100:.0f}% of points)")

        except Exception as e:
            print(f"Plane removal skipped (detection failed): {e}")

    # ---- STEP 5: DETECT OBJECTS ----
    if merge_clusters:
        print("Merge mode: filtering noise, merging all clusters")
        eps = compute_eps(pcd)
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=10))
        noise_mask = labels < 0
        if np.any(~noise_mask):
            pcd = pcd.select_by_index(np.where(~noise_mask)[0])
            print(f"  Removed {noise_mask.sum():,} noise points, kept {len(pcd.points):,}")
        objects = [pcd]
    else:
        objects = detect_top_k_objects(pcd, k=k, visualize=visualize)

    # ---- STEP 6: SAVE ----
    output_paths = []
    for i, obj in enumerate(objects):
        path = os.path.join(output_folder, f"object_{i}.ply")
        o3d.io.write_point_cloud(path, obj)
        output_paths.append(path)
        pts = np.asarray(obj.points)
        rng_x = f"[{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]"
        rng_y = f"[{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]"
        rng_z = f"[{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]"
        print(f"Saved: {path} ({len(obj.points):,} pts) X{rng_x} Y{rng_y} Z{rng_z}")

    return output_paths


def clean_and_extract(ply_path, output_dir, num_objects=2, seed=42):
    """Pipeline wrapper around clean_and_extract_objects.

    Returns the list of per-object PLY paths, or None on failure.
    """
    print()
    print("=" * 60)
    print("STAGE 3: Cleaning point cloud and extracting objects")
    print("=" * 60)

    clean_output_dir = os.path.join(output_dir, "clean_objects")
    os.makedirs(clean_output_dir, exist_ok=True)

    try:
        object_paths = clean_and_extract_objects(
            input_path=ply_path,
            output_folder=clean_output_dir,
            k=num_objects,
            visualize=False,
            seed=seed,
        )
        print(f"  Extracted {len(object_paths)} objects:")
        for p in object_paths:
            print(f"    → {p}")
        return object_paths
    except Exception as e:
        print(f"  ERROR during cleaning: {e}")
        return None
