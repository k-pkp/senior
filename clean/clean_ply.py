import os

import numpy as np
import open3d as o3d


def compute_eps(pcd, factor=4.0):
    """Estimate DBSCAN epsilon from average nearest-neighbor distance."""
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    print(f"Avg NN distance: {avg_dist:.6f}")
    return avg_dist * factor


def get_cluster_info(cluster):
    """Return extent, density, and maximum dimension of one cluster."""
    bbox = cluster.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    volume = np.prod(extent) if np.all(extent > 0) else 1e-6
    density = len(cluster.points) / volume
    max_dim = max(extent)
    return extent, density, max_dim


def detect_top_k_objects(pcd, k=2, visualize=False):
    """Detect and rank top-k clusters from a point cloud."""
    print("Running DBSCAN...")

    eps = compute_eps(pcd)
    print(f"Adaptive eps: {eps:.5f}")

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=10))
    valid = labels >= 0

    if not np.any(valid):
        print("No clusters found → using whole cloud as single object")
        return [pcd]

    unique_ids = np.unique(labels[valid])
    print(f"Total clusters found: {len(unique_ids)}")

    scene_bbox = pcd.get_axis_aligned_bounding_box()
    scene_size = np.linalg.norm(scene_bbox.get_extent())

    clusters = []

    for cid in unique_ids:
        idx = np.where(labels == cid)[0]
        cluster = pcd.select_by_index(idx)

        extent, density, max_dim = get_cluster_info(cluster)

        # Filter out tiny noise clusters (less than 1% of total)
        if len(cluster.points) < 0.01 * len(pcd.points):
            continue

        # Scoring: favor large, dense clusters
        score = (
            len(cluster.points) * 0.5 +
            density * 0.3 -
            max_dim * 0.2
        )

        clusters.append((cluster, score, len(cluster.points)))

    if len(clusters) == 0:
        print("No valid clusters → using whole cloud")
        return [pcd]

    clusters.sort(key=lambda x: x[1], reverse=True)

    selected = [c[0] for c in clusters[:k]]
    for i, (_, score, npts) in enumerate(clusters[:k]):
        print(f"  Object {i}: {npts:,} points, score={score:.1f}")

    print(f"Selected {len(selected)} objects")

    if visualize:
        print("Visualizing clusters...")
        max_label = labels.max()
        colors = np.random.rand(max_label + 1, 3)
        colors[labels < 0] = [0, 0, 0]
        pcd.colors = o3d.utility.Vector3dVector(colors[labels])
        o3d.visualization.draw_geometries([pcd])

    return selected


def clean_and_extract_objects(
    input_path,
    output_folder="output_objects",
    k=2,
    visualize=False,
    skip_plane=False,
    merge_clusters=False
):
    """Clean a point cloud, split into objects, and save each object as PLY.

    Adaptive cleaning: adjusts aggressiveness based on point cloud density
    and distribution. Avoids over-removing points from sparse reconstructions.

    If merge_clusters=True, skips DBSCAN and treats the whole cleaned cloud
    as a single object (useful when segmentation already isolated the target).
    """
    os.makedirs(output_folder, exist_ok=True)

    pcd = o3d.io.read_point_cloud(input_path)
    if len(pcd.points) == 0:
        raise ValueError("Empty point cloud")

    initial_count = len(pcd.points)
    print(f"Loaded point cloud: {initial_count:,} points")

    # ---- gentle outlier removal ----
    # Use adaptive std_ratio based on point count
    # Sparse clouds need gentler filtering
    if initial_count < 50000:
        std_ratio = 3.0
        nb_neighbors = 15
    elif initial_count < 200000:
        std_ratio = 2.5
        nb_neighbors = 20
    else:
        std_ratio = 2.0
        nb_neighbors = 20

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"Outlier removal: {initial_count:,} → {len(pcd.points):,} (std_ratio={std_ratio})")

    # ---- adaptive downsampling ----
    # Only downsample if very dense; preserve detail for sparse clouds
    if len(pcd.points) > 100000:
        voxel_size = 0.002
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Downsampled to {len(pcd.points):,} (voxel={voxel_size})")
    else:
        print(f"Skipping downsampling (only {len(pcd.points):,} points)")

    # ---- smart plane removal ----
    # Skip plane removal if segmentation already isolated the object
    if skip_plane:
        print("Plane removal skipped (segmentation already applied)")
    else:
        # Only remove a plane if it accounts for a significant portion but NOT most of the cloud
        # (if the plane IS the object, don't remove it)
        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.02,
                ransac_n=3,
                num_iterations=1000
            )
            plane_ratio = len(inliers) / len(pcd.points)

            if 0.15 < plane_ratio < 0.70:
                # Plane is a background surface (table, floor) — remove it
                pcd = pcd.select_by_index(inliers, invert=True)
                print(f"Plane removed ({plane_ratio*100:.0f}% of points)")
            else:
                print(f"Plane skipped ({plane_ratio*100:.0f}% — {'too small' if plane_ratio <= 0.15 else 'too large, likely the object itself'})")
        except Exception:
            print("Plane removal skipped (detection failed)")

    if len(pcd.points) == 0:
        raise ValueError("No points left after cleaning")

    print(f"Points after cleaning: {len(pcd.points):,}")

    # ---- detect objects ----
    if merge_clusters:
        print("Merge mode: treating entire cloud as single object (segmentation pre-filtered)")
        objects = [pcd]
    else:
        objects = detect_top_k_objects(pcd, k=k, visualize=visualize)

    # ---- save ----
    output_paths = []

    for i, obj in enumerate(objects):
        path = os.path.join(output_folder, f"object_{i}.ply")
        o3d.io.write_point_cloud(path, obj)
        output_paths.append(path)
        print(f"Saved: {path} ({len(obj.points):,} points)")

    return output_paths


if __name__ == "__main__":

    input_file = "input/input.ply"  # adjust if needed

    outputs = clean_and_extract_objects(
        input_file,
        output_folder="output_objects",
        k=2,
        visualize=False
    )

    print("Done:", outputs)
