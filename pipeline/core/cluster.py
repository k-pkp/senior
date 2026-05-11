"""DBSCAN clustering + ArUco-aware ranking."""
import numpy as np


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


def aruco_cubeness(cluster):
    """Score 0..1: how cube-like the cluster bbox is (ArUco = 14cm cube → ~1.0)."""
    extent = cluster.get_axis_aligned_bounding_box().get_extent()
    mx = float(np.max(extent))
    if mx <= 1e-9:
        return 0.0
    return float(np.min(extent) / mx)


def aruco_bw_ratio(cluster):
    """Score 0..1: fraction of points near-black or near-white (ArUco signature)."""
    if not cluster.has_colors():
        return 0.0
    colors = np.asarray(cluster.colors)
    if len(colors) == 0:
        return 0.0
    brightness = colors.mean(axis=1)
    frac_black = float((brightness < 0.20).mean())
    frac_white = float((brightness > 0.80).mean())
    return frac_black + frac_white


def detect_top_k_objects(pcd, k=2, visualize=False):
    """Detect and rank top-k clusters from a point cloud.

    Cluster order: object_0 = non-ArUco object(s), object_{k-1} = ArUco reference.
    ArUco identified by combined cubeness + black/white color signature.
    """
    del visualize  # currently unused; reserved for future debug viewer
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

    clusters = []
    for cid in unique_ids:
        idx = np.where(labels == cid)[0]
        cluster = pcd.select_by_index(idx)
        _extent, density, max_dim = get_cluster_info(cluster)

        if len(cluster.points) < 0.01 * len(pcd.points):
            continue

        score = (len(cluster.points) * 0.5 + density * 0.3 - max_dim * 0.2)
        clusters.append((cluster, score, len(cluster.points)))

    if len(clusters) == 0:
        print("No valid clusters → using whole cloud")
        return [pcd]

    clusters.sort(key=lambda x: x[1], reverse=True)
    top = clusters[:k]

    # Rank ArUco-likeness: cubeness (0.6) + B/W color signature (0.4)
    aruco_scores = []
    for cluster, _, _ in top:
        cube = aruco_cubeness(cluster)
        bw = aruco_bw_ratio(cluster)
        aruco_scores.append(cube * 0.6 + bw * 0.4)

    # Reorder: lowest ArUco score first (object_0), highest last (object_{k-1})
    order = np.argsort(aruco_scores)
    selected = [top[i][0] for i in order]

    print("  ArUco-likeness reorder (cubeness*0.6 + bw*0.4):")
    for new_idx, src_idx in enumerate(order):
        _, score, npts = top[src_idx]
        a = aruco_scores[src_idx]
        tag = "  <- ArUco ref" if new_idx == len(order) - 1 else ""
        print(f"    object_{new_idx}: {npts:,} pts, cluster_score={score:.1f}, "
              f"aruco_score={a:.3f}{tag}")

    return selected
