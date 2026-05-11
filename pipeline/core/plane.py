"""Deterministic RANSAC plane detection and orientation helpers."""
import numpy as np


def auto_ransac_threshold(pcd, base_factor=3):
    """Estimate RANSAC distance_threshold from point cloud density."""
    distances = pcd.compute_nearest_neighbor_distance()
    median_nn = np.median(distances)
    threshold = median_nn * base_factor
    print(f"Auto RANSAC threshold: {threshold:.6f} (median nn: {median_nn:.6f} x {base_factor})")
    return threshold


def detect_plane_ransac_deterministic(pcd, distance_threshold=0.015, num_iterations=1000, seed=42):
    """Deterministic RANSAC plane detection using numpy."""
    pts = np.asarray(pcd.points)
    n = len(pts)
    if n < 3:
        raise ValueError("Need at least 3 points for plane detection")

    rng = np.random.RandomState(seed)
    best_plane = None
    best_inliers = np.array([], dtype=np.int64)

    for _ in range(num_iterations):
        i, j, k = rng.choice(n, size=3, replace=False)
        p1, p2, p3 = pts[i], pts[j], pts[k]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-12:
            continue
        normal = normal / norm_len

        d = -np.dot(normal, p1)
        plane = np.array([normal[0], normal[1], normal[2], d])

        distances = np.abs(np.dot(pts, normal) + d)
        inlier_mask = distances <= distance_threshold
        inlier_count = np.sum(inlier_mask)

        if inlier_count > len(best_inliers):
            best_inliers = np.where(inlier_mask)[0]
            best_plane = plane

    if best_plane is None:
        raise ValueError("Could not detect any plane")

    return best_plane, best_inliers


def get_rotation_to_z_axis(normal):
    """Calculates a rotation matrix to align the given normal with [0, 0, 1]."""
    normal = normal / np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal

    z_axis = np.array([0, 0, 1])
    v = np.cross(normal, z_axis)
    s = np.linalg.norm(v)
    c = np.dot(normal, z_axis)

    if s < 1e-8:
        return np.eye(3)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c) / (s ** 2))
    return r


def relocate_to_origin(pcd):
    """Moves the point cloud so XY is centered at 0 and Z-min is at 0."""
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()

    tx = -(min_bound[0] + max_bound[0]) / 2
    ty = -(min_bound[1] + max_bound[1]) / 2
    tz = -min_bound[2]

    pcd.translate((tx, ty, tz))
    return pcd
