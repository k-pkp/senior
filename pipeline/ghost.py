"""Ghost reduction — voxel dedup + normal-aware rejection.

VGGT emits duplicated, slightly offset copies of surfaces across views ("ghost"
layers). Voxel dedup collapses them to one point per cell; the normal-aware pass
then drops points whose orientation disagrees with their neighbourhood, which is
what a residual ghost layer looks like.

This is the pipeline's dominant decimation step — see GHOST_VOXEL_FACTOR.
"""
import numpy as np


def compute_voxel_size(points, factor=None, sample_size=5000):
    """voxel_size = factor * mean nearest-neighbour distance.

    Lower factor keeps more surface detail and more ghosts. Defaults to
    GHOST_VOXEL_FACTOR so the knob lives in config, not buried here.
    """
    if factor is None:
        from pipeline.config import GHOST_VOXEL_FACTOR
        factor = GHOST_VOXEL_FACTOR
    if len(points) < 5:
        return 0.01

    from scipy.spatial import cKDTree

    rng = np.random.default_rng(42)
    pts = points
    if len(points) > sample_size:
        pts = points[rng.choice(len(points), sample_size, replace=False)]
    dists, _ = cKDTree(pts).query(pts, k=2)
    return max(dists[:, 1].mean() * factor, 0.001)


def voxel_dedup(points, colors, voxel_size):
    """Collapse each occupied voxel to the centroid of its points.

    voxel_size <= 0 skips deduplication entirely and returns the cloud intact.
    That keeps maximum surface detail, at the cost of retaining VGGT's duplicate
    ghost layers — the normal-aware pass then has to remove them alone.
    """
    if len(points) == 0 or voxel_size <= 0:
        return points, colors

    origin = points.min(axis=0)
    voxel_idx = np.floor((points - origin) / voxel_size).astype(np.int32)
    _, inverse, counts = np.unique(voxel_idx, axis=0,
                                   return_inverse=True, return_counts=True)

    pts_out = np.zeros((len(counts), 3), dtype=np.float32)
    cols_out = np.zeros((len(counts), 3), dtype=np.float32)
    np.add.at(pts_out, inverse, points)
    np.add.at(cols_out, inverse, colors.astype(np.float32))
    pts_out /= counts[:, None]
    cols_out = (cols_out / counts[:, None]).astype(np.uint8)
    return pts_out, cols_out


def normal_aware_filter(points, colors, voxel_size, max_deviation=0.3, k=20):
    """Drop points whose normal disagrees with their local neighbourhood.

    A ghost layer sits parallel to the true surface but is populated sparsely,
    so its points' normals scatter relative to the local mean. Deviation is
    1 - |dot(n_i, mean_n)|, i.e. ~45 degrees at the 0.3 default.
    """
    if len(points) < k:
        return points, colors

    try:
        import open3d as o3d
    except ImportError:
        return points, colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=float(voxel_size * 3), max_nn=30))
    normals = np.asarray(pcd.normals)

    # One batched kNN query instead of a Python loop with a tree query per
    # point. At ~13k points the loop was tolerable; at 100k+ it dominates the
    # whole stage, and the density is exactly what we now want.
    from scipy.spatial import cKDTree

    _, idx = cKDTree(points).query(points, k=k, workers=-1)   # (N, k)

    mean_n = normals[idx].mean(axis=1)                        # (N, 3)
    norms = np.linalg.norm(mean_n, axis=1)
    valid = norms > 1e-8
    mean_n[valid] /= norms[valid, None]

    dev = 1.0 - np.abs(np.einsum("ij,ij->i", normals, mean_n))
    keep = ~(valid & (dev > max_deviation))

    rejected = int((~keep).sum())
    if rejected:
        print(f"  Normal-aware: {len(points):,} → {int(keep.sum()):,} points "
              f"(rejected {rejected:,}, k={k}, max_dev={max_deviation})")
    return points[keep], colors[keep]
