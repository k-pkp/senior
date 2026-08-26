"""Ghost reduction — voxel dedup, normal-aware rejection, and MLS projection.

VGGT emits duplicated, slightly offset copies of a surface across views ("ghost"
sheets, typically ~2 mm apart). Removing them takes all three functions here, and
they do genuinely different jobs — the chain was mis-attributed twice before it
was measured, so it is worth stating plainly:

  - `ghost_voxel_downsample` replaces the points in each occupied voxel with
    their centroid. It does **not** remove the ghost: two sheets about 2.7 mm
    apart usually land in different voxels and both survive. What it does is set
    the point spacing, and since MLS's radius is measured in multiples of
    spacing, this is what makes MLS's neighbourhood physically wide enough to
    span the gap. It is the dominant decimation step — see GHOST_VOXEL_FACTOR —
    and removing it costs +2.59% on the reported volume.

  - `normal_aware_filter` drops points whose orientation disagrees with their
    neighbourhood. A surviving ghost fragment is sparse, so its normals scatter.
    This removes ~3% of points and is not load-bearing on the volume.

  - `mls_project` is what actually collapses the sheets. Rather than deciding
    which one to delete, it fits a local surface through each neighbourhood and
    projects every point onto it, so both sheets land on one and the shell goes
    from ~1.76 mm thick to ~0.79 mm.

Measured step by step in docs/experiments/ghost_removal_chain.md.
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


def ghost_voxel_downsample(points, colors, voxel_size):
    """Replace the points in each occupied voxel with their centroid.

    This is `open3d.geometry.PointCloud.voxel_down_sample`. It replaced a
    hand-rolled 30-line grid (`voxel_dedup`) on 2026-08-23, after an A/B on
    `inputs/small_leg` put the two **0.15% apart** on the reported limb volume —
    1083.54 cm³ against 1081.94. The only mechanical difference is where the grid
    is anchored: the old code used `points.min(axis=0)`, Open3D uses its own
    origin, so a few points near a cell boundary land on the other side and the
    centroids shift slightly. See docs/experiments/ghost_removal_chain.md.

    The name says downsample, not dedup, because that is what it does. It does
    **not** remove the ghost: the two sheets are about 2.7 mm apart and usually
    fall in different voxels, so both survive. Its job is to set the point
    spacing, which is what gives `mls_project` a neighbourhood wide enough to
    span the gap — `MLS_RADIUS_MULT` is measured in spacings, not millimetres.

    Three things this wrapper exists to guarantee, all of which a bare library
    call would lose:

      - `voxel_size <= 0` returns the cloud untouched. `GHOST_VOXEL_FACTOR = 0`
        is the documented way to disable this step, and the experiment measuring
        what the step is worth (+2.59% on the limb) depends on it.
      - colours survive as uint8. Open3D wants floats in 0-1 and hands them back
        the same way, and it averages colours only if the cloud actually carries
        them.
      - an empty cloud is returned as-is rather than raising.
    """
    if len(points) == 0 or voxel_size <= 0:
        return points, colors

    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    has_colours = colors is not None and len(colors) == len(points)
    if has_colours:
        cloud.colors = o3d.utility.Vector3dVector(
            np.asarray(colors, dtype=np.float64) / 255.0)

    thinned = cloud.voxel_down_sample(voxel_size)

    points_out = np.asarray(thinned.points, dtype=np.float32)
    if not has_colours:
        return points_out, colors
    colours_out = (np.clip(np.asarray(thinned.colors), 0.0, 1.0) * 255.0
                   ).round().astype(np.uint8)
    return points_out, colours_out


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


# ---------------------------------------------------------------------------
# Moving-least-squares projection
# ---------------------------------------------------------------------------
# Moved here from pipeline/mls.py, which no longer exists. It lived in its own
# module while the ghost mechanism was still being attributed; now that the
# chain above is measured, keeping the three steps in one file is what makes the
# division of labour between them legible.
#
# Filtering alone cannot remove the ghost — see docs/experiments.md T10: it is
# the same model error repeated in every view, so multi-view corroboration
# accepts it, and it is parallel to the true surface so the normal-aware filter
# is blind to it. MLS takes the other route and moves the points instead.
#
# This MOVES points rather than removing them, so it is a genuine change to the
# geometry, not a cleanup. Over-smoothing erases real detail, which is why the
# neighbourhood radius is expressed in multiples of point spacing rather than as
# an absolute distance.


def mls_project(points, colors=None, radius_mult=3.0, min_neighbors=8,
                polynomial=True, verbose=True):
    """Project every point onto a locally fitted surface.

    Args:
        points: (N, 3) float
        colors: (N, 3) uint8 or None — carried through untouched
        radius_mult: neighbourhood radius, in multiples of mean NN spacing.
            Must exceed the ghost separation or the two sheets never meet in a
            single neighbourhood and nothing merges.
        min_neighbors: below this a point is left where it is.
        polynomial: fit a degree-2 height field over the local plane, which
            preserves curvature. False fits a plane, which flattens it.

    Returns:
        (points_projected, colors, stats dict)
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    point_count = len(points)

    def _stats(spacing=0.0, radius=0.0, median_move=0.0, p95_move=0.0, skipped=0):
        """One shape for the stats dict, whichever way the function returns.

        The early return used to hand back {"moved_mm": 0.0} while the normal
        path returned five different keys, so any caller that read the normal
        keys would raise KeyError on a cloud too small to project.
        """
        return {"spacing": spacing, "radius": radius, "median_move": median_move,
                "p95_move": p95_move, "skipped": skipped}

    if point_count < min_neighbors:
        return points, colors, _stats(skipped=point_count)

    tree = cKDTree(points)

    # Mean nearest-neighbour distance, sampled rather than measured over every
    # point because the mean converges long before 5000 samples and the query is
    # the expensive part. k=2 because the nearest neighbour of a point is itself.
    sample_indices = np.random.default_rng(0).choice(
        point_count, min(5000, point_count), replace=False)
    neighbour_distances, _ = tree.query(points[sample_indices], k=2, workers=-1)
    point_spacing = float(neighbour_distances[:, 1].mean())
    neighbourhood_radius = point_spacing * radius_mult

    neighbourhoods = tree.query_ball_point(points, r=neighbourhood_radius, workers=-1)

    projected = points.copy()
    distance_moved = np.zeros(point_count)
    skipped_count = 0

    for point_index, neighbour_indices in enumerate(neighbourhoods):
        if len(neighbour_indices) < min_neighbors:
            skipped_count += 1
            continue

        neighbour_points = points[neighbour_indices]
        neighbourhood_centre = neighbour_points.mean(axis=0)
        centred = neighbour_points - neighbourhood_centre

        # Local frame from the neighbourhood's own spread: the least-variance
        # direction is the surface normal, the other two span the tangent plane.
        _, _, principal_axes = np.linalg.svd(centred, full_matrices=False)
        normal = principal_axes[2]
        tangent_u, tangent_v = principal_axes[0], principal_axes[1]

        # Re-express each neighbour as (position in the tangent plane, height
        # above it) so the surface can be fitted as a height field.
        offset_u = centred @ tangent_u
        offset_v = centred @ tangent_v
        height = centred @ normal

        if polynomial and len(neighbour_indices) >= 6:
            # height ~ c0 + c1*u + c2*v + c3*u² + c4*uv + c5*v² — curved, so a
            # limb keeps its curvature instead of being flattened.
            design = np.column_stack([np.ones_like(offset_u), offset_u, offset_v,
                                      offset_u * offset_u,
                                      offset_u * offset_v,
                                      offset_v * offset_v])
            try:
                coefficients, *_ = np.linalg.lstsq(design, height, rcond=None)
            except np.linalg.LinAlgError:
                skipped_count += 1
                continue
        else:
            # height ~ c0 + c1*u + c2*v — a flat patch. Pad the quadratic terms
            # with zeros so the evaluation below is the same expression either way.
            design = np.column_stack([np.ones_like(offset_u), offset_u, offset_v])
            coefficients, *_ = np.linalg.lstsq(design, height, rcond=None)
            coefficients = np.concatenate([coefficients, np.zeros(3)])

        # Evaluate the fitted surface at this point's own tangent coordinates,
        # then move the point there.
        this_u = float((points[point_index] - neighbourhood_centre) @ tangent_u)
        this_v = float((points[point_index] - neighbourhood_centre) @ tangent_v)
        fitted_height = (coefficients[0]
                         + coefficients[1] * this_u
                         + coefficients[2] * this_v
                         + coefficients[3] * this_u * this_u
                         + coefficients[4] * this_u * this_v
                         + coefficients[5] * this_v * this_v)

        target = (neighbourhood_centre
                  + this_u * tangent_u
                  + this_v * tangent_v
                  + fitted_height * normal)
        distance_moved[point_index] = np.linalg.norm(target - points[point_index])
        projected[point_index] = target

    stats = _stats(spacing=point_spacing,
                   radius=neighbourhood_radius,
                   median_move=float(np.median(distance_moved)),
                   p95_move=float(np.percentile(distance_moved, 95)),
                   skipped=skipped_count)
    if verbose:
        print(f"  MLS: radius {neighbourhood_radius:.5f} "
              f"({radius_mult:.1f}x spacing), "
              f"moved median {stats['median_move']:.5f}, "
              f"p95 {stats['p95_move']:.5f}, skipped {skipped_count:,}")
    return projected.astype(np.float32), colors, stats
