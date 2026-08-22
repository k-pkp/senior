"""Moving-least-squares surface projection.

VGGT emits a duplicate copy of the surface a couple of millimetres from the true
one. Filtering cannot remove it — see docs/experiments.md T10: the ghost is the
same model error repeated in every view, so multi-view corroboration accepts it,
and it is parallel to the true surface so the normal-aware filter cannot see it.

MLS takes the other route. Rather than deciding which sheet to delete, it fits a
local surface through each point's neighbourhood and projects the point onto it.
Both sheets collapse onto one, and where they are equally populated the result
lands between them — plausibly where the true surface is.

This MOVES points rather than removing them, so it is a genuine change to the
geometry, not a cleanup. Over-smoothing erases real detail, which is why the
neighbourhood radius is expressed in multiples of point spacing rather than as
an absolute distance.
"""
import numpy as np


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
