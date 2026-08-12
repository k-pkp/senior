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

    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < min_neighbors:
        return points, colors, {"moved_mm": 0.0}

    tree = cKDTree(pts)
    d, _ = tree.query(pts[np.random.default_rng(0).choice(
        n, min(5000, n), replace=False)], k=2, workers=-1)
    spacing = float(d[:, 1].mean())
    radius = spacing * radius_mult

    neighbours = tree.query_ball_point(pts, r=radius, workers=-1)

    out = pts.copy()
    moved = np.zeros(n)
    skipped = 0

    for i, nb in enumerate(neighbours):
        if len(nb) < min_neighbors:
            skipped += 1
            continue
        q = pts[nb]
        centroid = q.mean(axis=0)
        c = q - centroid

        # Local frame: normal is the least-variance direction.
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        normal = vt[2]
        u, v = vt[0], vt[1]

        # Coordinates in the tangent frame.
        a = c @ u
        b = c @ v
        h = c @ normal

        if polynomial and len(nb) >= 6:
            # h ~ c0 + c1*a + c2*b + c3*a² + c4*ab + c5*b²
            A = np.column_stack([np.ones_like(a), a, b, a * a, a * b, b * b])
            try:
                coef, *_ = np.linalg.lstsq(A, h, rcond=None)
            except np.linalg.LinAlgError:
                skipped += 1
                continue
        else:
            A = np.column_stack([np.ones_like(a), a, b])
            coef, *_ = np.linalg.lstsq(A, h, rcond=None)
            coef = np.concatenate([coef, np.zeros(3)])

        # Evaluate the fitted height at this point's own tangent coordinates.
        pa = float((pts[i] - centroid) @ u)
        pb = float((pts[i] - centroid) @ v)
        h_fit = (coef[0] + coef[1] * pa + coef[2] * pb
                 + coef[3] * pa * pa + coef[4] * pa * pb + coef[5] * pb * pb)

        target = centroid + pa * u + pb * v + h_fit * normal
        moved[i] = np.linalg.norm(target - pts[i])
        out[i] = target

    stats = {
        "spacing": spacing,
        "radius": radius,
        "median_move": float(np.median(moved)),
        "p95_move": float(np.percentile(moved, 95)),
        "skipped": skipped,
    }
    if verbose:
        print(f"  MLS: radius {radius:.5f} ({radius_mult:.1f}x spacing), "
              f"moved median {stats['median_move']:.5f}, "
              f"p95 {stats['p95_move']:.5f}, skipped {skipped:,}")
    return out.astype(np.float32), colors, stats
