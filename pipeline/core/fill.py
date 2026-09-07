import numpy as np
import open3d as o3d
import alphashape
from shapely.geometry import Point


def _plane_basis(normal):
    """Orthonormal (u, v) spanning the plane with the given normal."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, seed)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    return u, v, n


def cap_points_on_plane(points, colors_u8, plane_point, plane_normal,
                        spacing=None, alpha=2.0, slab_mult=3.0, label=""):
    """Fill the flat cross-section left by a marker cut with synthetic points.

    The marker cut slices the limb on an arbitrarily oriented plane, leaving an
    open elliptical boundary. Left open, the surface solver rounds it off and
    PyMeshFix closes it with whatever shape it likes, so the cut face is neither
    flat nor volumetrically correct.

    The cap grid is built in the plane's own (u, v) basis derived from the
    marker normal — not an axis-aligned one — so it stays coplanar with the cut
    however the marker is oriented.
    """
    points = np.asarray(points, dtype=np.float64)
    tag = f"[{label}] " if label else ""
    if len(points) < 20:
        return points, colors_u8

    u, v, n = _plane_basis(plane_normal)
    c = np.asarray(plane_point, dtype=np.float64)

    if spacing is None:
        from scipy.spatial import cKDTree
        sample = points if len(points) <= 5000 else points[
            np.random.default_rng(42).choice(len(points), 5000, replace=False)]
        d, _ = cKDTree(sample).query(sample, k=2)
        spacing = float(np.clip(d[:, 1].mean(), 0.0005, 0.005))

    # Boundary ring: points lying within a thin slab of the cut plane.
    dist = (points - c) @ n
    band = np.abs(dist) <= slab_mult * spacing
    if band.sum() < 10:
        print(f"  -> {tag}cut-plane cap skipped ({int(band.sum())} boundary pts)")
        return points, colors_u8

    ring = points[band]
    xy = np.column_stack([(ring - c) @ u, (ring - c) @ v])

    poly = _footprint_polygon(xy, alpha, label)
    if poly is None:
        print(f"  -> {tag}cut-plane cap skipped (empty hull)")
        return points, colors_u8

    min_x, min_y, max_x, max_y = poly.bounds
    gx = np.arange(min_x, max_x + spacing, spacing)
    gy = np.arange(min_y, max_y + spacing, spacing)
    grid = np.c_[np.repeat(gx, len(gy)), np.tile(gy, len(gx))]
    inside = np.array([poly.contains(Point(p)) for p in grid])
    if not inside.any():
        print(f"  -> {tag}cut-plane cap skipped (no interior grid points)")
        return points, colors_u8

    g2 = grid[inside]
    cap = c + g2[:, 0:1] * u + g2[:, 1:2] * v

    out_pts = np.vstack([points, cap])
    out_cols = colors_u8
    if colors_u8 is not None and len(colors_u8) == len(points):
        ring_cols = colors_u8[band]
        avg = ring_cols.mean(axis=0) if len(ring_cols) else np.array([128, 128, 128])
        out_cols = np.vstack([colors_u8,
                              np.tile(avg.astype(colors_u8.dtype), (len(cap), 1))])

    print(f"  -> {tag}cut-plane cap: +{len(cap):,} pts from {int(band.sum()):,} "
          f"boundary pts (spacing {spacing:.4f})")
    return out_pts, out_cols


def _footprint_polygon(xy, alpha, label=""):
    """Outline of a 2D point set, guaranteed sane or None.

    alphashape is not safe to trust blindly. On input with no interior points —
    a hollow ring, which is exactly what a one-layer extruded wall gives it — the
    Delaunay step hits a singular matrix and returns a polygon whose coordinates
    are nan or astronomically large, with no exception raised. Handing those
    bounds to a grid generator asks for an array of 8e17 cells.

    Every result is therefore checked against the data it came from: a real
    footprint lies inside the points' own bounding box. Anything else falls back
    to the convex hull, which is exact and cannot degenerate.
    """
    tag = f"[{label}] " if label else ""
    xy = np.unique(np.asarray(xy, dtype=np.float64), axis=0)
    if len(xy) < 4:
        return None
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = hi - lo
    if not np.all(span > 0):
        return None

    pad = 0.02 * span.max()

    def usable(poly):
        """Returns True if the polygon is a finite, non-empty Polygon inside the padded bounds."""
        if poly is None or poly.is_empty or poly.geom_type != "Polygon":
            return False
        b = np.asarray(poly.bounds, dtype=np.float64)
        if not np.all(np.isfinite(b)) or poly.area <= 0:
            return False
        # A hull of these points cannot escape their bounding box.
        if not (np.all(b[:2] >= lo - pad) and np.all(b[2:] <= hi + pad)):
            return False
        # A sliver has finite bounds and positive area but encloses nothing. It
        # survives every check above and then divides into the grid spacing,
        # so test the one thing a footprint must do: contain its own points.
        from shapely import contains_xy
        return bool(contains_xy(poly.buffer(pad), xy[:, 0], xy[:, 1]).mean() >= 0.9)

    poly = None
    if alpha > 0:
        try:
            poly = alphashape.alphashape(xy, alpha)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda a: a.area)
        except Exception:
            poly = None
        if not usable(poly):
            print(f"  -> {tag}alpha outline degenerate — using convex hull")
            poly = None

    if poly is None:
        from shapely.geometry import MultiPoint
        poly = MultiPoint([tuple(p) for p in xy]).convex_hull
        if not usable(poly):
            return None
    return poly


def _sample_outline(poly, spacing, height, rng, max_pts=60000):
    """Random points on the vertical surface swept by a footprint's boundary.

    Sampled at random rather than on a ring-by-level lattice, and that is the
    whole point. Alpha shape reaches the surface through a Delaunay
    tetrahedralisation, and a regular lattice is cospherical in bulk — the
    degenerate case. Open3D reports it as "invalid tetra" and returns a shredded
    surface that never closes; a regular solid lattice measured Euler 80,773
    where the same geometry sampled at random measured 2. Real scan points are
    irregular, which is why copying them worked at all.

    Returns (N, 2) positions along the boundary paired with (N,) heights in
    [0, height], to be offset by the caller.
    """
    rings = [poly.exterior] + list(poly.interiors)
    lengths = np.array([r.length for r in rings], dtype=float)
    total = float(lengths.sum())
    if total <= 0 or height <= 0:
        return np.empty((0, 2)), np.empty(0)

    n = int(np.clip(round(total * height / spacing ** 2), 32, max_pts))
    share = np.maximum((lengths / total * n).astype(int), 4)
    xy = np.vstack([
        np.array([r.interpolate(u, normalized=True).coords[0]
                  for u in rng.random(k)])
        for r, k in zip(rings, share)])
    return xy, rng.uniform(0.0, height, len(xy))


def extend_point_cloud_to_floor(pcd, floor_z, alpha=2.0, max_gap_frac=0.35,
                                label="", seed=42):
    """Extend an object's side walls down to the detected floor plane.

    VGGT does not reconstruct the shadowed base where an object meets the
    ground, so the cluster floats above the floor and every downstream height
    and volume reads short. The missing wall is rebuilt as a vertical extrusion
    of the object's own bottom silhouette, one layer thick and sampled at the
    cloud's own density — a surface, which is what the solver in Stage 4 is
    entitled to assume it is being given.

    This replaced copying the whole bottom band down at four discrete offsets.
    Those copies overlapped into a solid roughly seven times denser than real
    scanned surface, half the leg's entire cloud was fabricated, and neither
    Poisson nor ball pivoting could seat a base on it.

    The silhouette is taken from a thin rim near z_min rather than a thick slab,
    because on a tapering object a slab's outline is the cross-section some way
    up, not the one that actually meets the floor.

    No-op when the gap is below one point spacing (nothing missing) or above
    max_gap_frac of the object height (floor detection is untrustworthy).
    """
    points = np.asarray(pcd.points)
    if len(points) < 10 or floor_z is None:
        return pcd

    z_min = float(points[:, 2].min())
    height = float(points[:, 2].max()) - z_min
    gap = z_min - floor_z
    tag = f"[{label}] " if label else ""

    spacing = float(np.clip(np.mean(pcd.compute_nearest_neighbor_distance()),
                            0.0005, 0.005))
    if gap <= spacing:
        print(f"  -> {tag}already at floor (gap {gap:.4f} <= spacing {spacing:.4f})")
        return pcd
    if height > 0 and gap > max_gap_frac * height:
        print(f"  -> {tag}floor gap {gap:.4f} > {max_gap_frac:.0%} of height "
              f"{height:.4f} — skipping extension (floor suspect)")
        return pcd

    # Thinnest rim that still carries a usable outline. Widening in steps beats
    # one fixed fraction: a dense cloud gets a true bottom silhouette, a sparse
    # one still gets something rather than being skipped.
    band = None
    for mult in (3.0, 6.0, 12.0, 25.0, 50.0):
        thick = max(mult * spacing, 0.005 * height)
        sel = points[:, 2] <= z_min + thick
        if int(sel.sum()) >= 40:
            band = points[sel]
            break
    if band is None:
        print(f"  -> {tag}bottom rim too sparse — skipping extension")
        return pcd

    poly = _footprint_polygon(band[:, :2], alpha, label)
    if poly is None:
        print(f"  -> {tag}no bottom outline — skipping extension")
        return pcd

    rng = np.random.default_rng(seed)
    xy, h = _sample_outline(poly, spacing, gap, rng)
    if len(xy) == 0:
        print(f"  -> {tag}empty wall sample — skipping extension")
        return pcd
    wall = np.column_stack([xy, floor_z + h])

    ext = o3d.geometry.PointCloud()
    ext.points = o3d.utility.Vector3dVector(wall)
    if pcd.has_colors():
        from scipy.spatial import cKDTree
        cols = np.asarray(pcd.colors)[points[:, 2] <= z_min + thick]
        _, near = cKDTree(band[:, :2]).query(xy, k=1, workers=-1)
        ext.colors = o3d.utility.Vector3dVector(cols[near])

    print(f"  -> {tag}extended to floor: gap {gap:.4f}, +{len(wall):,} wall pts "
          f"at spacing {spacing:.4f} (outline from {len(band):,} pts "
          f"within {thick:.4f})")
    return pcd + ext


def cap_point_cloud_bottom(pcd, alpha=2.0, z_offset=0.0, slice_thickness=0.01):
    """
    Creates a synthetic flat bottom cap by matching the exact point density
    of the original scan, preventing the reconstruction filter from deleting it.
    """
    print("  -> Extracting coordinates...")
    points = np.asarray(pcd.points)

    if len(pcd.normals) == 0:
        print("  -> Normals not found. Estimating outward normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)

    print("  -> Analyzing scan density...")
    distances = pcd.compute_nearest_neighbor_distance()
    avg_spacing = np.mean(distances)

    scan_spacing = float(np.clip(avg_spacing, 0.0005, 0.005))

    z_min = np.min(points[:, 2])
    target_z = z_min + z_offset

    print(f"  -> Extracting bottom cross-section (thickness: {slice_thickness} units)...")
    bottom_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_min + slice_thickness)
    bottom_points = points[bottom_mask]

    if len(bottom_points) < 10:
        print("  -> Warning: Too few points in slice. Using full object footprint instead.")
        points_2d = points[:, :2]
        bottom_mask = np.ones(len(points), dtype=bool)
    else:
        points_2d = bottom_points[:, :2]

    print("  -> Calculating boundary hull...")
    hull_polygon = _footprint_polygon(points_2d, alpha, "cap")
    if hull_polygon is None:
        print("  -> Warning: no usable bottom outline — cap skipped.")
        return pcd

    # Grid spacing depends on the hull's area, so it can only be chosen now.
    from pipeline.config import CAP_SPACING_MULT, CAP_MIN_PTS, CAP_MAX_PTS
    area = float(hull_polygon.area)
    point_spacing = scan_spacing * CAP_SPACING_MULT
    if area > 0:
        n_est = area / (point_spacing ** 2)
        if n_est > CAP_MAX_PTS:
            point_spacing = np.sqrt(area / CAP_MAX_PTS)
        elif n_est < CAP_MIN_PTS:
            point_spacing = np.sqrt(area / CAP_MIN_PTS)
    print(f"  -> Scan spacing {scan_spacing:.5f}, hull area {area:.5f}, "
          f"cap grid {point_spacing:.5f} ({point_spacing/scan_spacing:.1f}x scan)")

    print("  -> Generating uniform point grid inside the outline...")
    min_x, min_y, max_x, max_y = hull_polygon.bounds
    # Belt and braces: the hull is validated, but the spacing is derived from an
    # area, so cap the cell count directly rather than trusting the arithmetic.
    cells = ((max_x - min_x) / point_spacing) * ((max_y - min_y) / point_spacing)
    if cells > 4 * CAP_MAX_PTS:
        point_spacing *= np.sqrt(cells / (4 * CAP_MAX_PTS))
        print(f"  -> Grid would be {cells:,.0f} cells; coarsened to "
              f"{point_spacing:.5f}")
    x_grid = np.arange(min_x, max_x, point_spacing)
    y_grid = np.arange(min_y, max_y, point_spacing)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]

    # Vectorised point-in-polygon. The per-point Python loop this replaces made
    # one shapely call per grid cell, which at scan resolution was >100k calls.
    try:
        from shapely import contains_xy
        keep = contains_xy(hull_polygon, grid_points_2d[:, 0], grid_points_2d[:, 1])
        synthetic_points_2d = grid_points_2d[keep]
    except ImportError:
        synthetic_points_2d = np.array(
            [pt for pt in grid_points_2d if hull_polygon.contains(Point(pt))])
    # Break the lattice. A regular grid is cocircular in bulk, which is the
    # degenerate case for the Delaunay step behind alpha shape; jitter well
    # under half a cell keeps the coverage and loses the degeneracy. Only x and
    # y move, so the cap stays exactly flat and the volume it closes is honest.
    if len(synthetic_points_2d):
        synthetic_points_2d = synthetic_points_2d + np.random.default_rng(
            42).normal(0.0, 0.25 * point_spacing, synthetic_points_2d.shape)
    num_synthetic_points = len(synthetic_points_2d)
    print(f"  -> Generated {num_synthetic_points} synthetic points for the cap.")

    z_array = np.full((num_synthetic_points, 1), target_z)
    synthetic_points_3d = np.hstack((synthetic_points_2d, z_array))

    synthetic_normals = np.tile([0.0, 0.0, -1.0], (num_synthetic_points, 1))

    cap_pcd = o3d.geometry.PointCloud()
    cap_pcd.points = o3d.utility.Vector3dVector(synthetic_points_3d)
    cap_pcd.normals = o3d.utility.Vector3dVector(synthetic_normals)

    if pcd.has_colors():
        print("  -> Preserving colors and painting the cap...")
        colors = np.asarray(pcd.colors)
        bottom_colors = colors[bottom_mask]
        if len(bottom_colors) > 0:
            avg_color = np.mean(bottom_colors, axis=0)
        else:
            avg_color = [0.5, 0.5, 0.5]
        synthetic_colors = np.tile(avg_color, (num_synthetic_points, 1))
        cap_pcd.colors = o3d.utility.Vector3dVector(synthetic_colors)

    merged_pcd = pcd + cap_pcd
    return merged_pcd
