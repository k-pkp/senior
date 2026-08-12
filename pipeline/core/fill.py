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

    try:
        poly = alphashape.alphashape(xy, alpha)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda a: a.area)
    except Exception:
        poly = alphashape.alphashape(xy, 0.0)
    if poly.is_empty or poly.area <= 0:
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


def extend_point_cloud_to_floor(pcd, floor_z, band_frac=0.10, max_gap_frac=0.35,
                                label=""):
    """Extend an object's side walls down to the detected floor plane.

    VGGT does not reconstruct the shadowed base where an object meets the
    ground, so the cluster floats above the floor and every downstream height
    and volume reads short. The bottom band is swept down to floor_z at the
    cloud's own point spacing, which restores the missing wall before
    cap_point_cloud_bottom() closes it.

    No-op when the gap is below one point spacing (nothing missing) or above
    max_gap_frac of the object height (floor detection is untrustworthy).
    """
    points = np.asarray(pcd.points)
    if len(points) < 10 or floor_z is None:
        return pcd

    z_min = float(points[:, 2].min())
    z_max = float(points[:, 2].max())
    height = z_max - z_min
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

    band = points[points[:, 2] <= z_min + band_frac * height]
    if len(band) < 10:
        print(f"  -> {tag}bottom band too sparse ({len(band)}) — skipping extension")
        return pcd

    n_levels = int(np.ceil(gap / spacing))
    offsets = np.linspace(gap, 0.0, n_levels, endpoint=False)
    tiled = np.repeat(band[:, None, :], n_levels, axis=1).reshape(-1, 3)
    tiled[:, 2] -= np.tile(offsets, len(band))

    ext = o3d.geometry.PointCloud()
    ext.points = o3d.utility.Vector3dVector(tiled)
    if pcd.has_colors():
        cols = np.asarray(pcd.colors)[points[:, 2] <= z_min + band_frac * height]
        ext.colors = o3d.utility.Vector3dVector(
            np.repeat(cols[:, None, :], n_levels, axis=1).reshape(-1, 3))

    print(f"  -> {tag}extended to floor: gap {gap:.4f} ({n_levels} levels), "
          f"+{len(tiled):,} pts from {len(band):,} band pts")
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
    try:
        hull_polygon = alphashape.alphashape(points_2d, alpha)

        if hull_polygon.geom_type == 'MultiPolygon':
            hull_polygon = max(hull_polygon.geoms, key=lambda a: a.area)
    except Exception:
        print(f"  -> Warning: Hull generation failed with alpha={alpha}. Defaulting to convex hull.")
        hull_polygon = alphashape.alphashape(points_2d, 0.0)

    # Grid spacing depends on the hull's area, so it can only be chosen now.
    from pipeline.config import CAP_SPACING_MULT, CAP_MIN_PTS, CAP_MAX_PTS
    area = float(getattr(hull_polygon, "area", 0.0))
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
