"""Marker-based leg surface segmentation using HSV + Excess Green color detection.

Steps:
    1. Detect marker points via HSV color thresholding + Excess Green Index.
    2. Cluster markers spatially with DBSCAN.
    3. Fit a plane to each marker cluster via SVD to capture tilt/slant.
    4. Cut the point cloud using signed distance to the tilted marker planes.

Axis convention: After pipeline leveling (Stage 3), Z is the vertical axis.
Use ``height_axis="z"`` for post-leveled point clouds.
"""

import numpy as np
from sklearn.cluster import DBSCAN

AXIS_MAP = {"x": 0, "y": 1, "z": 2}


def rgb_to_hsv(r, g, b):
    """Vectorized RGB [0-255] -> HSV (H in degrees, S/V in %)."""
    r_n = r.astype(np.float32) / 255.0
    g_n = g.astype(np.float32) / 255.0
    b_n = b.astype(np.float32) / 255.0

    mx = np.maximum(np.maximum(r_n, g_n), b_n)
    mn = np.minimum(np.minimum(r_n, g_n), b_n)
    delta = mx - mn

    h = np.zeros_like(mx)
    mask_r = (mx == r_n) & (delta > 1e-9)
    mask_g = (mx == g_n) & (delta > 1e-9)
    mask_b = (mx == b_n) & (delta > 1e-9)

    h[mask_r] = (60.0 * ((g_n[mask_r] - b_n[mask_r]) / delta[mask_r]) + 360) % 360
    h[mask_g] = (60.0 * ((b_n[mask_g] - r_n[mask_g]) / delta[mask_g]) + 120) % 360
    h[mask_b] = (60.0 * ((r_n[mask_b] - g_n[mask_b]) / delta[mask_b]) + 240) % 360

    s = np.zeros_like(mx)
    s[mx > 1e-9] = (delta[mx > 1e-9] / mx[mx > 1e-9]) * 100.0

    v = mx * 100.0
    return h, s, v


def marker_mask_by_contrast(colors_uint8, band_rgb, limb_rgb,
                            threshold=0.5, val_floor=None):
    """Separate the band from the limb by what actually distinguishes them.

    Centring a colour window on the measured band was wrong, and measurably so:
    this khaki band sits at hue 26 while skin sits at 11-20, so a window around
    the band admitted the entire limb -- 102,988 points selected where the real
    band is 210. The config's hand-tuned window works not because it describes
    the band but because it EXCLUDES skin, and leaves excess green to do the
    separating (+14 against -54).

    Generalising that means learning the contrast rather than the colour. Band
    and limb are two measured points in RGB; the line between them is the
    direction along which they differ, and everything else is irrelevant. Each
    pixel is projected onto that line, 0 at the limb and 1 at the band, so the
    test is "closer to the band than to the limb" and carries no assumption
    about which colour either one is.
    """
    def _chroma(x):
        """Colour with brightness divided out.

        Working in raw RGB fails on this data for a reason worth stating: the
        band is darker than skin as well as differently coloured, so the axis
        between them points partly along brightness -- and shadowed skin then
        scores as band. Selecting 3,244 points where the band is ~210 is that
        error. Normalising by intensity keeps only the chromatic difference,
        which is what actually distinguishes a marker from the limb under it,
        and is why the hand-tuned rule leans on excess green rather than value.
        """
        x = np.asarray(x, dtype=np.float64)
        total = x.sum(axis=-1, keepdims=True)
        return x / np.maximum(total, 1e-6)

    band = _chroma(band_rgb)
    limb = _chroma(limb_rgb)
    axis = band - limb
    denom = float(axis @ axis)
    if denom < 1e-9:
        return np.zeros(len(colors_uint8), dtype=bool), {"n_markers": 0,
                                                         "degenerate": True}
    rgb = np.asarray(colors_uint8, dtype=np.float64)
    score = ((_chroma(rgb) - limb) @ axis) / denom
    mask = score > threshold
    if val_floor is not None:
        mask &= rgb.max(axis=1) > val_floor
    return mask, {
        "n_markers": int(mask.sum()),
        "separation": round(float(np.sqrt(denom)), 4),
        "threshold": threshold,
    }


def detect_markers(colors_uint8, hue_min=None, hue_max=None, sat_min=None,
                   val_min=None, exg_min=None):
    """Detect green marker points by hue window and Excess Green Index.

    Rules:
        val > VAL_MIN AND sat > SAT_MIN AND HUE_MIN < hue < HUE_MAX   -- HSV
        val > VAL_MIN AND (2*G - R - B) > EXG_MIN                     -- ExG

    Both rules carry the brightness floor. Hue is computed as a ratio of
    channel differences to the channel maximum, so as a pixel darkens the hue
    becomes arbitrary — an RGB(8,6,8) shadow reports a confident hue that means
    nothing. Without the floor those pixels dominate the detection; see
    config.MARKER_VAL_MIN.

    The previous rule was `sat > 15 AND hue > 60`, an upper-open hue test that
    admitted everything but red/orange/yellow. Skin at hue 358 passed it.

    Args:
        colors_uint8: (N, 3) numpy array of RGB colors [0-255].
        hue_min/hue_max/sat_min/val_min/exg_min: override the config defaults.

    Returns:
        marker_mask: boolean (N,) array, True where markers are detected.
        stats: dict with diagnostic counts and colour statistics.
    """
    from pipeline import config as _cfg

    hue_min = _cfg.MARKER_HUE_MIN if hue_min is None else hue_min
    hue_max = _cfg.MARKER_HUE_MAX if hue_max is None else hue_max
    sat_min = _cfg.MARKER_SAT_MIN if sat_min is None else sat_min
    val_min = _cfg.MARKER_VAL_MIN if val_min is None else val_min
    exg_min = _cfg.MARKER_EXG_MIN if exg_min is None else exg_min

    r = colors_uint8[:, 0].astype(np.int32)
    g = colors_uint8[:, 1].astype(np.int32)
    b = colors_uint8[:, 2].astype(np.int32)

    h, s, v = rgb_to_hsv(r, g, b)

    bright = v > val_min
    hsv_mask = bright & (s > sat_min) & (h > hue_min) & (h < hue_max)

    exg = 2 * g - r - b
    exg_mask = bright & (exg > exg_min)

    marker_mask = hsv_mask | exg_mask

    stats = {
        "n_total": len(colors_uint8),
        "n_hsv": int(hsv_mask.sum()),
        "n_exg": int(exg_mask.sum()),
        "n_markers": int(marker_mask.sum()),
        "hsv_only": int((hsv_mask & ~exg_mask).sum()),
        "exg_only": int((exg_mask & ~hsv_mask).sum()),
        "both": int((hsv_mask & exg_mask).sum()),
    }

    if marker_mask.sum() > 0:
        stats["h_mean"] = float(h[hsv_mask].mean()) if hsv_mask.sum() > 0 else 0
        stats["s_mean"] = float(s[hsv_mask].mean()) if hsv_mask.sum() > 0 else 0
        stats["exg_mean"] = float(exg[exg_mask].mean()) if exg_mask.sum() > 0 else 0
        stats["mr"], stats["mg"], stats["mb"] = (
            float(r[marker_mask].mean()),
            float(g[marker_mask].mean()),
            float(b[marker_mask].mean()),
        )

    return marker_mask, stats


def cluster_markers(coords, eps=0.03, min_samples=10):
    """Spatial DBSCAN clustering of marker points.

    Args:
        coords: (M, 3) numpy array of marker point coordinates.
        eps: DBSCAN epsilon (default 0.03 ~ 3 cm).
        min_samples: Minimum samples per cluster (default 10).

    Returns:
        labels: cluster label for each point (-1 = noise).
        n_clusters: number of valid clusters (excluding noise).
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters


def compute_cluster_planes(coords, labels, colors_uint8, axis_idx,
                           min_cluster_size=None):
    """Compute per-cluster centroid and SVD-based plane normal.

    For each marker cluster, fits a best-fit plane via Singular Value
    Decomposition.  The normal vector captures the tilt/slant of the
    marker surface.

    Returns list of (cluster_id, centroid, normal, n_points, color_mean)
    sorted by centroid position along the height axis ascending.
    Noise (label == -1) and clusters smaller than min_cluster_size are
    excluded.
    """
    if min_cluster_size is None:
        from pipeline import config as _cfg
        min_cluster_size = _cfg.MARKER_MIN_CLUSTER_PTS

    planes = []
    skipped = 0
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        mask = labels == cid
        npts = mask.sum()
        if npts < min_cluster_size:
            skipped += 1
            continue
        cluster_coords = coords[mask]
        centroid = cluster_coords.mean(axis=0)

        centered = cluster_coords - centroid
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        normal = Vt[-1]

        if normal[axis_idx] < 0:
            normal = -normal

        color_mean = colors_uint8[mask].mean(axis=0)
        planes.append((cid, centroid, normal, npts, color_mean))

    if skipped:
        print(f"    (skipped {skipped} cluster(s) with < {min_cluster_size} points)")

    planes.sort(key=lambda x: x[1][axis_idx])
    return planes


def cut_surface_plane(coords, planes, axis_idx, axis_name="Z"):
    """Create keep-mask based on signed distance to tilted marker planes.

    Uses the SVD-fitted plane for each marker cluster so that slanted
    and tilted markers are handled correctly.

    For point P and plane (centroid C, unit normal N):
        signed_distance(P) = dot(P - C, N)

    Normal is oriented so that positive N[axis_idx] points "upward".
    Therefore:
        distance < 0  ->  below the plane
        distance > 0  ->  above the plane

    Case logic:
        0 markers -> keep all points
        1 marker  -> keep points *below* the tilted plane (distance < 0)
        2 markers -> keep points *above* bottom plane (dist > 0) AND
                     *below* top plane (dist < 0)
        3+ markers -> keep points between lowest and highest marker planes

    Returns boolean mask of points to keep.
    """
    n = len(planes)

    if n == 0:
        return np.ones(len(coords), dtype=bool), {"case": 0, "kept": len(coords),
                                                    "total": len(coords)}

    if n == 1:
        cid, centroid, normal, npts, clr = planes[0]

        heights = coords[:, axis_idx]
        min_h = heights.min()
        max_h = heights.max()
        threshold = min_h + 0.3 * (max_h - min_h)

        if centroid[axis_idx] < threshold:
            return np.ones(len(coords), dtype=bool), {
                "case": 0,
                "reason": "marker_below_30pct",
                "marker_height": float(centroid[axis_idx]),
                "threshold": float(threshold),
                "kept": len(coords),
                "total": len(coords),
            }

        dist = np.dot(coords - centroid, normal)
        keep = dist < 0
        info = {
            "case": 1,
            "centroid": centroid.tolist(),
            "normal": normal.tolist(),
        }

    elif n == 2:
        cid_low, centroid_low, normal_low, npts_low, clr_low = planes[0]
        cid_high, centroid_high, normal_high, npts_high, clr_high = planes[1]

        dist_low = np.dot(coords - centroid_low, normal_low)
        dist_high = np.dot(coords - centroid_high, normal_high)

        keep = (dist_low > 0) & (dist_high < 0)
        info = {
            "case": 2,
            "centroid_low": centroid_low.tolist(),
            "normal_low": normal_low.tolist(),
            "centroid_high": centroid_high.tolist(),
            "normal_high": normal_high.tolist(),
        }

    else:
        cid_low, centroid_low, normal_low, npts_low, clr_low = planes[0]
        cid_high, centroid_high, normal_high, npts_high, clr_high = planes[-1]

        dist_low = np.dot(coords - centroid_low, normal_low)
        dist_high = np.dot(coords - centroid_high, normal_high)

        keep = (dist_low > 0) & (dist_high < 0)
        info = {
            "case": 3,
            "n_markers": n,
            "centroid_low": centroid_low.tolist(),
            "normal_low": normal_low.tolist(),
            "centroid_high": centroid_high.tolist(),
            "normal_high": normal_high.tolist(),
        }

    info["kept"] = int(keep.sum())
    info["total"] = len(coords)
    return keep, info


def segment_point_cloud(pcd, height_axis="z", verbose=True,
                        marker_colour=None):
    """Run full marker-based leg segmentation on an Open3D point cloud.

    Marker planes are fitted via SVD so that slanted/tilted markers
    are handled by cutting with signed distance to the tilted plane.

    Steps:
        1. HSV + ExG color detection (same universal marker logic).
        2. DBSCAN spatial clustering.
        3. SVD plane fitting per cluster (centroid + normal vector).
        4. Signed-distance cut using the tilted marker planes.

    Args:
        pcd: Open3D PointCloud with colors.
        height_axis: axis used for normal orientation and sorting ("x","y","z").
                     Normal is flipped to point positively along this axis,
                     so that "below plane" = negative signed distance.
                     Default "z" (vertical after pipeline leveling).
        verbose: print progress messages.

    Returns:
        segmented_pcd: Open3D PointCloud containing only the kept points.
        summary: dict with detection/clustering/cut statistics.
    """
    import open3d as o3d

    axis_idx = AXIS_MAP[height_axis.lower()]
    axis_name = height_axis.upper()

    coords = np.asarray(pcd.points, dtype=np.float64)
    if not pcd.has_colors():
        raise ValueError("Point cloud has no color information")

    colors_float = np.asarray(pcd.colors, dtype=np.float64)
    colors = np.clip(colors_float * 255.0, 0, 255).astype(np.uint8)

    n_total = len(coords)

    # --- Step 1: Color Detection ---
    if verbose:
        print(f"  Points: {n_total:,}  "
              f"BBox Z[{coords[:,2].min():.3f},{coords[:,2].max():.3f}]")
        print("  Detecting markers (HSV S>15 & H>60, ExG>10)...")

    # Stage 0 measures the marker's colour when it locates the band, so prefer
    # that over the config defaults, which describe one particular khaki band.
    # Stage 0 measures both the band and the limb it sits on when it locates
    # the band, so prefer the contrast between them over the config defaults,
    # which describe one particular khaki band against one particular skin.
    marker_mask = stats = None
    if marker_colour and marker_colour.get("rgb") and marker_colour.get("limb_rgb"):
        from pipeline import config as _cfg
        marker_mask, stats = marker_mask_by_contrast(
            colors, marker_colour["rgb"], marker_colour["limb_rgb"],
            val_floor=_cfg.MARKER_VAL_MIN * 255.0 / 100.0)
        if verbose:
            print(f"    marker by contrast: band RGB "
                  f"{[int(v) for v in marker_colour['rgb']]} vs limb "
                  f"{[int(v) for v in marker_colour['limb_rgb']]}, "
                  f"separation {stats.get('separation')} -> "
                  f"{stats['n_markers']:,} points")
    if marker_mask is None:
        marker_mask, stats = detect_markers(colors)

    if verbose:
        if "n_hsv" not in stats:
            print(f"    markers: {stats['n_markers']:,} "
                  f"({stats['n_markers']/n_total*100:.2f}%)")
        else:
            print(f"    HSV-only: {stats['n_hsv']:,}  ExG-only: {stats['exg_only']:,}  "
              f"both: {stats['both']:,}  total: {stats['n_markers']:,} "
              f"({stats['n_markers']/n_total*100:.1f}%)")

    summary = {"detection": stats, "n_clusters": 0, "cut": {}}

    # --- Too few markers: keep all ---
    if marker_mask.sum() < 10:
        if verbose:
            print("  Too few marker points (< 10) — keeping all points")
        segmented_pcd = pcd
        summary["cut"] = {"case": 0, "reason": "too_few_markers"}
        summary["n_kept"] = n_total
        return segmented_pcd, summary

    marker_coords = coords[marker_mask]
    marker_colors = colors[marker_mask]

    # --- Step 2: Spatial Clustering ---
    if verbose:
        print(f"  Clustering {len(marker_coords):,} marker points (DBSCAN eps=0.03)...")

    labels, n_clusters = cluster_markers(marker_coords)
    summary["n_clusters"] = n_clusters

    if verbose:
        n_noise = (labels == -1).sum()
        print(f"    {n_clusters} clusters, {n_noise} noise points")

    # --- All noise: keep all ---
    valid = labels != -1
    if not np.any(valid):
        if verbose:
            print("  All markers classified as noise — keeping all points")
        segmented_pcd = pcd
        summary["cut"] = {"case": 0, "reason": "all_noise"}
        summary["n_kept"] = n_total
        return segmented_pcd, summary

    # --- Step 3: SVD plane fitting per cluster ---
    planes = compute_cluster_planes(marker_coords, labels, marker_colors, axis_idx)

    if verbose:
        print(f"  Marker planes (sorted by {axis_name} centroid):")
        for cid, centroid, normal, npts, color_mean in planes:
            print(f"    cluster {cid}: centroid=({centroid[0]:.4f},{centroid[1]:.4f},"
                  f"{centroid[2]:.4f})  normal=({normal[0]:+.4f},{normal[1]:+.4f},"
                  f"{normal[2]:+.4f})  {npts} pts  "
                  f"RGB=({color_mean[0]:.0f},{color_mean[1]:.0f},{color_mean[2]:.0f})")

    # --- Step 4: Cut using signed distance to tilted planes ---
    if verbose:
        print("  Cutting surface via plane signed-distance...")

    keep_mask, cut_info = cut_surface_plane(coords, planes, axis_idx, axis_name)
    summary["cut"] = cut_info
    summary["planes"] = planes

    if verbose:
        kept = cut_info["kept"]
        print(f"    Kept {kept:,} / {n_total:,} ({kept/n_total*100:.1f}%) "
              f"({cut_info.get('case', '?')} marker(s))")

    summary["n_kept"] = cut_info["kept"]

    filtered_coords = coords[keep_mask]
    filtered_colors = colors[keep_mask]

    segmented_pcd = o3d.geometry.PointCloud()
    segmented_pcd.points = o3d.utility.Vector3dVector(filtered_coords)
    segmented_pcd.colors = o3d.utility.Vector3dVector(filtered_colors.astype(np.float64) / 255.0)

    return segmented_pcd, summary


MAX_MARKERS = 2


def apply_marker_cut(points, markers, up=(0.0, 0.0, 1.0)):
    """Cut the limb against at most two marker planes, by height.

    Expects `points` in levelled space, where `up` is the vertical axis.

    The rule is stated in terms of above/below rather than sides:

        0 markers -> no cut
        1 marker  -> keep what is BELOW the plane
        2 markers -> keep what is BETWEEN the two planes

    This replaces the earlier centroid-side rule, which decided each side from
    where the cloud's centroid happened to fall. That was fine for a one-shot
    run but had three defects: dragging a plane past the centroid inverted the
    whole selection; two planes only meant "keep between" when the centroid
    already sat between them; and the user could not overrule a centroid that
    landed on the wrong side of a lopsided cloud. Height is a property of the
    scene, not of the cloud's mass distribution, so none of that applies here.

    Markers beyond the first two are dropped: a limb segment is bounded by at
    most two cuts, and a third plane can only contradict one of them.

    Args:
        points: (N, 3) in levelled space.
        markers: list of {"centroid": (3,), "normal": (3,)}. Normals may point
            either way — each is flipped to point along `up` first, so the
            sign of the detected normal cannot change the outcome.
        up: unit vertical axis of `points`.

    Returns:
        (keep_mask, case_label)
    """
    if len(markers) == 0:
        return np.ones(len(points), dtype=bool), "no_markers"

    up = np.asarray(up, dtype=np.float64)
    up = up / (np.linalg.norm(up) + 1e-8)
    pts = points.astype(np.float64)

    planes = []
    for m in markers[:MAX_MARKERS]:
        cen = np.asarray(m["centroid"], dtype=np.float64)
        nrm = np.asarray(m["normal"], dtype=np.float64)
        nrm = nrm / (np.linalg.norm(nrm) + 1e-8)
        vert = float(np.dot(nrm, up))
        if abs(vert) < 1e-3:
            # Plane stands vertical, so it has no above or below and the rule
            # is undefined. Skipping beats guessing a side.
            continue
        if vert < 0:
            nrm = -nrm
        # Signed height of every point relative to this plane, along its own
        # (now upward) normal. Negative is below.
        planes.append(pts @ nrm - float(np.dot(cen, nrm)))

    if not planes:
        return np.ones(len(points), dtype=bool), "no_usable_markers"

    if len(planes) == 1:
        keep = planes[0] <= 0.0
        case = "case_1_below"
    else:
        # Which plane is upper is decided per point by its own signed distance,
        # so keeping "below one and above the other" needs no ordering: a point
        # between them is below the upper and above the lower, giving one
        # negative and one positive distance.
        a, b = planes
        keep = (a <= 0.0) != (b <= 0.0)
        case = "case_2_between"

    if int(keep.sum()) < 50:
        return np.ones(len(points), dtype=bool), "too_few_after_cut"

    return keep, case
