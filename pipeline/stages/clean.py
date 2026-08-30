"""Stage 3 — segment the scene, cut the limb at its marker, close and export.

Clusters the dense cloud once, detects marker cut planes on the dense limb,
then ghost-filters each identified cluster separately.
"""
import os

import numpy as np
import open3d as o3d

from pipeline.core.plane import (
    auto_ransac_threshold,
    detect_plane_ransac_deterministic,
    get_rotation_to_z_axis,
    remove_dominant_plane,
)
from pipeline.core.cluster import detect_top_k_objects
from pipeline.core.segmentation import (segment_point_cloud, apply_marker_cut,
                                        MAX_MARKERS, AXIS_MAP)
from pipeline.config import (MARKER_MIN_HEIGHT_FRAC, MARKER_MIN_HEIGHT_CUBES,
                             MARKER_CUT_MODE,
                             MARKER_MAX_AXIS_ANGLE_DEG)
from pipeline.core.fill import (
    cap_point_cloud_bottom,
    cap_points_on_plane,
    extend_point_cloud_to_floor,
)


# ── Core ──

# ── Phase A helpers ────────────────────────────────────────────────────────
#
# Stage 3 used to be one 468-line function holding ninety local names, ten of
# them alive across more than two hundred lines. The phases it printed were
# already the natural seams, so each is a function now: what used to be implicit
# shared state is an argument or a return value you can read at the call site.


def _load_and_thin(dense_ply):
    """Read the dense cloud, drop outliers, and thin it enough to cluster.

    The outlier thresholds loosen as the cloud gets smaller: on a sparse cloud a
    2.5-sigma cut removes real surface, because the mean neighbour distance is
    itself noisy.
    """
    cloud = o3d.io.read_point_cloud(dense_ply)
    if len(cloud.points) == 0:
        raise ValueError("Empty dense point cloud")
    initial_count = len(cloud.points)
    print(f"Loaded dense cloud: {initial_count:,} points")

    if initial_count < 50000:
        std_ratio, neighbours = 3.5, 10
    elif initial_count < 200000:
        std_ratio, neighbours = 3.0, 15
    else:
        std_ratio, neighbours = 2.5, 20
    cloud, _ = cloud.remove_statistical_outlier(
        nb_neighbors=neighbours, std_ratio=std_ratio)
    print(f"SOR: {initial_count:,} → {len(cloud.points):,} (std_ratio={std_ratio})")

    # Purely a speed measure: RANSAC and DBSCAN both scale badly, and this runs
    # on the whole scene before either. The surface-scale decimation happens
    # later, in the ghost filter, at a size derived from the cloud itself.
    if len(cloud.points) > 100000:
        cloud = cloud.voxel_down_sample(voxel_size=0.002)
        print(f"Voxel downsample: → {len(cloud.points):,}")
    return cloud


def _split_into_objects(dense_cloud, num_objects):
    """Separate the scene into the reference cube and the subject.

    The floor is removed from a *copy* first, because while it is present it
    touches both objects and DBSCAN sees one connected blob. The dense cloud
    itself keeps its floor — Phase B needs it to find the ground plane.
    """
    without_floor = o3d.geometry.PointCloud(dense_cloud)
    without_floor, _ = remove_dominant_plane(without_floor)

    box_cluster, object_cluster = detect_top_k_objects(without_floor, k=num_objects)
    if box_cluster is None:
        box_cluster = without_floor
        object_cluster = None
        print("  Only 1 cluster on dense — treating as box")
    return box_cluster, object_cluster


CUT_MODES = ("upper", "span", "auto")


def resolve_cut_mode(cut_mode, n_bands=None, n_planes=None):
    """Which cut this run makes: the caller's choice, else the config default.

        "upper" — keep what is BELOW the highest valid plane
        "span"  — keep what lies BETWEEN the outermost two valid planes
        "auto"  — span when two marker planes survive Stage 3's gates,
                  upper otherwise

    None means "whatever config says", so an existing call site keeps behaving
    as configured.

    **What auto reads.** `n_planes` is how many marker planes Stage 3 fitted
    and then gated — at least one reference-cube height above the floor and
    within MARKER_MAX_AXIS_ANGLE_DEG of perpendicular to the limb's own axis.
    Those gates are the measured discriminator (genuine bands 2.4-27 degrees
    off the axis, false planes fitted to shorts and floor junctions 53-89), and
    a plane is the only thing that can actually cut, so the plane count decides.

    `n_bands` is Stage 0's count from the photographs, and it is a cross-check
    rather than a vote. It is not used to veto a span, because it is measured to
    UNDER-count: on inputs/sunshine2, a capture wearing an ankle cord and a
    below-knee cord, GroundingDINO returns the upper cord on 1 frame of 8 and
    below the corroboration bar, so requiring both sources to agree would make a
    span unreachable on the one capture that needs it. When the two disagree the
    run says so and keeps going — the disagreement is worth a reviewer's eye,
    not a silent override.

    The mode is still a claim about what was physically measured, and auto only
    infers it from what the pipeline can see. A capture compared against a
    ground truth taken foot-to-band must be run with an explicit --cut-mode
    upper even when two planes survive.
    """
    mode = (MARKER_CUT_MODE if cut_mode is None else cut_mode).lower()
    if mode not in CUT_MODES:
        raise ValueError(f"cut_mode must be one of {CUT_MODES}, got {cut_mode!r}")
    if mode != "auto":
        return mode

    if n_planes is None:
        # Called before detection has run — the banner, a summary line. Report
        # the configured intent rather than pretending to know the answer.
        return "auto"
    if n_planes < 2:
        # A span needs two planes to cut between, whatever Stage 0 saw.
        if n_bands is not None and n_bands >= 2:
            print(f"  Cut mode auto → upper: Stage 0 counted {n_bands} bands "
                  f"but only {n_planes} plane(s) survived Stage 3's gates, and "
                  f"a span has to be bounded by two")
        return "upper"
    if n_bands is not None and n_bands < 2:
        print(f"  Cut mode auto → span: {n_planes} gated planes. Stage 0 "
              f"counted {n_bands} band(s) in the photographs — worth a look at "
              f"the review, since the two disagree")
    else:
        print(f"  Cut mode auto → span: {n_planes} gated planes"
              + (f", and Stage 0 counted {n_bands} bands"
                 if n_bands is not None else ""))
    return "span"


# How close two planes' centroids may sit, along the height axis, before they
# are taken to be the same band seen twice. In VGGT world units, where a 10 cm
# reference cube is about 0.13 across, so 0.02 is roughly 1.5 cm -- wider than
# the disagreement between a colour fit and a projected fit on one cord, far
# narrower than the gap between two cords bounding a measured segment.
SAME_BAND_DISTANCE = 0.02


def _merge_projected_planes(colour_planes, band_planes, height_axis):
    """Add planes projected from Stage 0's boxes that colour detection missed.

    Additive by construction: a colour plane is never dropped or moved, so a
    capture cannot end up with fewer planes, or different ones, than it has
    today. What it can gain is a band the colour rule could not see -- which is
    the common failure, since the rule needs the cord and the limb to be
    chromatically separable and refuses outright below MARKER_MIN_AXIS.

    Where both sources found the same band, the colour plane wins. It is fitted
    to the cord's own points; the projected one is fitted to a slab of limb
    surface centred on the cord, which is the right plane but a blunter
    instrument for locating it.
    """
    if not band_planes:
        return colour_planes

    axis_idx = AXIS_MAP[height_axis.lower()]
    merged = list(colour_planes)
    for candidate in band_planes:
        height = float(candidate["centroid"][axis_idx])
        twin = next((m for m in colour_planes
                     if abs(float(m["centroid"][axis_idx]) - height)
                     < SAME_BAND_DISTANCE), None)
        if twin is not None:
            print(f"  Band projection: {candidate['npts']:,}-pt plane agrees "
                  f"with a {twin['npts']}-pt colour plane — keeping the colour "
                  f"fit")
            continue
        merged.append({"centroid": np.asarray(candidate["centroid"], dtype=np.float64),
                       "normal": np.asarray(candidate["normal"], dtype=np.float64),
                       "npts": int(candidate["npts"]),
                       "source": "projected"})
        print(f"  Band projection: adding a plane colour detection missed "
              f"({candidate['npts']:,} pts, {candidate['frames']} frames)")
    return merged


def _detect_marker_planes(object_cluster, height_axis, marker_colour):
    """Cutting planes from the coloured band, in original VGGT space.

    Detection runs on the dense cluster rather than the filtered one because it
    is a colour test, and thinning throws away exactly the sparse coloured points
    it depends on. Returns [] rather than raising: a capture with no band is a
    valid capture that simply will not be cut.
    """
    markers = []
    if object_cluster is None or len(object_cluster.points) == 0:
        return markers
    try:
        if marker_colour:
            print(f"  Marker colour from Stage 0: "
                  f"RGB {[int(v) for v in marker_colour['rgb']]} "
                  f"(ExG {marker_colour.get('exg', 0):+.0f}) — "
                  f"thresholds follow the measured band, not the config "
                  f"defaults")
        _, summary = segment_point_cloud(object_cluster,
                                         height_axis=height_axis,
                                         verbose=False,
                                         marker_colour=marker_colour)
        for _, centroid, normal, point_count, _ in summary.get("planes", []):
            unit_normal = normal / (np.linalg.norm(normal) + 1e-8)
            markers.append({"centroid": np.array(centroid, dtype=np.float64),
                            "normal": np.array(unit_normal, dtype=np.float64),
                            "npts": int(point_count)})
        print(f"  Dense leg markers: {len(markers)} plane(s) found")
    except Exception as e:
        print(f"  Marker detection skipped: {e}")
    return markers


def _limb_axis_near(leg_pts_rot, z, window_frac=0.25, n_slices=10, min_pts=8):
    """The limb's own direction near height `z`, from slice centroids.

    Fitting the principal direction of a slab of limb points does NOT give the
    limb's axis, and getting that wrong rejects real bands. A calf is roughly
    10 cm across; a slab thin enough to be local is thinner than that, so its
    direction of greatest extent is the limb's WIDTH and comes out horizontal.
    Measured consequence: genuine bands on inputs/orange_shirt and black_shirt
    scored 53 and 65 degrees "off the limb axis" and were thrown out.

    Slice centroids do not have that failure. Each centroid sits on the limb's
    centre line whatever the cross-section looks like, so a line through them
    follows the limb however wide it is. This is the same construction the
    band-colour experiment used to fit a limb axis.

    Returns a unit vector, or None when there are too few usable slices.
    """
    pts = np.asarray(leg_pts_rot, dtype=np.float64)
    lo, hi = float(pts[:, 2].min()), float(pts[:, 2].max())
    span = hi - lo
    if span <= 0:
        return None

    for widen in (1.0, 1.6, 2.4):
        half = window_frac * widen * span
        z0, z1 = max(lo, z - half), min(hi, z + half)
        if z1 - z0 <= 0:
            continue
        edges = np.linspace(z0, z1, n_slices + 1)
        centroids = []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = pts[(pts[:, 2] >= a) & (pts[:, 2] < b)]
            if len(sel) >= min_pts:
                centroids.append(sel.mean(axis=0))
        if len(centroids) >= 4:
            c = np.asarray(centroids)
            _, _, Vt = np.linalg.svd(c - c.mean(axis=0), full_matrices=False)
            axis = Vt[0, :]
            return axis / (np.linalg.norm(axis) + 1e-12)
    return None


def _plane_vs_limb_angle(leg_pts_rot, marker):
    """Angle in degrees between a marker plane's normal and the limb's axis.

    A cord tied round a limb lies across it, so its plane's normal points along
    the limb and this angle is small. A plane fitted to a blob of skin or
    clothing takes that blob's own orientation instead, and this is the only
    test that sees the difference — the blob can be large, well clustered and
    at a perfectly plausible height.

    The axis is fitted locally, near the plane's own height, because a limb is
    not straight: a global axis would score a band on the shin against the mean
    of shin and thigh and reject it for the bend rather than for being wrong.

    Returns None when the limb is too sparse there to fit an axis, which means
    "do not judge" rather than pass or fail.
    """
    axis = _limb_axis_near(leg_pts_rot, float(marker["centroid"][2]))
    if axis is None:
        return None
    n = np.asarray(marker["normal"], dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    # Both directions are sign-free, so compare with |dot|.
    cos = abs(float(np.dot(n, axis)))
    return float(np.degrees(np.arccos(min(1.0, cos))))


def _save_cluster_debug(debug_dir, object_cluster, box_cluster, markers):
    """The dense cluster and the planes found on it, in ORIGINAL VGGT space.

    Written before levelling, so these coordinates do not share a frame with any
    exported mesh. `cutting_line_levelled.json`, written later, is the one a
    viewer should read.
    """
    import json as _json

    source = object_cluster if (object_cluster is not None
                                and len(object_cluster.points) > 0) else box_cluster
    if source is not None and len(source.points) > 0:
        _quick_save_ply(np.asarray(source.points, dtype=np.float32), None,
                        os.path.join(debug_dir, "leg_cluster.ply"))

    with open(os.path.join(debug_dir, "cutting_line.json"), "w") as f:
        _json.dump({"markers": [{"centroid": m["centroid"].tolist(),
                                 "normal": m["normal"].tolist(),
                                 "npts": m.get("npts", 0)} for m in markers]},
                   f, indent=2)
    print("  Saved: debug/leg_cluster.ply + debug/cutting_line.json")


def _ghost_filter_scales(dense_cloud):
    """(dedup voxel size, length scale for the normal filter).

    One size for the whole scene, not per cluster: derived per object the two
    would differ and leave the cube and the limb at inconsistent densities,
    which then feeds MLS two different neighbourhood radii.

    GHOST_VOXEL_FACTOR = 0 disables deduplication but the normal filter still
    needs a length scale, so that is computed either way.
    """
    from pipeline.config import GHOST_VOXEL_FACTOR
    from pipeline.ghost import compute_voxel_size

    all_points = np.asarray(dense_cloud.points, dtype=np.float32)
    if GHOST_VOXEL_FACTOR <= 0:
        print("  Ghost voxel dedup: DISABLED (GHOST_VOXEL_FACTOR=0) — keeping "
              "every point; ghost layers survive for the normal filter to catch")
        return 0.0, compute_voxel_size(all_points, factor=1.0)
    voxel_size = compute_voxel_size(all_points)
    print(f"  Ghost voxel_size: {voxel_size:.4f} (shared across clusters)")
    return voxel_size, voxel_size


def _clean_cluster(cluster, label, voxel_size, normal_filter_scale):
    """Deduplicate, drop misoriented points, and project onto a fitted surface.

    Runs per cluster so no MLS neighbourhood ever spans two objects.
    """
    from pipeline.config import MLS_RADIUS_MULT, MLS_BOX_POLYNOMIAL
    from pipeline.ghost import ghost_voxel_downsample, normal_aware_filter

    if cluster is None or len(cluster.points) == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8))
    points = np.asarray(cluster.points, dtype=np.float32)
    colours = (np.clip(np.asarray(cluster.colors, dtype=np.float32), 0, 1)
               * 255).astype(np.uint8)
    before = len(points)
    points, colours = ghost_voxel_downsample(points, colours, voxel_size)
    points, colours = normal_aware_filter(points, colours, normal_filter_scale)
    print(f"  Ghost filter [{label}]: {before:,} → {len(points):,} pts")

    if MLS_RADIUS_MULT and MLS_RADIUS_MULT > 0 and len(points) > 50:
        from pipeline.ghost import mls_project
        # The reference is known to be planar; the limb is not.
        use_quadratic = MLS_BOX_POLYNOMIAL if label == "box" else True
        print(f"  MLS [{label}]:", end=" ")
        points, colours, _ = mls_project(points, colours,
                                         radius_mult=MLS_RADIUS_MULT,
                                         polynomial=use_quadratic)
    return points, colours


# ── Phase B and C helpers ──────────────────────────────────────────────────


def _level_to_ground(dense_cloud, seed):
    """Rotate the scene so the ground plane is flat and Z points up.

    Rotates `dense_cloud` IN PLACE and returns the total rotation, because every
    other cloud -- and the marker planes, which were detected in original VGGT
    space -- has to receive exactly the same transform. Handing back the matrix
    is what lets the caller apply it to all of them.

    The second RANSAC decides which way up the scene ended: if the bulk of the
    non-floor points sits BELOW the floor plane, the fit picked the normal's
    other direction and everything is upside down.
    """
    threshold = auto_ransac_threshold(dense_cloud, base_factor=3)
    plane_model, _ = detect_plane_ransac_deterministic(
        dense_cloud, distance_threshold=threshold, num_iterations=1000, seed=seed)
    ground_normal = plane_model[:3]
    print(f"RANSAC plane normal: ({ground_normal[0]:.4f}, "
          f"{ground_normal[1]:.4f}, {ground_normal[2]:.4f})")

    upright_rotation = get_rotation_to_z_axis(ground_normal)
    total_rotation = np.asarray(upright_rotation, dtype=np.float64)
    dense_cloud.rotate(upright_rotation, center=(0, 0, 0))
    levelled_points = np.asarray(dense_cloud.points, dtype=np.float32)

    try:
        _, floor_indices = detect_plane_ransac_deterministic(
            dense_cloud, distance_threshold=threshold, num_iterations=500, seed=seed)
        floor_height = np.mean(levelled_points[floor_indices, 2])
        off_floor = np.ones(len(levelled_points), dtype=bool)
        off_floor[floor_indices] = False
        if off_floor.any() and np.mean(levelled_points[off_floor, 2]) < floor_height:
            print("Upside-down detected — flipping 180°")
            flip = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))
            dense_cloud.rotate(flip, center=(0, 0, 0))
            levelled_points = np.asarray(dense_cloud.points, dtype=np.float32)
            total_rotation = np.asarray(flip, dtype=np.float64) @ total_rotation
    except Exception as e:
        print(f"Upside-down check failed: {e}")

    return total_rotation, levelled_points


def _find_ground_height(dense_cloud, levelled_points, seed):
    """Height of the floor in levelled space, or None if it cannot be trusted.

    Three tests, all of which must pass, because cutting at a plane that is not
    the floor would silently remove part of the object:
      horizontal  -- its normal is within ~32 degrees of vertical
      at the bottom -- it sits in the lowest quarter of the scene
      substantial -- it accounts for at least 5% of the points

    Returning None means "do not cut", which is the safe direction: the object
    keeps a little floor rather than losing its base.
    """
    try:
        threshold = min(auto_ransac_threshold(dense_cloud, base_factor=3), 0.015)
        plane_model, floor_indices = detect_plane_ransac_deterministic(
            dense_cloud, distance_threshold=threshold, num_iterations=1000, seed=seed)
        plane_normal = plane_model[:3]
        is_horizontal = abs(np.dot(plane_normal, [0, 0, 1])) > 0.85

        heights = levelled_points[:, 2]
        lowest, highest = np.min(heights), np.max(heights)
        plane_height = np.median(levelled_points[floor_indices, 2])
        is_at_the_bottom = (plane_height - lowest) < 0.25 * (highest - lowest)
        share_of_points = len(floor_indices) / len(levelled_points)

        if is_horizontal and is_at_the_bottom and share_of_points > 0.05:
            print(f"Floor plane at Z={plane_height:.4f} "
                  f"(ratio={share_of_points:.1%})")
            return plane_height
        print(f"Floor cut skipped (horiz={is_horizontal}, "
              f"bottom={is_at_the_bottom}, ratio={share_of_points:.1%})")
    except Exception as e:
        print(f"Floor detection failed: {e}")
    return None


def _drop_below_floor(points, colours, floor_height, margin=0.008):
    """Remove everything at or under the floor. A no-op if there is no floor."""
    if floor_height is None or len(points) == 0:
        return points, colours
    above = points[:, 2] > (floor_height + margin)
    return points[above], colours[above]


def _segment_and_export(dense_ply, output_dir, num_objects=2, seed=42,
                        marker_colour=None,
                            segment_leg=False, segment_height_axis="z",
                            fill_enabled=True, apply_cut=True,
                            override_planes=None, cut_mode=None, n_bands=None,
                            band_planes=None):
    """Cluster-first clean pipeline.

    Phase A: Segment the dense cloud once (floor removal + DBSCAN), detect
             marker cut planes on the dense leg, then ghost-filter each
             identified cluster separately. Clustering on the dense cloud has
             better statistics than on the filtered one, and splitting before
             filtering keeps point identity so no label transfer is needed.
    Phase B: RANSAC leveling on dense; the same rotation is applied to every
             sub-cloud (including the marker planes — see R_total).
    Phase C: Floor cut, extend to the detected floor, bottom cap — producing a
             complete UNCUT cloud — then optionally the centroid-side marker cut
             and its cut-plane cap.

    apply_cut=False stops after the uncut cloud and publishes the cutting planes
    instead. That is what an interactive review needs: the user sees a complete
    object plus where the pipeline would cut it, and decides.

    override_planes closes that loop: it is the decision coming back. Given a
    list of {"centroid", "normal"} in LEVELLED space, it replaces whatever
    detection found, and everything downstream — the cut, the cross-section
    caps, the volume — follows the person rather than the colour threshold.
    """
    from pipeline.ghost import ghost_voxel_downsample, normal_aware_filter, compute_voxel_size

    objects_dir = os.path.join(output_dir, "objects")
    debug_dir_out = os.path.join(output_dir, "debug")
    os.makedirs(objects_dir, exist_ok=True)
    os.makedirs(debug_dir_out, exist_ok=True)

    import json as _json

    # ── Load ──
    pcd_dense = _load_and_thin(dense_ply)

    # ── Phase A: cluster in original VGGT space ──
    #
    # Everything here happens BEFORE levelling and before any thinning of the
    # surface, because both of the things this phase decides -- which cluster is
    # the reference, and where the marker band is -- are damaged by thinning.
    print()
    print("─" * 40)
    print("PHASE A: Clustering (original VGGT space)")
    print("─" * 40)

    box_dense, obj_dense = _split_into_objects(pcd_dense, num_objects)

    markers = []
    if segment_leg:
        markers = _detect_marker_planes(obj_dense, segment_height_axis, marker_colour)
        markers = _merge_projected_planes(markers, band_planes, segment_height_axis)

    _save_cluster_debug(debug_dir_out, obj_dense, box_dense, markers)

    # Ghost filter each identified cluster separately.
    #
    # Filtering before clustering (the old flow) forced a second DBSCAN on a
    # ~14x sparser cloud and left the two results free to disagree about which
    # object is the box -- nothing checked them. Splitting first keeps each
    # point's identity, so no label transfer is needed and the segmentation is
    # decided once, on the denser and better-conditioned cloud.
    voxel_size, nn_scale = _ghost_filter_scales(pcd_dense)

    box_pts_arr, box_cols_arr = _clean_cluster(box_dense, "box", voxel_size, nn_scale)
    leg_pts_arr, leg_cols_arr = _clean_cluster(obj_dense, "obj", voxel_size, nn_scale)

    if obj_dense is None and len(box_pts_arr) > 0:
        print("  Only 1 cluster — treating it as the box (no obj)")

    print(f"  Clean: leg={len(leg_pts_arr):,}, box={len(box_pts_arr):,}")

    # ── Phase B: Leveling ──
    print()
    print("─" * 40)
    print("PHASE B: Leveling (shared transform)")
    print("─" * 40)

    R_total, dense_pts = _level_to_ground(pcd_dense, seed)
    leg_pts_rot = _rotate_points(leg_pts_arr, R_total)
    box_pts_rot = _rotate_points(box_pts_arr, R_total)

    # ── Phase C: Post-leveling cuts + cap ──
    print()
    print("─" * 40)
    print("PHASE C: Cuts and export")
    print("─" * 40)

    floor_z = _find_ground_height(pcd_dense, dense_pts, seed)
    leg_pts_rot, leg_cols_arr = _drop_below_floor(leg_pts_rot, leg_cols_arr, floor_z)
    box_pts_rot, box_cols_arr = _drop_below_floor(box_pts_rot, box_cols_arr, floor_z)
    print(f"Floor cut: leg → {len(leg_pts_rot):,}, box → {len(box_pts_rot):,}")

    # Rotate the marker planes into levelled space. This happens before any
    # cutting so the planes can be published for review whether or not the cut
    # is applied here.
    leg_path = os.path.join(objects_dir, "leg.ply")
    markers_rotated = []
    candidates = []
    if len(markers) > 0:
        # Rotate EVERY candidate first, gate them, and only then cap the count.
        #
        # The cap used to run first, keeping the two with the most points. That
        # ranking prefers exactly the wrong thing: a real band is small because
        # only its camera-facing arc reconstructs -- 964 points on inputs/champ,
        # 299 on black_shirt, 208 on keng -- while a false plane fitted to
        # clothing or floor is thousands. On champ the genuine band (964 pts)
        # was dropped in favour of the shorts (5,265) and a floor-junction blob
        # (2,597); the height gate then removed the blob and the cut ran on the
        # shorts alone. Nothing had looked at either survivor before the real
        # one was discarded.
        for m in markers:
            cen = np.array(m["centroid"])
            norm = np.array(m["normal"])
            cen_r = (R_total @ cen).astype(np.float64)
            norm_r = (R_total @ norm).astype(np.float64)
            norm_r /= np.linalg.norm(norm_r) + 1e-8
            markers_rotated.append({"centroid": cen_r.tolist(), "normal": norm_r.tolist(),
                                    "npts": m.get("npts", 0),
                                    "source": m.get("source", "colour")})

        # Drop planes sitting in the bottom fraction of the object. Feet, arch
        # shadows and the floor junction all live there, and a cut line that low
        # would discard nearly the whole limb.
        #
        # This has to happen HERE rather than in segment_point_cloud: detection
        # runs in original VGGT space, where the vertical axis is whatever the
        # camera happened to give (on small_leg the limb's long axis is Y, not
        # Z). Only after R_total does "height" mean height.
        if len(leg_pts_rot) > 0 and len(markers_rotated) > 0:
            lo = float(leg_pts_rot[:, 2].min())
            span = float(leg_pts_rot[:, 2].max()) - lo
            # Prefer a floor measured in reference-cube heights: the cube is a
            # known physical length standing on the same ground, where the
            # limb's span is only however much leg was in shot.
            cube_h = (float(box_pts_rot[:, 2].max() - box_pts_rot[:, 2].min())
                      if len(box_pts_rot) > 10 else 0.0)
            if cube_h > 0 and MARKER_MIN_HEIGHT_CUBES > 0:
                min_z = lo + MARKER_MIN_HEIGHT_CUBES * cube_h
                rule = (f"< {MARKER_MIN_HEIGHT_CUBES:g} cube height"
                        f"{'s' if MARKER_MIN_HEIGHT_CUBES != 1 else ''} "
                        f"({cube_h:.3f}) above the floor")
            elif span > 0 and MARKER_MIN_HEIGHT_FRAC > 0:
                min_z = lo + MARKER_MIN_HEIGHT_FRAC * span
                rule = (f"< {MARKER_MIN_HEIGHT_FRAC*100:.0f}% of the limb's "
                        f"span — no cube to measure against")
            else:
                min_z = None
            if min_z is not None:
                keep_m, drop_m = [], []
                for mr in markers_rotated:
                    # A plane projected from Stage 0's boxes is exempt. This
                    # gate rejects UNCORROBORATED blobs near the floor — feet,
                    # arch shadows, the floor junction — and a band the
                    # detector saw on most of the photographs is precisely the
                    # corroboration it was standing in for. It is also what
                    # would throw the low band away: on the capture this was
                    # built for, the real ankle cord sits at 0.66 cube heights
                    # against a 1.0 floor.
                    low = mr["centroid"][2] < min_z
                    if low and mr.get("source") == "projected":
                        h = (mr["centroid"][2] - lo) / span if span > 0 else 0.0
                        print(f"  Marker kept at {h*100:.0f}% of height despite "
                              f"{rule} — projected from Stage 0's band boxes, "
                              f"which is the corroboration the gate wants")
                        low = False
                    (drop_m if low else keep_m).append(mr)
                for mr in drop_m:
                    h = (mr["centroid"][2] - lo) / span if span > 0 else 0.0
                    print(f"  Marker rejected: {h*100:.0f}% of height, "
                          f"{rule}, {mr['npts']} pts")
                markers_rotated = keep_m

        # Drop planes that are not perpendicular to the limb.
        #
        # A cord tied round a limb lies across it, so the plane's normal points
        # along the limb. A plane fitted to a blob of skin or clothing instead
        # takes that blob's own principal direction, which is unrelated -- and
        # this is the only test that sees the difference, because the blob can
        # be large, well-clustered and at a perfectly plausible height. Measured
        # false planes: 87.4 deg on sunshine (43,468 pts), 83.1 on keng,
        # 41.5 on champ's shorts, against 15.7-19.5 deg for every genuine band.
        if len(markers_rotated) > 0 and len(leg_pts_rot) > 20:
            keep_m, drop_m = [], []
            for mr in markers_rotated:
                ang = _plane_vs_limb_angle(leg_pts_rot, mr)
                (drop_m if (ang is not None and ang > MARKER_MAX_AXIS_ANGLE_DEG)
                 else keep_m).append((mr, ang))
            for mr, ang in drop_m:
                print(f"  Marker rejected: {ang:.0f}° off the limb's own axis "
                      f"(> {MARKER_MAX_AXIS_ANGLE_DEG:.0f}°), {mr['npts']} pts")
            markers_rotated = [mr for mr, _ in keep_m]

        # Select which validated planes actually cut. Done LAST, once every
        # survivor has passed the gates above, and here rather than inside the
        # cut so the published planes, the cross-section caps and the cut all
        # see the same set.
        #
        # MARKER_CUT_MODE decides, because the number of cuts is a property of
        # what was measured rather than of how many bands the detector found. A
        # subject wearing an ankle band and a knee band is one capture whether
        # the ruler measured the whole leg below the knee or only the segment
        # between the two, and guessing from the band count answers a different
        # question from the one the ground truth answers.
        #
        # `cut_mode` is the per-run answer to that question and defaults to
        # MARKER_CUT_MODE, so one capture set can hold both a single-band
        # subject and a two-band one without editing config between runs.
        #
        # Every survivor of the gates is kept as a CANDIDATE regardless of what
        # the mode then selects, and published alongside the selection. The two
        # answer different questions: "which planes did this run cut on" and
        # "which bands does this capture actually have". The review screen needs
        # the second — seeded from the selection alone, a two-band capture run
        # in 'upper' mode showed the reviewer one plane, and re-adding the other
        # by hand replaced a fitted marker plane with a guess at mid-height.
        candidates = sorted(markers_rotated, key=lambda m: m["centroid"][2])

        mode = resolve_cut_mode(cut_mode, n_bands=n_bands,
                                n_planes=len(markers_rotated))
        if len(markers_rotated) > 1:
            ordered = candidates
            if mode == "upper":
                markers_rotated = [ordered[-1]]
                dropped = ", ".join(f"{m['npts']} pts" for m in ordered[:-1])
                print(f"  Markers: {len(ordered)} valid — cut mode 'upper', "
                      f"cutting on the UPPERMOST only (keep below); not "
                      f"cutting on {dropped}")
            else:
                markers_rotated = [ordered[0], ordered[-1]]
                print(f"  Markers: {len(ordered)} valid — cut mode 'span', "
                      f"keeping the outermost {MAX_MARKERS} (keep between)")
        elif len(markers_rotated) == 1 and mode == "span":
            # Saying so matters: 'span' names a segment bounded at both ends,
            # and one plane cannot bound it. The run still measures something
            # — everything below the single plane — but that is a different
            # quantity from the one asked for, and a silent substitution here
            # would be read as the segment volume.
            print("  Markers: 1 valid — cut mode 'span' asks for the segment "
                  "BETWEEN two planes, but only one survived the gates. "
                  "Cutting on it alone (keep below); this is NOT a span "
                  "measurement.")

        # Also publish the planes in LEVELLED space. cutting_line.json is written
        # pre-levelling, but leg_no_cut.ply is post-levelling, so the two do not
        # share a frame — anything drawing them together (a UI, a debug viewer)
        # needs this version, and "height along Z" only means something here.
        #
        # "markers" keeps its meaning byte for byte — the planes this run cut on
        # — because Stage 6 reads it to report a circumference per cutting
        # plane, and a candidate that did not cut has no cut face to measure.
        # "candidates" is additive: every gated survivor, lowest first.
        with open(os.path.join(debug_dir_out, "cutting_line_levelled.json"), "w") as _f:
            _json.dump({"markers": markers_rotated,
                        "candidates": candidates,
                        "cut_mode": mode,
                        "space": "levelled"}, _f, indent=2)

    # Publish the levelling rotation itself. Stage 1's pointmap lives in the
    # unlevelled frame while every exported cloud and mesh lives in this one, so
    # without R_total nothing downstream can relate the two — which is what
    # Stage 6 needs to express a marker measurement as vertical or horizontal.
    # Rotation preserves length, so pure distances do not need it; directions do.
    with open(os.path.join(debug_dir_out, "levelling.json"), "w") as _f:
        _json.dump({"R_total": np.asarray(R_total, dtype=float).tolist(),
                    # The height the floor sits at in levelled space. A cut
                    # applied later has to close down to the same floor this
                    # pass used, or the two would disagree about where the
                    # object ends.
                    "floor_z": None if floor_z is None else float(floor_z),
                    "note": "levelled = R_total @ pointmap; floor normal is +Z"},
                   _f, indent=2)

    # ── Planes supplied by a review, if any ──
    #
    # These arrive already in levelled space — the frame the review works in and
    # the frame cutting_line_levelled.json publishes — so they need no rotation.
    # Applied here rather than earlier so the detected planes are still written
    # out above: a review that overrules detection should not erase the record
    # of what was measured.
    #
    # MARKER_MIN_HEIGHT_FRAC is deliberately not applied. It exists to reject
    # spurious detections near the floor, and a plane a person placed by hand is
    # not one; silently discarding it would look like the cut was ignored.
    if override_planes is not None:
        markers_rotated = []
        for m in list(override_planes)[:MAX_MARKERS]:
            cen = np.asarray(m["centroid"], dtype=np.float64)
            norm = np.asarray(m["normal"], dtype=np.float64)
            norm = norm / (np.linalg.norm(norm) + 1e-8)
            markers_rotated.append({"centroid": cen.tolist(),
                                    "normal": norm.tolist(),
                                    "npts": int(m.get("npts", 0))})
        print(f"Markers: {len(markers_rotated)} supplied by review "
              f"(detection found {len(markers)})")
        # Detection's candidates travel with the review's choice. The review
        # copy is what the UI reloads after an edit, so dropping them here would
        # lose the detected bands the moment the user cut once — the reviewer
        # could no longer put back a plane they had removed.
        with open(os.path.join(debug_dir_out, "cutting_line_review.json"), "w") as _f:
            _json.dump({"markers": markers_rotated,
                        "candidates": candidates,
                        "space": "levelled",
                        "source": "review"}, _f, indent=2)

    # ── Close whatever the cut actually leaves open ──
    #
    # The floor extension and bottom cap fabricate a base at floor level. That
    # is correct only when the floor really is the bottom of the region being
    # measured, which depends on the cut:
    #
    #   0 markers -> no cut, bottom is the floor            -> floor cap
    #   1 marker  -> keep below, bottom is still the floor  -> floor cap
    #   2 markers -> keep between, bottom is the lower cut face -> plane cap
    #
    # Filling before the cut got the 2-marker case wrong. The fabricated skirt
    # was usually discarded by the cut and merely wasted, but a marker placed
    # low on the limb left part of it inside the kept segment — invented
    # geometry in the middle of a measurement, and a lower face that bulged
    # instead of sitting flat.
    #
    # leg_no_cut.ply is exempt: it is the review cloud and what gets measured
    # if the user declines to cut, so it is always floor-closed. It is built
    # from its own copy, and the cut runs on the unfilled cloud.
    o3d_box = _array_to_o3d(box_pts_rot, box_cols_arr)
    if fill_enabled and len(o3d_box.points) > 0:
        # The box is never cut, so its base is always the floor.
        # Extend down to the floor first — capping at the raw z_min would seal
        # the object above the ground and lose the shadowed base entirely.
        # alpha matches the cap below: the cube's silhouette is convex, so a
        # concave outline could only be noise biting into a corner.
        o3d_box = extend_point_cloud_to_floor(o3d_box, floor_z, alpha=0.0,
                                              label="box")
        o3d_box = cap_point_cloud_bottom(o3d_box, alpha=0.0)

    def _close_to_floor(pts, cols):
        """Sweep the bottom band down to the floor and cap it."""
        if not fill_enabled or len(pts) == 0:
            return pts, cols
        p = _array_to_o3d(pts, cols)
        p = extend_point_cloud_to_floor(p, floor_z, label="obj")
        p = cap_point_cloud_bottom(p, alpha=2.0)
        out_pts = np.asarray(p.points, dtype=np.float32)
        out_cols = ((np.clip(np.asarray(p.colors, dtype=np.float32), 0, 1) * 255)
                    .astype(np.uint8) if p.has_colors() else cols)
        return out_pts, out_cols

    def _close_top(pts, cols):
        """Cap the limb's open upper end, the way the bottom is capped.

        The limb does not end at the top; the reconstruction simply stops where
        the frame did, leaving an open tube. Poisson wants a closed surface, and
        on an open one it returns chi=-2 -- not a single closed solid -- which
        sends Stage 4 to its alpha-shape fallback. That ladder has to open to 40x
        point spacing before it seals, and a wrap that loose reads volume high:
        it is what put inputs/sunshine 10% over its displacement volume.

        This did not arise while Stage 3 cut the cloud, because Stage 4 then only
        ever saw a limb whose top was the cut face, already capped. Now that the
        UNCUT limb is what gets reconstructed, its top has to be closed here.
        """
        if not fill_enabled or len(pts) == 0:
            return pts, cols
        height_index = {"x": 0, "y": 1, "z": 2}[segment_height_axis]
        plane_normal = [0.0, 0.0, 0.0]
        plane_normal[height_index] = 1.0
        top_height = float(np.asarray(pts)[:, height_index].max())
        plane_point = [0.0, 0.0, 0.0]
        plane_point[height_index] = top_height
        return cap_points_on_plane(pts, cols, plane_point, plane_normal,
                                   label="limb top")

    # The complete limb, closed at both ends. Stage 3 publishes exactly one limb
    # cloud now: the open/closed pair existed only because the cut used to happen
    # here and had to run on the open cloud before closing what it left open. The
    # cut is applied to Stage 5's solid instead, so there is nothing left to
    # sequence around -- but the limb must now be closed at the top as well as
    # the bottom, because it is reconstructed uncut.
    nc_pts, nc_cols = _close_to_floor(leg_pts_rot, leg_cols_arr)
    nc_pts, nc_cols = _close_top(nc_pts, nc_cols)
    _quick_save_ply(nc_pts, nc_cols, leg_path)
    print(f"Saved (complete limb): {leg_path} ({len(nc_pts):,} pts)")

    n_markers = len(markers_rotated)
    if apply_cut and n_markers > 0 and len(leg_pts_rot) > 0:
        keep_mask, cut_case = apply_marker_cut(leg_pts_rot.astype(np.float64),
                                               markers_rotated)
        n_kept = int(keep_mask.sum())
        if n_kept >= 50:
            leg_pts_rot = leg_pts_rot[keep_mask]
            leg_cols_arr = leg_cols_arr[keep_mask]
            print(f"Marker cut ({cut_case}): leg → {len(leg_pts_rot):,} pts")

            # Cap the exposed cross-section. Left open, the surface solver
            # rounds it into a dome instead of the flat face the cut produced.
            if fill_enabled:
                for i, mr in enumerate(markers_rotated):
                    leg_pts_rot, leg_cols_arr = cap_points_on_plane(
                        leg_pts_rot, leg_cols_arr,
                        mr["centroid"], mr["normal"], label=f"marker {i}")
                leg_pts_rot = leg_pts_rot.astype(np.float32)
            if n_markers < 2:
                # One plane truncates the top; the bottom is still the floor.
                leg_pts_rot, leg_cols_arr = _close_to_floor(leg_pts_rot, leg_cols_arr)
        else:
            print(f"Marker cut ({cut_case}): only {n_kept} pts (< 50) — keeping full leg")
            leg_pts_rot, leg_cols_arr = nc_pts, nc_cols
        o3d_leg = _array_to_o3d(leg_pts_rot, leg_cols_arr)
    else:
        # No cut applied, so the review cloud is also the measured cloud.
        if not apply_cut:
            print("Marker cut deferred — cutting planes saved for interactive review")
        leg_pts_rot, leg_cols_arr = nc_pts, nc_cols
        o3d_leg = _array_to_o3d(leg_pts_rot, leg_cols_arr)

    # Export
    leg_cut_path = os.path.join(objects_dir, "leg_cut.ply")
    box_path = os.path.join(objects_dir, "box.ply")
    merged_path = os.path.join(objects_dir, "merged.ply")

    # the limb + box
    #
    # Stage 3 no longer cuts: it hands the UNCUT limb forward so Stage 4 can
    # reconstruct it and Stage 5 can cut the resulting solid. That is what puts a
    # surface in front of the person placing the cut, instead of a point cloud,
    # and it leaves no reconstruction step between the cut and the measurement.
    #
    # No leg_cut.ply is written here in either case. Stage 6 measures the cut
    # solid Stage 5 produces, so a cut cloud under that name would be a second,
    # unmeasured answer to the same question.
    output_paths = []
    if apply_cut and len(o3d_leg.points) > 0:
        o3d.io.write_point_cloud(leg_cut_path, o3d_leg)
        output_paths.append(leg_cut_path)
        print(f"Saved: {leg_cut_path} ({len(o3d_leg.points):,} pts)")
    elif not apply_cut:
        if os.path.exists(leg_cut_path):
            os.remove(leg_cut_path)   # a stale one would be measured instead
        if os.path.exists(leg_path):
            output_paths.append(leg_path)
        print("Stage 3 deferred the cut — passing the uncut limb to Stage 4")
    if len(o3d_box.points) > 0:
        o3d.io.write_point_cloud(box_path, o3d_box)
        output_paths.append(box_path)
        print(f"Saved: {box_path} ({len(o3d_box.points):,} pts)")

    # merged.ply
    if apply_cut and len(o3d_leg.points) > 0 and len(o3d_box.points) > 0:
        merged_pcd = o3d.geometry.PointCloud()
        mp = np.vstack([np.asarray(o3d_leg.points), np.asarray(o3d_box.points)])
        mc = np.vstack([np.asarray(o3d_leg.colors), np.asarray(o3d_box.colors)])
        merged_pcd.points = o3d.utility.Vector3dVector(mp)
        merged_pcd.colors = o3d.utility.Vector3dVector(mc)
        o3d.io.write_point_cloud(merged_path, merged_pcd)
        print(f"Saved: {merged_path} ({len(mp):,} pts)")

    if not output_paths:
        print("WARNING: No objects exported")

    return output_paths


def _rotate_points(points, R):
    """Apply 3x3 rotation matrix to (N,3) point array."""
    if points.size == 0 or points.ndim < 2:
        return points
    return (points @ R.T).astype(np.float32)


def _quick_save_ply(points, colors, path):
    """Save numpy points to PLY via trimesh."""
    import trimesh as _trimesh
    pc = _trimesh.PointCloud(points, colors=colors) if colors is not None else _trimesh.PointCloud(points)
    pc.export(path)


def _array_to_o3d(points, colors_uint8):
    """Build Open3D PointCloud from numpy arrays."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors_uint8 is not None and len(colors_uint8) == len(points):
        colors_float = colors_uint8.astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_float)
    return pcd


def clean_and_extract(ply_path, output_dir, num_objects=2, seed=42,
                      segment_leg=False, segment_height_axis="z",
                      fill_enabled=True, clean_ply_path=None, apply_cut=False,
                      marker_colour=None, override_planes=None, cut_mode=None,
                      n_bands=None, band_planes=None):
    """Pipeline wrapper. clean_ply_path is accepted for call-site compatibility
    and ignored — Stage 3 derives its own ghost-filtered clouds from ply_path."""
    print()
    print("=" * 60)
    print("STAGE 3: Cleaning point cloud and extracting objects")
    print("=" * 60)

    del clean_ply_path  # historically selected the flow; only one flow remains
    try:
        object_paths = _segment_and_export(
            dense_ply=ply_path,
            output_dir=output_dir,
            num_objects=num_objects,
            seed=seed,
            segment_leg=segment_leg,
            segment_height_axis=segment_height_axis,
            fill_enabled=fill_enabled,
            apply_cut=apply_cut,
            marker_colour=marker_colour,
            override_planes=override_planes,
            cut_mode=cut_mode,
            n_bands=n_bands,
            band_planes=band_planes,
        )
        print(f"  Extracted {len(object_paths)} objects:")
        for p in object_paths:
            print(f"    → {p}")
        return object_paths
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ERROR during stage 3: {e}")
        return None

