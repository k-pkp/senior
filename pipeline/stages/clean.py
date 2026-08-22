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
                                        MAX_MARKERS)
from pipeline.config import MARKER_MIN_HEIGHT_FRAC
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
    from pipeline.ghost import voxel_dedup, normal_aware_filter

    if cluster is None or len(cluster.points) == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8))
    points = np.asarray(cluster.points, dtype=np.float32)
    colours = (np.clip(np.asarray(cluster.colors, dtype=np.float32), 0, 1)
               * 255).astype(np.uint8)
    before = len(points)
    points, colours = voxel_dedup(points, colours, voxel_size)
    points, colours = normal_aware_filter(points, colours, normal_filter_scale)
    print(f"  Ghost filter [{label}]: {before:,} → {len(points):,} pts")

    if MLS_RADIUS_MULT and MLS_RADIUS_MULT > 0 and len(points) > 50:
        from pipeline.mls import mls_project
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
                            override_planes=None):
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
    from pipeline.ghost import voxel_dedup, normal_aware_filter, compute_voxel_size

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
    leg_no_cut_path = os.path.join(objects_dir, "leg_no_cut.ply")
    markers_rotated = []
    if len(markers) > 0:
        # A limb segment is bounded by at most two cuts, so only the two
        # best-supported markers can matter; a third can only contradict one of
        # them. Trim here rather than in the cut so the published planes, the
        # cross-section caps and the cut all see the same set.
        if len(markers) > MAX_MARKERS:
            markers = sorted(markers, key=lambda m: -m.get("npts", 0))[:MAX_MARKERS]
            print(f"Markers: keeping the {MAX_MARKERS} best-supported")
        for m in markers:
            cen = np.array(m["centroid"])
            norm = np.array(m["normal"])
            cen_r = (R_total @ cen).astype(np.float64)
            norm_r = (R_total @ norm).astype(np.float64)
            norm_r /= np.linalg.norm(norm_r) + 1e-8
            markers_rotated.append({"centroid": cen_r.tolist(), "normal": norm_r.tolist(),
                                    "npts": m.get("npts", 0)})

        # Drop planes sitting in the bottom fraction of the object. Feet, arch
        # shadows and the floor junction all live there, and a cut line that low
        # would discard nearly the whole limb.
        #
        # This has to happen HERE rather than in segment_point_cloud: detection
        # runs in original VGGT space, where the vertical axis is whatever the
        # camera happened to give (on small_leg the limb's long axis is Y, not
        # Z). Only after R_total does "height" mean height.
        frac = MARKER_MIN_HEIGHT_FRAC
        if frac > 0 and len(leg_pts_rot) > 0 and len(markers_rotated) > 0:
            lo = float(leg_pts_rot[:, 2].min())
            span = float(leg_pts_rot[:, 2].max()) - lo
            if span > 0:
                keep_m, drop_m = [], []
                for mr in markers_rotated:
                    h = (mr["centroid"][2] - lo) / span
                    (keep_m if h >= frac else drop_m).append((mr, h))
                for mr, h in drop_m:
                    print(f"  Marker rejected: {h*100:.0f}% of height "
                          f"(< {frac*100:.0f}%), {mr['npts']} pts")
                markers_rotated = [mr for mr, _ in keep_m]

        # Also publish the planes in LEVELLED space. cutting_line.json is written
        # pre-levelling, but leg_no_cut.ply is post-levelling, so the two do not
        # share a frame — anything drawing them together (a UI, a debug viewer)
        # needs this version, and "height along Z" only means something here.
        with open(os.path.join(debug_dir_out, "cutting_line_levelled.json"), "w") as _f:
            _json.dump({"markers": markers_rotated, "space": "levelled"}, _f, indent=2)

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
        with open(os.path.join(debug_dir_out, "cutting_line_review.json"), "w") as _f:
            _json.dump({"markers": markers_rotated, "space": "levelled",
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

    # The cut operates on the OPEN cloud — floor-cut but not yet floor-closed —
    # and only then closes what the cut leaves open. Saving it is what lets a
    # deferred cut reproduce this pass exactly instead of approximating it by
    # cutting the closed cloud, which carries a fabricated skirt the real path
    # never cuts through.
    _quick_save_ply(leg_pts_rot, leg_cols_arr,
                    os.path.join(objects_dir, "leg_open.ply"))

    # Complete but uncut — the review cloud. Always floor-closed.
    nc_pts, nc_cols = _close_to_floor(leg_pts_rot, leg_cols_arr)
    _quick_save_ply(nc_pts, nc_cols, leg_no_cut_path)
    print(f"Saved (complete, uncut): {leg_no_cut_path} ({len(nc_pts):,} pts)")

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

    # leg_cut + box
    #
    # With the cut deferred there is deliberately no leg_cut.ply. Writing the
    # uncut cloud under that name would send stages 4-6 off to reconstruct and
    # integrate a limb whose extent nobody has agreed to yet, and every second
    # of that is thrown away the moment the cut is confirmed. The reference cube
    # still goes through, because its measurement does not depend on the cut and
    # the review needs its edge length to show anything in centimetres.
    output_paths = []
    if apply_cut and len(o3d_leg.points) > 0:
        o3d.io.write_point_cloud(leg_cut_path, o3d_leg)
        output_paths.append(leg_cut_path)
        print(f"Saved: {leg_cut_path} ({len(o3d_leg.points):,} pts)")
    elif not apply_cut:
        if os.path.exists(leg_cut_path):
            os.remove(leg_cut_path)   # a stale one would be measured instead
        print("Deferred: leg_cut.ply not written — awaiting the confirmed cut")
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
                      fill_enabled=True, clean_ply_path=None, apply_cut=True,
                      marker_colour=None, override_planes=None):
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


def cut_only(stage3_dir, planes, fill_enabled=True):
    """Apply confirmed cutting planes to a stage 3 output that deferred its cut.

    This is the second half of a split Stage 3. The first half did all the
    expensive work — SOR, RANSAC, DBSCAN, ghost filtering, MLS, levelling,
    marker detection — and saved the levelled limb in the exact state the cut
    operates on (`leg_open.ply`: floor-cut, not yet floor-closed). None of that
    depends on where the cut goes, so none of it is repeated here.

    The steps below are the same ones, in the same order, that
    `_segment_and_export` runs after its own detection. They have to be: the
    whole point of splitting is that confirming the detected plane must give
    the identical answer to never having split at all.

    `planes` are {"centroid", "normal"} in LEVELLED space. An empty list means
    the user chose to measure the object whole.
    """
    import json as _json

    objects_dir = os.path.join(stage3_dir, "objects")
    debug_dir_out = os.path.join(stage3_dir, "debug")
    open_path = os.path.join(objects_dir, "leg_open.ply")
    box_path = os.path.join(objects_dir, "box.ply")
    if not os.path.exists(open_path):
        raise SystemExit(
            f"ERROR: {open_path} missing — run stage 3 with --no-cut first, so "
            f"there is a detected-but-uncut result to apply a cut to")

    with open(os.path.join(debug_dir_out, "levelling.json")) as f:
        floor_z = _json.load(f).get("floor_z")

    pcd = o3d.io.read_point_cloud(open_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = ((np.clip(np.asarray(pcd.colors, dtype=np.float32), 0, 1) * 255)
            .astype(np.uint8) if pcd.has_colors() else None)
    print(f"Loaded uncut limb: {len(pts):,} pts")

    markers = []
    for m in list(planes)[:MAX_MARKERS]:
        n = np.asarray(m["normal"], dtype=np.float64)
        n = n / (np.linalg.norm(n) + 1e-8)
        markers.append({"centroid": np.asarray(m["centroid"], dtype=np.float64).tolist(),
                        "normal": n.tolist(), "npts": int(m.get("npts", 0))})

    def _close_to_floor(p_pts, p_cols):
        if not fill_enabled or floor_z is None or len(p_pts) == 0:
            return p_pts, p_cols
        p = _array_to_o3d(p_pts, p_cols)
        p = extend_point_cloud_to_floor(p, floor_z, label="obj")
        p = cap_point_cloud_bottom(p, alpha=2.0)
        out = np.asarray(p.points, dtype=np.float32)
        oc = ((np.clip(np.asarray(p.colors, dtype=np.float32), 0, 1) * 255)
              .astype(np.uint8) if p.has_colors() else p_cols)
        return out, oc

    if markers:
        keep, case = apply_marker_cut(pts.astype(np.float64), markers)
        if int(keep.sum()) >= 50:
            pts, cols = pts[keep], (None if cols is None else cols[keep])
            print(f"Marker cut ({case}): leg → {len(pts):,} pts")
            if fill_enabled:
                for i, mr in enumerate(markers):
                    pts, cols = cap_points_on_plane(pts, cols, mr["centroid"],
                                                    mr["normal"], label=f"marker {i}")
                pts = pts.astype(np.float32)
            if len(markers) < 2:
                pts, cols = _close_to_floor(pts, cols)
        else:
            print(f"Marker cut ({case}): only {int(keep.sum())} pts (< 50) — "
                  f"keeping the whole limb")
            pts, cols = _close_to_floor(pts, cols)
    else:
        print("No cutting planes supplied — measuring the limb whole")
        pts, cols = _close_to_floor(pts, cols)

    with open(os.path.join(debug_dir_out, "cutting_line_review.json"), "w") as f:
        _json.dump({"markers": markers, "space": "levelled", "source": "review"},
                   f, indent=2)

    leg_cut_path = os.path.join(objects_dir, "leg_cut.ply")
    o3d_leg = _array_to_o3d(pts, cols)
    o3d.io.write_point_cloud(leg_cut_path, o3d_leg)
    print(f"Saved: {leg_cut_path} ({len(pts):,} pts)")

    out = [leg_cut_path]
    if os.path.exists(box_path):
        out.append(box_path)
        box = o3d.io.read_point_cloud(box_path)
        merged = o3d.geometry.PointCloud()
        merged.points = o3d.utility.Vector3dVector(
            np.vstack([np.asarray(o3d_leg.points), np.asarray(box.points)]))
        if o3d_leg.has_colors() and box.has_colors():
            merged.colors = o3d.utility.Vector3dVector(
                np.vstack([np.asarray(o3d_leg.colors), np.asarray(box.colors)]))
        o3d.io.write_point_cloud(os.path.join(objects_dir, "merged.ply"), merged)
    return out
