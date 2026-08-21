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

    # ── Load ──
    pcd_dense = o3d.io.read_point_cloud(dense_ply)
    if len(pcd_dense.points) == 0:
        raise ValueError("Empty dense point cloud")
    initial_count = len(pcd_dense.points)
    print(f"Loaded dense cloud: {initial_count:,} points")

    # SOR
    if initial_count < 50000:
        std_ratio, nb_neighbors = 3.5, 10
    elif initial_count < 200000:
        std_ratio, nb_neighbors = 3.0, 15
    else:
        std_ratio, nb_neighbors = 2.5, 20
    pcd_dense, _ = pcd_dense.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"SOR: {initial_count:,} → {len(pcd_dense.points):,} (std_ratio={std_ratio})")

    if len(pcd_dense.points) > 100000:
        pcd_dense = pcd_dense.voxel_down_sample(voxel_size=0.002)
        print(f"Voxel downsample: → {len(pcd_dense.points):,}")

    # ── Phase A: Cluster both flows in original VGGT space ──
    print()
    print("─" * 40)
    print("PHASE A: Clustering (original VGGT space)")
    print("─" * 40)

    # A1: Remove floor from dense copy for DBSCAN
    pcd_cluster = o3d.geometry.PointCloud(pcd_dense)
    pcd_cluster, _ = remove_dominant_plane(pcd_cluster)

    # A2: DBSCAN on dense → box/obj clusters via cubeness
    box_dense, obj_dense = detect_top_k_objects(pcd_cluster, k=num_objects)
    if box_dense is None:
        box_dense = pcd_cluster
        obj_dense = None
        print("  Only 1 cluster on dense — treating as box")

    # A3: Marker detection on dense leg (if segment_leg enabled)
    markers = []
    if segment_leg and obj_dense is not None and len(obj_dense.points) > 0:
        try:
            if marker_colour:
                print(f"  Marker colour from Stage 0: "
                      f"RGB {[int(v) for v in marker_colour['rgb']]} "
                      f"(ExG {marker_colour.get('exg', 0):+.0f}) — "
                      f"thresholds follow the measured band, not the config "
                      f"defaults")
            _, summary = segment_point_cloud(obj_dense,
                                             height_axis=segment_height_axis,
                                             verbose=False,
                                             marker_colour=marker_colour)
            planes_raw = summary.get("planes", [])
            for _, centroid, normal, npts, _ in planes_raw:
                nrm = normal / (np.linalg.norm(normal) + 1e-8)
                markers.append({"centroid": np.array(centroid, dtype=np.float64),
                                "normal": np.array(nrm, dtype=np.float64),
                                "npts": int(npts)})
            print(f"  Dense leg markers: {len(markers)} plane(s) found")
        except Exception as e:
            print(f"  Marker detection skipped: {e}")

    # Save dense leg cluster and marker cut lines
    if obj_dense is not None and len(obj_dense.points) > 0:
        dlp = np.asarray(obj_dense.points, dtype=np.float32)
        _quick_save_ply(dlp, None, os.path.join(debug_dir_out, "leg_cluster.ply"))
    elif box_dense is not None and len(box_dense.points) > 0:
        dlp = np.asarray(box_dense.points, dtype=np.float32)
        _quick_save_ply(dlp, None, os.path.join(debug_dir_out, "leg_cluster.ply"))

    import json as _json
    cut_data = {"markers": []}
    for m in markers:
        cut_data["markers"].append({
            "centroid": m["centroid"].tolist(),
            "normal": m["normal"].tolist(),
            "npts": m.get("npts", 0),
        })
    with open(os.path.join(debug_dir_out, "cutting_line.json"), "w") as _f:
        _json.dump(cut_data, _f, indent=2)
    print("  Saved: debug/leg_cluster.ply + debug/cutting_line.json")

    # A4: Ghost filter each identified cluster separately.
    #
    # Filtering before clustering (the old flow) forced a second DBSCAN on a
    # ~14x sparser cloud and left the two results free to disagree about which
    # object is the box — nothing checked them. Splitting first keeps each
    # point's identity, so no label transfer is needed and the segmentation is
    # decided once, on the denser and better-conditioned cloud.
    #
    # The voxel size is computed once on the whole dense cloud: derived
    # per-cluster it would differ per object and give them inconsistent
    # densities.
    dense_pts_full = np.asarray(pcd_dense.points, dtype=np.float32)
    from pipeline.config import GHOST_VOXEL_FACTOR
    if GHOST_VOXEL_FACTOR <= 0:
        voxel_size = 0.0
        # Still needed as a length scale for the normal-aware search radius.
        nn_scale = compute_voxel_size(dense_pts_full, factor=1.0)
        print("  Ghost voxel dedup: DISABLED (GHOST_VOXEL_FACTOR=0) — keeping "
              "every point; ghost layers survive for the normal filter to catch")
    else:
        voxel_size = compute_voxel_size(dense_pts_full)
        nn_scale = voxel_size
        print(f"  Ghost voxel_size: {voxel_size:.4f} (shared across clusters)")

    def _ghost_filter(cluster, label):
        if cluster is None or len(cluster.points) == 0:
            return (np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.uint8))
        pts = np.asarray(cluster.points, dtype=np.float32)
        cols = (np.clip(np.asarray(cluster.colors, dtype=np.float32), 0, 1) * 255).astype(np.uint8)
        n_in = len(pts)
        pts, cols = voxel_dedup(pts, cols, voxel_size)
        pts, cols = normal_aware_filter(pts, cols, nn_scale)
        print(f"  Ghost filter [{label}]: {n_in:,} → {len(pts):,} pts")

        # Collapse the ghost sheet onto one surface. Runs per cluster so the
        # neighbourhood never spans two objects.
        from pipeline.config import MLS_RADIUS_MULT, MLS_BOX_POLYNOMIAL
        if MLS_RADIUS_MULT and MLS_RADIUS_MULT > 0 and len(pts) > 50:
            from pipeline.mls import mls_project
            # The reference is known to be planar; the limb is not.
            poly = MLS_BOX_POLYNOMIAL if label == "box" else True
            print(f"  MLS [{label}]:", end=" ")
            pts, cols, _ = mls_project(pts, cols,
                                       radius_mult=MLS_RADIUS_MULT,
                                       polynomial=poly)
        return pts, cols

    box_pts_arr, box_cols_arr = _ghost_filter(box_dense, "box")
    leg_pts_arr, leg_cols_arr = _ghost_filter(obj_dense, "obj")

    if obj_dense is None and len(box_pts_arr) > 0:
        print("  Only 1 cluster — treating it as the box (no obj)")

    print(f"  Clean: leg={len(leg_pts_arr):,}, box={len(box_pts_arr):,}")

    # ── Phase B: Leveling ──
    print()
    print("─" * 40)
    print("PHASE B: Leveling (shared transform)")
    print("─" * 40)

    ransac_thresh = auto_ransac_threshold(pcd_dense, base_factor=3)
    plane_model, _ = detect_plane_ransac_deterministic(
        pcd_dense, distance_threshold=ransac_thresh, num_iterations=1000, seed=seed)
    normal = plane_model[:3]
    print(f"RANSAC plane normal: ({normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f})")
    R = get_rotation_to_z_axis(normal)
    # Markers are detected in original VGGT space, so they must follow every
    # rotation the clouds receive — including the upside-down flip below.
    R_total = np.asarray(R, dtype=np.float64)

    pcd_dense.rotate(R, center=(0, 0, 0))
    dense_pts = np.asarray(pcd_dense.points, dtype=np.float32)

    leg_pts_rot = _rotate_points(leg_pts_arr, R)
    box_pts_rot = _rotate_points(box_pts_arr, R)

    # Upside-down check
    try:
        _, inliers_rot = detect_plane_ransac_deterministic(
            pcd_dense, distance_threshold=ransac_thresh, num_iterations=500, seed=seed)
        plane_z_val = np.mean(dense_pts[inliers_rot, 2])
        non_plane_mask = np.ones(len(dense_pts), dtype=bool)
        non_plane_mask[inliers_rot] = False
        if non_plane_mask.any() and np.mean(dense_pts[non_plane_mask, 2]) < plane_z_val:
            print("Upside-down detected — flipping 180°")
            flip_R = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))
            pcd_dense.rotate(flip_R, center=(0, 0, 0))
            dense_pts = np.asarray(pcd_dense.points, dtype=np.float32)
            leg_pts_rot = _rotate_points(leg_pts_rot, flip_R)
            box_pts_rot = _rotate_points(box_pts_rot, flip_R)
            R_total = np.asarray(flip_R, dtype=np.float64) @ R_total
    except Exception as e:
        print(f"Upside-down check failed: {e}")

    # ── Phase C: Post-leveling cuts + cap ──
    print()
    print("─" * 40)
    print("PHASE C: Cuts and export")
    print("─" * 40)

    # Floor cut on dense, apply to clean
    floor_z, floor_margin = None, 0.008
    try:
        r_thresh = auto_ransac_threshold(pcd_dense, base_factor=3)
        r_thresh = min(r_thresh, 0.015)
        plane_model2, inliers2 = detect_plane_ransac_deterministic(
            pcd_dense, distance_threshold=r_thresh, num_iterations=1000, seed=seed)
        normal2 = plane_model2[:3]
        is_horiz = abs(np.dot(normal2, [0, 0, 1])) > 0.85
        z_vals = dense_pts[:, 2]
        z_min, z_max = np.min(z_vals), np.max(z_vals)
        pz = np.median(dense_pts[inliers2, 2])
        is_bottom = (pz - z_min) < 0.25 * (z_max - z_min)
        ratio = len(inliers2) / len(dense_pts)
        if is_horiz and is_bottom and ratio > 0.05:
            floor_z = pz
            print(f"Floor plane at Z={floor_z:.4f} (ratio={ratio:.1%})")
        else:
            print(f"Floor cut skipped (horiz={is_horiz}, bottom={is_bottom}, ratio={ratio:.1%})")
    except Exception as e:
        print(f"Floor detection failed: {e}")

    if floor_z is not None and len(leg_pts_rot) > 0:
        lk = leg_pts_rot[:, 2] > (floor_z + floor_margin)
        leg_pts_rot = leg_pts_rot[lk]; leg_cols_arr = leg_cols_arr[lk]
    if floor_z is not None and len(box_pts_rot) > 0:
        bk = box_pts_rot[:, 2] > (floor_z + floor_margin)
        box_pts_rot = box_pts_rot[bk]; box_cols_arr = box_cols_arr[bk]
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
