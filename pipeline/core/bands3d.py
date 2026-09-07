"""Marker planes from Stage 0's band boxes, through Stage 1's own pointmap.

Stage 3 finds the cut by colour: it selects the band's points out of the dense
cloud and fits a plane to them. That works when the cord is chromatically far
enough from the limb to be separable, and fails silently when it is not -- and
"not" is common. On the job that prompted this module the learned band/limb
separation was 0.0259 against a floor of MARKER_MIN_AXIS = 0.05, so the learned
colour was refused, the config's khaki window matched 130 points on the entire
limb, and the lower of the subject's two cords survived as an 11-point cluster
that MARKER_MIN_CLUSTER_PTS then dropped. Stage 0 had detected that same cord on
6 of 7 photographs.

So the information exists; it is just in the wrong space. Stage 0 records each
band's box in frame pixels, and Stage 1's `world_points` is a 3D position for
every pixel of every frame it was given. Mapping one through the other turns a
2D detection into thousands of 3D points, without asking any colour question at
all -- which is the point, since colour is what failed.

What this does NOT do is replace colour detection. It is a second, independent
source, merged with the first in Stage 3, so a capture can only ever gain a
plane here. The band boxes come from a detector prompted with "a rubber band on
the leg", so a plane fitted here means the photographs showed a band there.

The pixel mapping has two cases, and getting them backwards puts the plane
somewhere else entirely:

  cropped frames    Stage 0 wrote `image[top:bottom, left:right]` resized to
                    518x518, so the map is linear from that window.
  uncropped frames  Stage 0 wrote the frame whole and VGGT cropped it itself:
                    width is resized to 518, the height to the nearest multiple
                    of 14, and the centre 518 rows are kept.
"""
import json
import os

import numpy as np
from sklearn.cluster import DBSCAN

AXIS_MAP = {"x": 0, "y": 1, "z": 2}

# Keep only the middle of each band box along the frame's vertical. The box is
# drawn around the cord with slack on both sides, and its outer rows are limb
# surface above and below the cord rather than the ring itself. 0.6 keeps the
# cord's own rows without narrowing the ring so far that it stops spanning the
# limb, which is what makes the plane fit well conditioned.
BOX_CORE_FRAC = 0.6

# Confidence percentile, within each frame, below which a pixel's 3D position is
# not used. VGGT reports low confidence on exactly the surfaces that reconstruct
# badly, and a plane fitted through those is worse than no plane.
CONF_PCT = 60.0

# DBSCAN radius for separating one band from another, in VGGT world units. The
# reference cube is ~0.13 units across for a 10 cm cube, so 0.03 is about 2.3 cm
# -- wider than the scatter on one cord, far narrower than the gap between two
# cords tied at the ends of a measured segment.
CLUSTER_EPS = 0.03
CLUSTER_MIN_SAMPLES = 10

# A cluster has to be seen from several viewpoints. One frame's worth of pixels
# can be a detector's mistake carried into 3D; the same band seen from several
# angles cannot be. This mirrors the frame-agreement rule Stage 0 already
# applies to the band colour and the band count.
MIN_FRAMES = 2
MIN_FRAME_FRAC = 0.5

# Most bands a capture can be cut against — the same two as everywhere else.
MAX_BANDS = 2


def _frame_to_input_box(record, input_size=518, patch=14):
    """Where this frame's pixels ended up in the array VGGT consumed.

    Returns (box, scale_x, scale_y, offset_y) such that an original-frame pixel
    (x, y) maps to ((x - box[0]) * scale_x, (y - box[1]) * scale_y - offset_y),
    or None when the frame cannot be mapped.
    """
    size = record.get("frame_size")
    if not size:
        return None
    width, height = float(size[0]), float(size[1])
    mode = record.get("mode") or ""
    cropped = mode.startswith("crop") and "uncropped" not in mode

    if cropped:
        window = record.get("window")
        if not window:
            return None
        left, top, right, bottom = (float(v) for v in window)
        if right - left <= 1 or bottom - top <= 1:
            return None
        return ([left, top], input_size / (right - left),
                input_size / (bottom - top), 0.0)

    # VGGT's own preprocessing: width to `input_size`, height to the nearest
    # multiple of `patch`, then the centre `input_size` rows.
    scaled_height = round(height * (input_size / width) / patch) * patch
    if scaled_height <= input_size:
        # Nothing is cropped away, and the array is not square. Every capture in
        # hand is 9:16, so this is unreached; refusing beats guessing at a
        # mapping that has never been checked against a real frame.
        return None
    start = (scaled_height - input_size) // 2
    return ([0.0, 0.0], input_size / width, scaled_height / height, float(start))


def _band_pixels(record, band_box, input_size=518):
    """Pixel bounds of one band box inside the frame VGGT consumed."""
    mapped = _frame_to_input_box(record, input_size)
    if mapped is None:
        return None
    (origin, scale_x, scale_y, offset_y) = mapped

    x0, y0, x1, y1 = (float(v) for v in band_box)
    # Keep the cord's own rows, not the slack around them.
    centre = (y0 + y1) / 2.0
    half = (y1 - y0) * BOX_CORE_FRAC / 2.0
    y0, y1 = centre - half, centre + half

    u0 = (x0 - origin[0]) * scale_x
    u1 = (x1 - origin[0]) * scale_x
    v0 = (y0 - origin[1]) * scale_y - offset_y
    v1 = (y1 - origin[1]) * scale_y - offset_y

    u0, u1 = sorted((int(np.floor(u0)), int(np.ceil(u1))))
    v0, v1 = sorted((int(np.floor(v0)), int(np.ceil(v1))))
    u0, v0 = max(0, u0), max(0, v0)
    u1, v1 = min(input_size, u1), min(input_size, v1)
    if u1 - u0 < 2 or v1 - v0 < 2:
        return None
    return u0, v0, u1, v1


def _fit_plane(points, axis_idx):
    """Centroid and unit normal of the best-fit plane, normal along `axis_idx`."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    if normal[axis_idx] < 0:
        normal = -normal
    return centroid, normal / (np.linalg.norm(normal) + 1e-8)


def band_planes_from_arrays(manifest, world_points, world_points_conf,
                            height_axis="z", verbose=True):
    """Marker planes in ORIGINAL VGGT space, from Stage 0's band boxes.

    Same space and same shape as `clean._detect_marker_planes` returns, so the
    two are merged by the caller and levelled together.

    Returns [] — never raises — whenever the inputs cannot be related: a run
    with no Stage 0, a manifest from before multi-band detection, a frame count
    that does not match the pointmap. A missing plane leaves the colour path
    exactly as it was; a wrong one would move the cut.
    """
    axis_idx = AXIS_MAP[height_axis.lower()]
    try:
        records = [r for r in manifest.get("frames", []) if r.get("output")]
    except AttributeError:
        return []
    if not records:
        return []

    points = np.asarray(world_points)
    conf = np.asarray(world_points_conf)
    if points.ndim != 4 or len(points) != len(records):
        if verbose:
            print(f"  band projection skipped: {len(points)} pointmaps for "
                  f"{len(records)} framed images — they cannot be paired")
        return []

    input_size = points.shape[1]
    collected, frame_ids = [], []
    frames_with_bands = 0
    for frame_index, record in enumerate(records):
        boxes = record.get("band_bboxes") or []
        if not boxes:
            continue
        frames_with_bands += 1
        frame_conf = conf[frame_index]
        floor = float(np.percentile(frame_conf, CONF_PCT))
        for box in boxes:
            bounds = _band_pixels(record, box, input_size)
            if bounds is None:
                continue
            u0, v0, u1, v1 = bounds
            patch = points[frame_index, v0:v1, u0:u1].reshape(-1, 3)
            patch_conf = frame_conf[v0:v1, u0:u1].reshape(-1)
            keep = patch_conf >= floor
            if keep.sum() < 4:
                continue
            collected.append(patch[keep])
            frame_ids.append(np.full(int(keep.sum()), frame_index, dtype=np.int32))

    if not collected:
        if verbose and frames_with_bands:
            print("  band projection: no confident 3D points under any band box")
        return []

    coords = np.concatenate(collected).astype(np.float64)
    owners = np.concatenate(frame_ids)
    finite = np.isfinite(coords).all(axis=1)
    coords, owners = coords[finite], owners[finite]
    if len(coords) < CLUSTER_MIN_SAMPLES:
        return []

    # Cluster in 3D rather than pairing boxes frame by frame. A frame that saw
    # only the lower cord makes its box the first in its own list, so pairing by
    # position across frames merges two different bands into one plane. Where
    # the points land in space does not have that failure.
    labels = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES).fit(coords).labels_

    needed_frames = max(MIN_FRAMES,
                        int(np.ceil(MIN_FRAME_FRAC * max(frames_with_bands, 1))))
    planes = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        selected = labels == cluster_id
        seen_in = len(set(owners[selected].tolist()))
        if seen_in < needed_frames:
            if verbose:
                print(f"  band projection: cluster of {int(selected.sum()):,} pts "
                      f"seen in {seen_in} frame(s), {needed_frames} needed")
            continue
        centroid, normal = _fit_plane(coords[selected], axis_idx)
        planes.append({"centroid": centroid, "normal": normal,
                       "npts": int(selected.sum()), "frames": seen_in,
                       "source": "projected"})

    planes.sort(key=lambda p: (-p["frames"], -p["npts"]))
    dropped = planes[MAX_BANDS:]
    planes = planes[:MAX_BANDS]
    for plane in dropped:
        if verbose:
            print(f"  band projection: dropped a {plane['npts']:,}-pt cluster — "
                  f"a cut is bounded by at most {MAX_BANDS} planes")
    planes.sort(key=lambda p: p["centroid"][axis_idx])

    if verbose and planes:
        detail = ", ".join(f"{p['npts']:,} pts from {p['frames']} frames"
                           for p in planes)
        print(f"  Band projection: {len(planes)} plane(s) from Stage 0's boxes "
              f"({detail})")
    return planes


def band_planes_from_dirs(prep_dir, predictions_path, height_axis="z",
                          verbose=True):
    """`band_planes_from_arrays`, reading the manifest and pointmap off disk."""
    manifest_path = os.path.join(prep_dir, "manifest.json")
    if not (os.path.exists(manifest_path) and os.path.exists(predictions_path)):
        return []
    try:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        with np.load(predictions_path) as predictions:
            if "world_points" not in predictions:
                return []
            return band_planes_from_arrays(
                manifest, predictions["world_points"],
                predictions["world_points_conf"], height_axis, verbose)
    except (OSError, ValueError, KeyError) as exc:
        if verbose:
            print(f"  band projection skipped: {type(exc).__name__}: {exc}")
        return []
