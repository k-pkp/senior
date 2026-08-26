"""Measure the reference cube with the markers printed on it.

Stage 6 derives scale from the cube's own fitted faces, so the cube always
measures REFERENCE_REAL_SIZE_CM whatever it really is, and the residual it
reports is a shape error that cannot see a wrong constant or a warped
reconstruction. The printed markers break that circularity: they are a second
structure in the scene that the calibration never touches.

They are also unusually good probes. ArUco locates corners to sub-pixel
accuracy, and those corner pixels index the pointmap directly — the frames in
predictions.npz are the exact tensors VGGT consumed — so a marker becomes a 3D
quadrilateral with no colour thresholding, no meshing and no coordinate
mapping between image and cloud.

Most of what that buys needs no physical constant at all. A printed marker is
flat and square, so any departure from flat or from square is reconstruction
error and nothing else. Measured on the two datasets:

    flatness     0.04-0.45 mm     the surface is locally accurate
    aspect       1.04-1.11        a square reconstructs 8% out of square
    diag ratio   0.999            so it is a rectangle, not a shear
    size spread  3.7-3.9%         one physical square, five faces

Only absolute size needs a ruler, and it is worth having: on inputs/small_leg
the photographs (rectified through the marker's own homography, no
reconstruction involved) put the printed square at 6.49-6.58 cm while the
reconstruction reads 6.325, a real -2.5 to -3.8% geometric error against a
length the calibration never used.
"""
import numpy as np

# ArUco family the reference cube carries. Confirmed against both datasets:
# ids 10-14, one per visible face, detected on every frame.
DEFAULT_DICT = "DICT_5X5_250"

# Minimum world_points_conf at all four corners. The pointmap is unreliable
# where confidence is low, and a single bad corner distorts every quantity
# derived from the quad. 1.5 keeps every marker in both datasets.
MIN_CONF = 1.5

# Reject a view whose marker is too foreshortened. A quad seen edge-on has a
# tiny image footprint, so its corners land within a pixel or two of each other
# and the plane through them is dominated by noise — this is what made naive
# normal averaging report impossible face geometry on est_325.
MIN_QUAD_PIXELS = 20.0


def _gray(frame):
    """One 518x518 uint8 image from the (3, H, W) float tensor VGGT was given."""
    import cv2
    rgb = (np.transpose(frame, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def detect_marker_quads(predictions, min_conf=MIN_CONF, dict_name=DEFAULT_DICT):
    """Lift every detected marker into 3D.

    Returns {marker_id: [record, ...]}, one record per view, each holding the
    four 3D corners in pointmap units along with the frame it came from and its
    image-space size. Corners are kept per view rather than averaged: the views
    disagree, and that disagreement is signal.
    """
    import cv2

    images = predictions["images"]
    world = predictions["world_points"]
    conf = predictions["world_points_conf"]
    h, w = world.shape[1:3]

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    out = {}
    for frame in range(images.shape[0]):
        corners, ids, _ = detector.detectMarkers(_gray(images[frame]))
        if ids is None:
            continue
        for quad, marker_id in zip(corners, ids.ravel()):
            px = quad.reshape(4, 2)
            side_px = float(np.mean([np.linalg.norm(px[i] - px[(i + 1) % 4])
                                     for i in range(4)]))
            if side_px < MIN_QUAD_PIXELS:
                continue
            u = np.round(px[:, 0]).astype(int)
            v = np.round(px[:, 1]).astype(int)
            if u.min() < 0 or v.min() < 0 or u.max() >= w or v.max() >= h:
                continue
            if float(conf[frame, v, u].min()) < min_conf:
                continue
            out.setdefault(int(marker_id), []).append({
                "frame": frame,
                "side_px": side_px,
                "corners": np.asarray(world[frame, v, u], dtype=np.float64),
            })
    return out


def quad_metrics(corners, scale=1.0):
    """Shape of one reconstructed marker, in whatever unit `scale` converts to.

    The marker is physically a flat square, so `aspect` and `flatness` are pure
    error terms and `diag_ratio` distinguishes a rectangle (sides unequal,
    diagonals still consistent) from a shear.
    """
    p = np.asarray(corners, dtype=np.float64) * scale
    sides = np.array([np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)])
    diags = np.array([np.linalg.norm(p[0] - p[2]), np.linalg.norm(p[1] - p[3])])
    centred = p - p.mean(axis=0)
    normal = np.linalg.svd(centred)[2][2]
    return {
        "sides": sides,
        "side": float(sides.mean()),
        "aspect": float(sides.max() / sides.min()),
        "diag_ratio": float(diags.mean() / (sides.mean() * np.sqrt(2.0))),
        "flatness": float(np.abs(centred @ normal).max()),
        "normal": normal,
        "centre": p.mean(axis=0),
    }


def edge_lengths_by_axis(corners, up, scale=1.0, cos_tol=0.7):
    """Split a marker's four edges into those along `up` and those across it.

    A printed square has no long axis, so if its reconstruction stretches along
    one scene direction that is an anisotropic scale error and it transfers to
    every other measurement the pipeline makes.
    """
    p = np.asarray(corners, dtype=np.float64) * scale
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    along, across = [], []
    for i in range(4):
        edge = p[(i + 1) % 4] - p[i]
        length = float(np.linalg.norm(edge))
        (along if abs(edge @ up) / length > cos_tol else across).append(length)
    return along, across


def infer_up_axis(quads):
    """The cube's vertical, from the marker layout alone.

    Only five faces carry markers; the sixth rests on the floor. So the top
    face is the one marked face with no antiparallel partner, and its normal is
    the vertical. This is a fallback for when the stage-3 levelling rotation is
    unavailable, and it is not always decisive — on est_325 every marker had a
    near-parallel partner, which cannot happen on a cube and means the normals
    there are too poorly conditioned to trust. Returns None in that case rather
    than a wrong axis.
    """
    normals = {}
    for marker_id, views in quads.items():
        per_view = [quad_metrics(v["corners"])["normal"] for v in views]
        # Sign is arbitrary per view; align them before averaging or they cancel.
        ref = per_view[0]
        aligned = [n if n @ ref >= 0 else -n for n in per_view]
        mean = np.mean(aligned, axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            normals[marker_id] = mean / norm
    if len(normals) < 3:
        return None, None
    partner = {k: max(abs(normals[k] @ normals[j]) for j in normals if j != k)
               for k in normals}
    top = min(partner, key=partner.get)
    # A genuine top face is perpendicular to all four sides; anything close to
    # parallel means the normals are noise and the answer would be arbitrary.
    if partner[top] > 0.5:
        return None, None
    return normals[top], top
