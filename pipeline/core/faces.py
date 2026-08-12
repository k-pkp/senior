"""Measure a box-shaped reference by fitting its own faces.

Stage 6 needs one number from the reference cube: its edge length. That number
sets the scale for every result the pipeline produces.

Taking it from an oriented bounding box is unreliable, and measurably so. On
inputs/est_325 the OBB around the reference exceeded the convex hull of its own
points by 6.8% in volume — an OBB must *contain* the hull, so that excess is
pure fitting error. Spread over three axes it is +2.2% per edge, consistent with
the box being mis-rotated by roughly 1.3 degrees. On a cube that lands directly
on the edge length:

    face-to-face box       2505.7 cm3     the cube's real size
    convex hull of points  2479.2 cm3     -1.1%
    oriented bounding box  2647.8 cm3     +5.7%

A cube does not need a bounding box to be measured. Its face normals *are* its
axes, so fitting the faces recovers the orientation exactly instead of hoping a
hull-based algorithm finds it. Measured effect on the reference's own volume
error: -10.7% -> -4.7%.

This module is for box-shaped references only. It makes no sense for a limb and
is never applied to one.
"""
import numpy as np

# A face must hold at least this share of total surface area to be trusted.
# Corner and edge triangles are numerous but tiny; area weighting means they
# cannot outvote a real face, and this floor keeps a sliver from being fitted
# as though it were one.
MIN_FACE_AREA_FRAC = 0.02

# Two normals belong to the same face below this angle.
FACE_NORMAL_TOL_DEG = 25.0


def _triangle_normals_and_areas(vertices, triangles):
    v = vertices[triangles]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    area = np.linalg.norm(n, axis=1) * 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    ok = np.isfinite(n).all(axis=1) & (area > 0)
    return n[ok], area[ok], v[ok]


def fit_box_faces(vertices, triangles,
                  min_area_frac=MIN_FACE_AREA_FRAC,
                  tol_deg=FACE_NORMAL_TOL_DEG):
    """Group a box mesh's triangles into faces and measure opposite separations.

    Greedy area-weighted clustering on the unit normal sphere: repeatedly take
    the largest remaining area direction, absorb everything within `tol_deg` of
    it, and fit a plane offset as the area-weighted mean of the face's triangle
    centroids projected on that normal.

    Faces are then paired by anti-parallel normals. Separation is the distance
    between the two plane offsets along the shared axis.

    Returns a list of dicts, one per opposite-face pair, sorted by area:
        {"axis": unit normal (3,), "separation": float,
         "area": combined area, "n_tris": int}
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if len(triangles) < 12:
        return []

    n, area, tri = _triangle_normals_and_areas(vertices, triangles)
    if len(n) == 0:
        return []
    total = float(area.sum())
    centroid = tri.mean(axis=1)
    cos_tol = np.cos(np.radians(tol_deg))

    remaining = np.ones(len(n), dtype=bool)
    faces = []
    while remaining.any():
        # Seed on the direction carrying the most unclaimed area. Summing the
        # area of every candidate's neighbourhood is what makes this pick a
        # real face rather than one arbitrary large triangle.
        idx = np.flatnonzero(remaining)
        if len(idx) > 4000:
            idx = idx[np.argsort(-area[idx])[:4000]]
        sims = n[idx] @ n[remaining].T
        weight = (sims >= cos_tol) @ area[remaining]
        seed = idx[int(np.argmax(weight))]

        member = remaining & (n @ n[seed] >= cos_tol)
        a = float(area[member].sum())
        if a / total >= min_area_frac:
            axis = np.average(n[member], axis=0, weights=area[member])
            axis /= np.linalg.norm(axis) + 1e-12
            offset = float(np.average(centroid[member] @ axis,
                                      weights=area[member]))
            faces.append({"normal": axis, "offset": offset,
                          "area": a, "n_tris": int(member.sum())})
        remaining &= ~member

    # Pair anti-parallel faces.
    #
    # Opposite faces are never *exactly* anti-parallel — on a real reconstructed
    # cube their normals differ by around a degree. Differencing the two plane
    # offsets then measures the gap where the planes would be if extended far
    # from the object, which on inputs/est_325 read 0.265 units against a true
    # 0.252. Measure the thickness through the mesh centroid instead: the sum of
    # the centroid's distances to the two planes. That is well defined however
    # the normals splay.
    centre = np.average(centroid, axis=0, weights=area)
    pairs = []
    used = set()
    for i, f in enumerate(faces):
        if i in used:
            continue
        best, best_dot = None, -cos_tol
        for j, g in enumerate(faces):
            if j == i or j in used:
                continue
            d = float(f["normal"] @ g["normal"])
            if d < best_dot:
                best, best_dot = j, d
        if best is None:
            continue
        g = faces[best]
        used.update({i, best})
        sep = (abs(f["offset"] - centre @ f["normal"]) +
               abs(g["offset"] - centre @ g["normal"]))
        pairs.append({"axis": f["normal"], "separation": float(sep),
                      "area": f["area"] + g["area"],
                      "n_tris": f["n_tris"] + g["n_tris"]})

    pairs.sort(key=lambda p: -p["area"])
    return pairs


def reference_edges(vertices, triangles, up=(0.0, 0.0, 1.0)):
    """Edge lengths of a box reference, split into vertical and horizontal.

    Returns (vertical_or_None, [horizontal...], diagnostics dict). The vertical
    pair is the one whose axis is most aligned with `up`; it is reported but the
    caller should exclude it from the scale, because the reference rests on the
    floor and its underside is fabricated by Stage 3's floor extension.
    """
    pairs = fit_box_faces(vertices, triangles)
    diag = {"n_pairs": len(pairs),
            "pairs": [(p["separation"], p["n_tris"]) for p in pairs]}
    if not pairs:
        return None, [], diag

    up = np.asarray(up, dtype=np.float64)
    up = up / (np.linalg.norm(up) + 1e-12)
    align = [abs(float(p["axis"] @ up)) for p in pairs]
    vi = int(np.argmax(align))
    vertical = pairs[vi]["separation"] if align[vi] > 0.7 else None
    horiz = [p["separation"] for i, p in enumerate(pairs)
             if i != vi or vertical is None]
    diag["vertical_alignment"] = align[vi]
    return vertical, horiz, diag
