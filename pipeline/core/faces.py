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
    """Unit normal, area and corner positions for every usable triangle.

    Degenerate triangles — zero area, or corners so close that the normal comes
    out non-finite — are dropped rather than carried with a fudged normal, since
    a meaningless direction would join whichever face cluster it happened to
    land nearest.
    """
    triangle_corners = vertices[triangles]
    cross_products = np.cross(triangle_corners[:, 1] - triangle_corners[:, 0],
                              triangle_corners[:, 2] - triangle_corners[:, 0])
    # |a x b| is twice the triangle's area.
    areas = np.linalg.norm(cross_products, axis=1) * 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = cross_products / np.maximum(
            np.linalg.norm(cross_products, axis=1, keepdims=True), 1e-12)
    usable = np.isfinite(normals).all(axis=1) & (areas > 0)
    return normals[usable], areas[usable], triangle_corners[usable]


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

    triangle_normals, triangle_areas, triangle_corners = \
        _triangle_normals_and_areas(vertices, triangles)
    if len(triangle_normals) == 0:
        return []
    total_area = float(triangle_areas.sum())
    triangle_centroids = triangle_corners.mean(axis=1)
    # Two normals count as the same face when their dot product clears this.
    same_face_cosine = np.cos(np.radians(tol_deg))

    unclaimed = np.ones(len(triangle_normals), dtype=bool)
    faces = []
    while unclaimed.any():
        # Seed on the direction carrying the most unclaimed area. Summing the
        # area of every candidate's neighbourhood is what makes this pick a
        # real face rather than one arbitrary large triangle.
        candidates = np.flatnonzero(unclaimed)
        if len(candidates) > 4000:
            # The similarity matrix below is candidates x unclaimed, so cap the
            # rows on a dense mesh. Largest-area first, since a real face is
            # never made only of slivers.
            candidates = candidates[np.argsort(-triangle_areas[candidates])[:4000]]
        similarity = triangle_normals[candidates] @ triangle_normals[unclaimed].T
        neighbourhood_area = (similarity >= same_face_cosine) @ triangle_areas[unclaimed]
        seed_triangle = candidates[int(np.argmax(neighbourhood_area))]

        on_this_face = unclaimed & (
            triangle_normals @ triangle_normals[seed_triangle] >= same_face_cosine)
        face_area = float(triangle_areas[on_this_face].sum())
        if face_area / total_area >= min_area_frac:
            face_normal = np.average(triangle_normals[on_this_face], axis=0,
                                     weights=triangle_areas[on_this_face])
            face_normal /= np.linalg.norm(face_normal) + 1e-12
            # Where the plane sits along its own normal.
            plane_offset = float(np.average(
                triangle_centroids[on_this_face] @ face_normal,
                weights=triangle_areas[on_this_face]))
            faces.append({"normal": face_normal, "offset": plane_offset,
                          "area": face_area, "n_tris": int(on_this_face.sum())})
        unclaimed &= ~on_this_face

    # Pair anti-parallel faces.
    #
    # Opposite faces are never *exactly* anti-parallel — on a real reconstructed
    # cube their normals differ by around a degree. Differencing the two plane
    # offsets then measures the gap where the planes would be if extended far
    # from the object, which on inputs/est_325 read 0.265 units against a true
    # 0.252. Measure the thickness through the mesh centroid instead: the sum of
    # the centroid's distances to the two planes. That is well defined however
    # the normals splay.
    mesh_centre = np.average(triangle_centroids, axis=0, weights=triangle_areas)
    pairs = []
    already_paired = set()
    for face_index, face in enumerate(faces):
        if face_index in already_paired:
            continue
        # The most anti-parallel partner, i.e. the smallest (most negative) dot
        # product. Starting the search at -same_face_cosine means a partner has
        # to be at least as anti-parallel as the tolerance allows.
        opposite_index, most_antiparallel = None, -same_face_cosine
        for other_index, other in enumerate(faces):
            if other_index == face_index or other_index in already_paired:
                continue
            alignment = float(face["normal"] @ other["normal"])
            if alignment < most_antiparallel:
                opposite_index, most_antiparallel = other_index, alignment
        if opposite_index is None:
            continue
        opposite = faces[opposite_index]
        already_paired.update({face_index, opposite_index})
        separation = (abs(face["offset"] - mesh_centre @ face["normal"]) +
                      abs(opposite["offset"] - mesh_centre @ opposite["normal"]))
        pairs.append({"axis": face["normal"], "separation": float(separation),
                      "area": face["area"] + opposite["area"],
                      "n_tris": face["n_tris"] + opposite["n_tris"]})

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
