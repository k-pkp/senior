"""Apply the confirmed marker planes to a watertight mesh.

This is the mesh counterpart of `core/segmentation.py:apply_marker_cut`, which
does the same job on a point cloud. The rule is identical and stated the same
way, in terms of above and below rather than sides:

    0 markers -> no cut
    1 marker  -> keep what is BELOW the plane
    2 markers -> keep what lies BETWEEN the two planes

Cutting the solid rather than the cloud is what lets a clinician place the cut
on a surface instead of a scatter of points. It also removes the reconstruction
step that used to sit between the cut and the measurement: the cloud cut had to
cap the exposed cross-section and then hand it to Poisson, which reconstructs
that flat face along with everything else. Slicing a closed mesh produces the
face directly, and nothing runs after it that could round it off.

The input must be watertight. A plane cut caps cleanly only when the
cross-section is a closed loop, which an open surface cannot guarantee -- on a
Stage 4 mesh the cap collapsed to a single triangle, and the result stayed
non-manifold (euler -9 before, -8 after). Stage 5's repaired mesh goes in
watertight with euler 2 and comes out the same way, which is what Stage 6's
exact signed-volume method requires.
"""
import numpy as np

# A limb segment is bounded by at most two cuts. A third plane can only
# contradict one of them, so extra planes are dropped rather than combined.
MAX_MARKERS = 2

# Below this the plane stands vertical and has no above or below, so the
# above/below rule is undefined and the plane is skipped instead of guessed at.
MIN_VERTICAL_COMPONENT = 1e-3

# A cut that leaves almost nothing behind is a failed cut, not a small limb.
# Matches the cloud path's floor of 50 points.
MIN_FACES_AFTER_CUT = 50


def _upward_plane(marker, up_axis):
    """Returns one marker as (origin, upward normal), or None if it stands vertical.

    The detected normal may point either way, so it is flipped to point along
    `up_axis`. That makes the sign of the detection irrelevant to the outcome, which
    is what lets a reviewer drag a plane without inverting the selection.
    """
    origin = np.asarray(marker["centroid"], dtype=np.float64)
    normal = np.asarray(marker["normal"], dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    vertical_component = float(np.dot(normal, up_axis))
    if abs(vertical_component) < MIN_VERTICAL_COMPONENT:
        return None
    if vertical_component < 0:
        normal = -normal
    return origin, normal


def _slice_keeping(mesh, origin, normal):
    """Slices the mesh at the plane, keeping the side the normal points toward.

    Returns None when the cut removes everything.
    """
    sliced = mesh.slice_plane(plane_origin=origin, plane_normal=normal, cap=True)
    if sliced is None or len(sliced.faces) == 0:
        return None
    sliced.merge_vertices()
    return sliced


def apply_marker_cut_to_mesh(mesh, markers, up_axis=(0.0, 0.0, 1.0)):
    """Cuts a watertight mesh against at most two marker planes, by height.

    Expects the mesh in levelled space, where `up_axis` is the vertical axis, and the
    markers as {"centroid", "normal"} in that same space.

    Args:
        mesh: a watertight trimesh.Trimesh in levelled space.
        markers: list of {"centroid": (3,), "normal": (3,)}. Normals may point
            either way.
        up_axis: unit vertical axis of the mesh.

    Returns:
        (cut_mesh, case_label). The mesh is returned uncut, with a label saying
        why, whenever the cut cannot be made or would leave almost nothing.
    """
    if len(markers) == 0:
        return mesh, "no_markers"

    up_axis = np.asarray(up_axis, dtype=np.float64)
    up_axis = up_axis / (np.linalg.norm(up_axis) + 1e-8)

    planes = []
    for marker in markers[:MAX_MARKERS]:
        plane = _upward_plane(marker, up_axis)
        if plane is not None:
            planes.append(plane)

    if not planes:
        return mesh, "no_usable_markers"

    if len(planes) == 1:
        # Keep what is BELOW the plane, so the kept side is the one the
        # downward normal points toward.
        origin, normal = planes[0]
        cut = _slice_keeping(mesh, origin, -normal)
        case = "case_1_below"
    else:
        # Keep what lies BETWEEN the two. Which plane is upper is decided by
        # each point's own signed distance in the cloud path; on a mesh the same
        # thing is achieved by keeping below one and above the other, applied in
        # turn. Ordering the planes by height first means the two slices cannot
        # both discard the same side and leave nothing.
        first, second = planes
        first_height = float(np.dot(first[0], up_axis))
        second_height = float(np.dot(second[0], up_axis))
        if first_height <= second_height:
            lower_plane, upper_plane = first, second
        else:
            lower_plane, upper_plane = second, first

        cut = _slice_keeping(mesh, lower_plane[0], lower_plane[1])
        if cut is not None:
            cut = _slice_keeping(cut, upper_plane[0], -upper_plane[1])
        case = "case_2_between"

    if cut is None or len(cut.faces) < MIN_FACES_AFTER_CUT:
        return mesh, "too_few_after_cut"

    return cut, case
