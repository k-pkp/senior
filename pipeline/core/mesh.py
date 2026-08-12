"""Mesh-level utilities shared by reconstruct/watertight stages."""
import numpy as np
import open3d as o3d


def merge_meshes(mesh_list):
    """Merge multiple meshes into one triangle mesh."""
    merged = o3d.geometry.TriangleMesh()
    vertex_offset = 0
    for mesh in mesh_list:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        merged.vertices.extend(o3d.utility.Vector3dVector(vertices))
        merged.triangles.extend(o3d.utility.Vector3iVector(triangles + vertex_offset))
        vertex_offset += len(vertices)
    merged.compute_vertex_normals()
    merged.compute_triangle_normals()
    return merged


def verify_watertight(mesh_path):
    """Verify watertightness using trimesh.

    process=False — PyMeshFix's fill_holes leaves a few intentional duplicate
    seam vertices that trimesh's default merge would collapse, breaking
    edge-manifoldness. The on-disk mesh has 2 faces per edge by index.
    """
    import trimesh
    t = trimesh.load(mesh_path, process=False)
    return t.is_watertight


def is_closed(mesh):
    """Watertight by the definition the rest of the pipeline uses (trimesh).

    Open3D's is_watertight is stricter about vertex-manifoldness and disagrees
    on meshes Stage 4, Stage 5 and Stage 6 all treat as closed, so guarding on
    it would protect a property nothing downstream reads.
    """
    import trimesh
    try:
        t = trimesh.Trimesh(np.asarray(mesh.vertices),
                            np.asarray(mesh.triangles), process=False)
        return bool(t.is_watertight)
    except Exception:
        return bool(mesh.is_watertight())


def clean_merged_scene(mesh):
    """Standard cleanup applied to merged scene meshes.

    remove_non_manifold_edges deletes every triangle touching a non-manifold
    edge, which tears a closed surface open. On a merged scene it was
    catastrophic: Stage 4 hands over two closed objects (watertight, euler 4)
    and this turned them into 7,554 fragments with euler 3113, deleting 12,285
    of 31,738 triangles. scene_mesh.ply is a published deliverable and what the
    web app renders, so it was shipping shredded geometry.

    Snapshot before the destructive step and revert if it opens the mesh. A
    closed surface has one component per object, so the cleanup has nothing
    legitimate to remove from it in the first place.
    """
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()

    closed_before = is_closed(mesh)
    snapshot = o3d.geometry.TriangleMesh(mesh) if closed_before else None

    mesh.remove_non_manifold_edges()

    if closed_before and not is_closed(mesh):
        print("  Scene cleanup opened a closed mesh — reverting to the "
              "pre-cleanup version")
        mesh = snapshot

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh
