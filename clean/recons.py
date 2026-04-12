import os
import subprocess
import sys
import numpy as np
import open3d as o3d


_RECONS_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recons_worker.py")
_MESHFIX_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshfix_worker.py")


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


def _verify_watertight(mesh_path):
    """Verify watertightness using trimesh (standard definition)."""
    import trimesh
    t = trimesh.load(mesh_path)
    return t.is_watertight


def reconstruct_multiple_objects(input_paths, output_folder="output_mesh", base_name="scene_recon"):
    """Stage 4: Reconstruct each object (Poisson only, non-watertight).

    Returns (scene_ply, scene_stl, recon_mesh_paths).
    Each object is saved as `object_N_recon.ply`.
    """
    os.makedirs(output_folder, exist_ok=True)

    meshes = []
    recon_paths = []

    for i, path in enumerate(input_paths):
        print(f"\nProcessing: {path}")

        recon_ply = os.path.join(output_folder, f"object_{i}_recon.ply")
        recon_stl = os.path.join(output_folder, f"object_{i}_recon.stl")

        result = subprocess.run(
            [sys.executable, _RECONS_WORKER, path, recon_ply],
            capture_output=True, text=True, timeout=600,
        )
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                print(f"  {line}")

        if result.returncode != 0 or not os.path.exists(recon_ply):
            err = result.stderr.strip() or "unknown error"
            print(f"  ERROR: {err[:500]}")
            continue

        mesh = o3d.io.read_triangle_mesh(recon_ply)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(recon_stl, mesh)

        size_mb = os.path.getsize(recon_ply) / (1024 * 1024)
        print(f"  Recon object {i}: {len(mesh.vertices):,} verts, "
              f"{len(mesh.triangles):,} faces ({size_mb:.1f} MB)")

        recon_paths.append(recon_ply)
        meshes.append(mesh)

    if len(meshes) == 0:
        raise ValueError("No meshes reconstructed")

    # Merge into scene
    print("\nMerging reconstruction meshes...")
    final_mesh = merge_meshes(meshes)
    final_mesh.remove_degenerate_triangles()
    final_mesh.remove_duplicated_triangles()
    final_mesh.remove_duplicated_vertices()
    final_mesh.remove_non_manifold_edges()
    final_mesh.compute_vertex_normals()
    final_mesh.compute_triangle_normals()
    print(f"Scene triangles: {len(final_mesh.triangles):,}")

    scene_ply = os.path.join(output_folder, f"{base_name}.ply")
    scene_stl = os.path.join(output_folder, f"{base_name}.stl")
    o3d.io.write_triangle_mesh(scene_ply, final_mesh)
    o3d.io.write_triangle_mesh(scene_stl, final_mesh)

    return scene_ply, scene_stl, recon_paths


def make_watertight_meshes(recon_paths, output_folder="output_mesh", base_name="scene"):
    """Stage 5: Fill each reconstructed mesh to be watertight (PyMeshFix + color transfer).

    Returns (scene_ply, scene_stl, watertight_mesh_paths).
    Each object is saved as `object_N.ply` (watertight version).
    """
    os.makedirs(output_folder, exist_ok=True)

    meshes = []
    watertight_paths = []

    for i, recon_path in enumerate(recon_paths):
        print(f"\nRepairing: {recon_path}")

        wt_ply = os.path.join(output_folder, f"object_{i}.ply")
        wt_stl = os.path.join(output_folder, f"object_{i}.stl")

        # Remove stale files from previous runs so existence == success
        for p in (wt_ply, wt_stl):
            if os.path.exists(p):
                os.remove(p)

        # meshfix_worker.py takes (input, output, color_source)
        result = subprocess.run(
            [sys.executable, _MESHFIX_WORKER, recon_path, wt_ply, recon_path],
            capture_output=True, text=True, timeout=300,
        )
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                print(f"  {line}")

        if result.returncode != 0 or not os.path.exists(wt_ply):
            err = result.stderr.strip() or "unknown error"
            print(f"  ERROR: MeshFix failed: {err[:500]}")
            # Fallback: copy recon as output (mesh will NOT be watertight)
            import shutil
            shutil.copy2(recon_path, wt_ply)
            print(f"  Fallback: saved recon mesh as output (NOT watertight)")

        mesh = o3d.io.read_triangle_mesh(wt_ply)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(wt_stl, mesh)

        is_wt = _verify_watertight(wt_ply)
        size_mb = os.path.getsize(wt_ply) / (1024 * 1024)
        print(f"  Watertight object {i}: {len(mesh.vertices):,} verts, "
              f"{len(mesh.triangles):,} faces, watertight={is_wt} ({size_mb:.1f} MB)")

        watertight_paths.append(wt_ply)
        meshes.append(mesh)

    if len(meshes) == 0:
        raise ValueError("No watertight meshes produced")

    # Merge into scene
    print("\nMerging watertight meshes...")
    final_mesh = merge_meshes(meshes)
    final_mesh.remove_degenerate_triangles()
    final_mesh.remove_duplicated_triangles()
    final_mesh.remove_duplicated_vertices()
    final_mesh.remove_non_manifold_edges()
    final_mesh.compute_vertex_normals()
    final_mesh.compute_triangle_normals()
    print(f"Scene triangles: {len(final_mesh.triangles):,}")

    scene_ply = os.path.join(output_folder, f"{base_name}.ply")
    scene_stl = os.path.join(output_folder, f"{base_name}.stl")
    o3d.io.write_triangle_mesh(scene_ply, final_mesh)
    o3d.io.write_triangle_mesh(scene_stl, final_mesh)

    return scene_ply, scene_stl, watertight_paths


def reconstruct_mesh(
    input_path: str,
    output_folder: str = "output_mesh",
    base_name: str = None,
    poisson_depth: int = None,
    density_quantile: float = 0.01,
    merge_tolerance: float = 1e-6,
):
    """Compatibility entry point for demo_gradio pipeline."""
    del merge_tolerance, poisson_depth, density_quantile

    os.makedirs(output_folder, exist_ok=True)
    if base_name is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]

    mesh_ply = os.path.join(output_folder, f"{base_name}.ply")
    mesh_stl = os.path.join(output_folder, f"{base_name}.stl")

    result = subprocess.run(
        [sys.executable, _RECONS_WORKER, input_path, mesh_ply],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode == 0 and os.path.exists(mesh_ply):
        mesh = o3d.io.read_triangle_mesh(mesh_ply)
        o3d.io.write_triangle_mesh(mesh_stl, mesh)

    return mesh_ply, mesh_stl
