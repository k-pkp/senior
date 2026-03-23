import os
from pathlib import Path
import numpy as np
import open3d as o3d


def reconstruct_mesh(
    input_path: str,
    output_folder: str = "output_mesh",
    base_name: str = None,
    poisson_depth: int = 11,
    density_quantile: float = 0.02,
    merge_tolerance: float = 1e-6,
):
    """
    Reconstruct watertight mesh from cleaned point cloud and export PLY + STL.

    Args:
        input_path: path to cleaned point cloud (.ply)
        output_folder: where mesh files will be saved
        base_name: output file name base
        poisson_depth: higher = more detailed but slower
        density_quantile: remove low confidence Poisson vertices
        merge_tolerance: tolerance for merging near-duplicate vertices

    Returns:
        mesh_ply_path, mesh_stl_path
    """

    os.makedirs(output_folder, exist_ok=True)

    in_path = Path(input_path)
    if base_name is None:
        base_name = in_path.stem

    mesh_ply_path = os.path.join(output_folder, f"mesh_{base_name}.ply")
    mesh_stl_path = os.path.join(output_folder, f"mesh_{base_name}.stl")

    # -----------------------------
    # Load point cloud
    # -----------------------------
    print("\nLoading point cloud...")
    pcd = o3d.io.read_point_cloud(str(in_path))

    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty")

    print("Points:", len(pcd.points))

    # -----------------------------
    # Estimate normals
    # -----------------------------
    print("\nEstimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.02,
            max_nn=30
        )
    )

    pcd.orient_normals_consistent_tangent_plane(100)

    # -----------------------------
    # Poisson reconstruction
    # -----------------------------
    print("\nRunning Poisson reconstruction...")
    mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=poisson_depth
    )

    densities = np.asarray(densities)

    print("Initial triangles:", len(mesh_poisson.triangles))

    # -----------------------------
    # Remove low density vertices
    # -----------------------------
    print("\nRemoving low density regions...")
    threshold = np.quantile(densities, density_quantile)

    vertices_to_remove = densities < threshold
    mesh_poisson.remove_vertices_by_mask(vertices_to_remove)

    # -----------------------------
    # Basic cleanup
    # -----------------------------
    print("\nCleaning mesh...")
    mesh_poisson.remove_degenerate_triangles()
    mesh_poisson.remove_duplicated_triangles()
    mesh_poisson.remove_duplicated_vertices()
    mesh_poisson.remove_non_manifold_edges()

    mesh_poisson.compute_vertex_normals()
    mesh_poisson.compute_triangle_normals()

    print("Triangles after cleaning:", len(mesh_poisson.triangles))

    # -----------------------------
    # Check watertight
    # -----------------------------
    if mesh_poisson.is_watertight():
        print("\nUsing Poisson mesh (watertight)")
        mesh = mesh_poisson

    else:
        print("\nPoisson not watertight → using convex hull fallback")

        mesh, _ = pcd.compute_convex_hull()

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

    # -----------------------------
    # STL-safe cleanup
    # -----------------------------
    print("\nPreparing mesh for STL export...")

    mesh.merge_close_vertices(merge_tolerance)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    print("Final triangle count:", len(mesh.triangles))
    print("Watertight before export:", mesh.is_watertight())

    # -----------------------------
    # Save files
    # -----------------------------
    print("\nSaving files...")

    o3d.io.write_triangle_mesh(
        mesh_ply_path,
        mesh,
        write_ascii=False
    )

    o3d.io.write_triangle_mesh(
        mesh_stl_path,
        mesh,
        write_ascii=False
    )

    # -----------------------------
    # Verify STL after reload
    # -----------------------------
    mesh_stl_check = o3d.io.read_triangle_mesh(mesh_stl_path)

    print("\nVerification:")
    print("PLY watertight:", mesh.is_watertight())
    print("STL watertight:", mesh_stl_check.is_watertight())

    print("\nSaved:")
    print(mesh_ply_path)
    print(mesh_stl_path)

    return mesh_ply_path, mesh_stl_path


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":

    input_ply = "clean_input_ply/clean_input.ply"

    reconstruct_mesh(
        input_path=input_ply,
        output_folder="output_mesh",
        base_name="input",

        poisson_depth=11,      # increase if surface has holes
        density_quantile=0.02, # remove noisy edges
        merge_tolerance=1e-6   # critical for STL
    )