import os
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
        merged.triangles.extend(
            o3d.utility.Vector3iVector(triangles + vertex_offset)
        )

        vertex_offset += len(vertices)

    merged.compute_vertex_normals()
    merged.compute_triangle_normals()

    return merged


def _estimate_adaptive_depth(num_points):
    """Choose Poisson depth based on point cloud size.

    More points → can support higher depth.
    Too high a depth for sparse data causes 'Failed to close loop' errors.
    """
    if num_points < 5000:
        return 7
    elif num_points < 20000:
        return 8
    elif num_points < 100000:
        return 9
    elif num_points < 500000:
        return 10
    else:
        return 11


def reconstruct_single_object(pcd, poisson_depth=None, density_quantile=0.01):
    """Reconstruct one mesh from a point cloud using Poisson + cleanup.

    Uses adaptive depth with fallback: tries the requested depth first,
    then reduces if Poisson fails (e.g. "Failed to close loop" errors
    at high octree depths with noisy/sparse data).
    """
    num_points = len(pcd.points)
    print(f"  Input points: {num_points:,}")

    if num_points < 100:
        raise ValueError(f"Too few points ({num_points}) for mesh reconstruction")

    # ---- auto depth if not specified ----
    if poisson_depth is None:
        poisson_depth = _estimate_adaptive_depth(num_points)
        print(f"  Auto Poisson depth: {poisson_depth} (based on {num_points:,} points)")

    # ---- downsample if very dense (Poisson struggles with >300k) ----
    if num_points > 300000:
        voxel_size = pcd.get_axis_aligned_bounding_box().get_max_extent() / 400
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"  Downsampled for Poisson: {num_points:,} → {len(pcd.points):,}")

    # ---- normals ----
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    # Adaptive search radius: wider for sparse, tighter for dense
    radius = max(avg_dist * 4.0, 0.005)
    max_nn = min(max(30, int(len(pcd.points) * 0.01)), 100)

    print(f"  Normal estimation: radius={radius:.4f}, max_nn={max_nn}")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn
        )
    )

    # Orient normals — use towards camera center if few points, tangent plane if many
    if len(pcd.points) < 10000:
        # For small clouds, orient toward the centroid works better
        centroid = np.mean(np.asarray(pcd.points), axis=0)
        pcd.orient_normals_towards_camera_location(centroid)
    else:
        try:
            pcd.orient_normals_consistent_tangent_plane(min(100, len(pcd.points) // 10))
        except Exception:
            centroid = np.mean(np.asarray(pcd.points), axis=0)
            pcd.orient_normals_towards_camera_location(centroid)

    # ---- Poisson with adaptive depth fallback ----
    mesh = None
    densities = None
    for depth in range(poisson_depth, 5, -1):
        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=depth
            )
            densities = np.asarray(densities)
            print(f"  Poisson succeeded at depth={depth}")
            break
        except Exception as e:
            err_msg = str(e)
            if len(err_msg) > 100:
                err_msg = err_msg[:100] + "..."
            print(f"  Poisson depth={depth} failed: {err_msg}")
            continue

    if mesh is None:
        # Final fallback: Ball Pivoting
        print("  Poisson failed at all depths, falling back to Ball Pivoting")
        radii = [avg_dist * 2, avg_dist * 4, avg_dist * 8, avg_dist * 16]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

    # ---- density filter (only for Poisson output) ----
    if densities is not None and len(densities) > 0:
        threshold = np.quantile(densities, density_quantile)
        vertices_before = len(mesh.vertices)
        mesh.remove_vertices_by_mask(densities < threshold)
        print(f"  Density filter: {vertices_before:,} → {len(mesh.vertices):,} vertices")

    # ---- cleanup ----
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    print(f"  Triangles: {len(mesh.triangles):,}")
    print(f"  Watertight: {mesh.is_watertight()}")

    return mesh


def reconstruct_multiple_objects(
    input_paths,
    output_folder="output_mesh",
    base_name="scene"
):
    """Reconstruct each object and merge all meshes into one scene mesh."""
    os.makedirs(output_folder, exist_ok=True)

    meshes = []

    for path in input_paths:
        print(f"\nProcessing: {path}")

        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) == 0:
            print("Skipped empty file")
            continue

        try:
            mesh = reconstruct_single_object(pcd)
            meshes.append(mesh)
        except Exception as e:
            print(f"  WARNING: Reconstruction failed for {path}: {e}")
            continue

    if len(meshes) == 0:
        raise ValueError("No meshes created from any input")

    # ---- merge ----
    print("\nMerging meshes...")
    final_mesh = merge_meshes(meshes)

    final_mesh.remove_degenerate_triangles()
    final_mesh.remove_duplicated_triangles()
    final_mesh.remove_duplicated_vertices()
    final_mesh.remove_non_manifold_edges()

    final_mesh.compute_vertex_normals()
    final_mesh.compute_triangle_normals()

    print(f"Final triangles: {len(final_mesh.triangles):,}")

    # ---- save ----
    ply_path = os.path.join(output_folder, f"{base_name}.ply")
    stl_path = os.path.join(output_folder, f"{base_name}.stl")

    print("\nSaving...")
    o3d.io.write_triangle_mesh(ply_path, final_mesh)
    o3d.io.write_triangle_mesh(stl_path, final_mesh)

    return ply_path, stl_path


def reconstruct_mesh(
    input_path: str,
    output_folder: str = "output_mesh",
    base_name: str = None,
    poisson_depth: int = None,
    density_quantile: float = 0.01,
    merge_tolerance: float = 1e-6,
):
    """
    Compatibility entry point for demo_gradio pipeline.

    This wraps the existing single-object reconstruction behavior and returns
    (mesh_ply_path, mesh_stl_path).
    """
    del merge_tolerance  # Preserved for call compatibility.

    os.makedirs(output_folder, exist_ok=True)

    if base_name is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]

    pcd = o3d.io.read_point_cloud(input_path)
    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty")

    mesh = reconstruct_single_object(
        pcd,
        poisson_depth=poisson_depth,
        density_quantile=density_quantile,
    )

    mesh_ply_path = os.path.join(output_folder, f"{base_name}.ply")
    mesh_stl_path = os.path.join(output_folder, f"{base_name}.stl")

    o3d.io.write_triangle_mesh(mesh_ply_path, mesh)
    o3d.io.write_triangle_mesh(mesh_stl_path, mesh)

    return mesh_ply_path, mesh_stl_path


if __name__ == "__main__":

    input_objects = [
        "output_objects/object_0.ply",
        "output_objects/object_1.ply",
    ]

    reconstruct_multiple_objects(
        input_paths=input_objects,
        output_folder="output_mesh",
        base_name="scene"
    )
