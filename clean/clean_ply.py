import os
from pathlib import Path

import numpy as np
import open3d as o3d


def clean_point_cloud(
    input_path: str,
    output_folder: str = "clean_input_ply",
    output_name: str = None,
) -> str:
    """Clean a point cloud and save the cleaned result as PLY.

    Args:
        input_path: Path to source point cloud file.
        output_folder: Folder for cleaned output.
        output_name: Output file name. If None, uses clean_<input_stem>.ply.

    Returns:
        Path to cleaned PLY file.
    """
    os.makedirs(output_folder, exist_ok=True)

    in_path = Path(input_path)
    if output_name is None:
        output_name = f"clean_{in_path.stem}.ply"
    output_path = os.path.join(output_folder, output_name)

    pcd = o3d.io.read_point_cloud(str(in_path))
    if len(pcd.points) == 0:
        raise ValueError(f"Point cloud is empty or unreadable: {input_path}")
    print("Loaded")

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=18, std_ratio=1.7)
    print("Light clean")

    pcd = pcd.voxel_down_sample(voxel_size=0.003)
    if len(pcd.points) == 0:
        raise ValueError("No points left after voxel downsampling.")
    print("Downsampled")

    _, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    pcd = pcd.select_by_index(inliers, invert=True)
    if len(pcd.points) == 0:
        raise ValueError("No points left after plane removal.")
    print("Plane removed")

    labels = np.array(pcd.cluster_dbscan(eps=0.03, min_points=25))
    valid = labels >= 0
    if not np.any(valid):
        main_object = pcd
        print("No DBSCAN clusters found; using all remaining points")
    else:
        unique_ids, counts = np.unique(labels[valid], return_counts=True)
        main_cluster_id = unique_ids[np.argmax(counts)]
        main_indices = np.where(labels == main_cluster_id)[0]
        main_object = pcd.select_by_index(main_indices)
        print(f"Clusters: {len(unique_ids)}")
        print("Object selected")

    main_object, _ = main_object.remove_statistical_outlier(nb_neighbors=18, std_ratio=1.7)
    labels2 = np.array(main_object.cluster_dbscan(eps=0.01, min_points=20))

    valid2 = labels2 >= 0
    if np.any(valid2):
        unique_ids2 = np.unique(labels2[valid2])
        print(f"Secondary clusters: {len(unique_ids2)}")
        clean_clusters = []
        for cluster_id in unique_ids2:
            idx = np.where(labels2 == cluster_id)[0]
            cluster = main_object.select_by_index(idx)
            if len(cluster.points) > 500:
                clean_clusters.append(cluster)

        if clean_clusters:
            merged = clean_clusters[0]
            for cluster in clean_clusters[1:]:
                merged += cluster
            main_object = merged
            print("Removed floating noise clusters")

    if len(main_object.points) == 0:
        raise ValueError("No points left after cleaning.")

    o3d.io.write_point_cloud(output_path, main_object)
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    default_input = "input/input.ply"
    default_output_folder = "clean_input_ply"
    default_output_name = "clean_input.ply"
    cleaned = clean_point_cloud(default_input, default_output_folder, default_output_name)
    print(f"Done: {cleaned}")