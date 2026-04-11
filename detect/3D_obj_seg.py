import open3d as o3d
import numpy as np

def segment_leg_from_mesh(mesh):
    print("🔄 Sampling mesh to point cloud...")

    # Convert mesh → point cloud
    pcd = mesh.sample_points_uniformly(number_of_points=50000)

    print("🔄 Running DBSCAN...")

    labels = np.array(
        pcd.cluster_dbscan(
            eps=0.02,        # tune this
            min_points=100   # tune this
        )
    )

    num_clusters = labels.max() + 1
    print(f"🔍 Found {num_clusters} clusters")

    clusters = []
    for i in range(num_clusters):
        idx = np.where(labels == i)[0]
        cluster = pcd.select_by_index(idx)
        clusters.append(cluster)

    # =========================
    # PICK LEG (largest vertical object)
    # =========================
    def score_cluster(c):
        bbox = c.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()

        height = extent[2]  # assuming Z = vertical
        size = len(c.points)

        return height * size  # prioritize tall + large

    leg_cluster = max(clusters, key=score_cluster)

    print("🦵 Leg extracted")

    return leg_cluster