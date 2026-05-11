#!/usr/bin/env python3
"""View .ply and .stl files. Usage: python viewer.py <file> [file2 ...]"""
import sys
from pathlib import Path

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d not installed. Run: pip install open3d")


def load(path: Path):
    ext = path.suffix.lower()
    if ext == ".stl":
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not mesh.has_vertices():
            raise ValueError(f"Empty/invalid STL: {path}")
        mesh.compute_vertex_normals()
        return mesh
    if ext == ".ply":
        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.has_triangles():
            mesh.compute_vertex_normals()
            return mesh
        pcd = o3d.io.read_point_cloud(str(path))
        if not pcd.has_points():
            raise ValueError(f"Empty/invalid PLY: {path}")
        return pcd
    raise ValueError(f"Unsupported extension: {ext}")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python viewer.py <file.ply|file.stl> [...]")

    geoms = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            print(f"skip (not found): {p}", file=sys.stderr)
            continue
        try:
            geoms.append(load(p))
            print(f"loaded: {p}")
        except Exception as e:
            print(f"skip ({e}): {p}", file=sys.stderr)

    if not geoms:
        sys.exit("Nothing to display.")

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries(
        geoms + [axes],
        window_name="PLY/STL Viewer",
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()
