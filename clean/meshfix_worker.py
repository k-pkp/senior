#!/usr/bin/env python3
"""Subprocess worker for watertight mesh repair using PyMeshFix.

Called by recons.py to avoid OpenGL/library conflicts between Open3D and PyMeshFix.

Usage:
    python meshfix_worker.py input.ply output.ply
    python meshfix_worker.py input.ply output.ply color_source.ply
"""
import sys
import traceback
import numpy as np
import pymeshfix
import pyvista as pv


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    color_source_path = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        mesh = pv.read(input_path)
        vertices = np.ascontiguousarray(mesh.points, dtype=np.float64)
        faces = np.ascontiguousarray(mesh.regular_faces, dtype=np.int32)

        print(f"Input: {len(vertices)} vertices, {len(faces)} faces")

        fixer = pymeshfix.MeshFix(vertices, faces)
        fixer.repair()

        v_out = fixer.points
        f_out = fixer.faces

        print(f"Output: {len(v_out)} vertices, {len(f_out)} faces")
        print(f"Retention: {len(f_out)/len(faces)*100:.1f}%")

        import trimesh

        # Transfer colors from source mesh if provided
        if color_source_path:
            from scipy.spatial import KDTree
            source = trimesh.load(color_source_path)
            src_colors = source.visual.vertex_colors
            if src_colors is not None and len(src_colors) > 0:
                tree = KDTree(source.vertices)
                dists, indices = tree.query(v_out)
                colors = src_colors[indices]
                out_mesh = trimesh.Trimesh(vertices=v_out, faces=f_out,
                                           vertex_colors=colors)
                print(f"Colors: transferred (mean_dist={dists.mean():.6f})")
            else:
                out_mesh = trimesh.Trimesh(vertices=v_out, faces=f_out)
                print("Colors: source has no vertex colors")
        else:
            out_mesh = trimesh.Trimesh(vertices=v_out, faces=f_out)

        out_mesh.export(output_path)
        print("OK")

    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
