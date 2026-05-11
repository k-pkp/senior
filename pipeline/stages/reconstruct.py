"""Stage 4 — Poisson reconstruction (non-watertight) for each object PLY."""
import os
import subprocess
import sys

import numpy as np
import open3d as o3d

from pipeline.core.mesh import merge_meshes, clean_merged_scene


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_RECONS_WORKER = os.path.join(_PROJECT_ROOT, "workers", "recons_worker.py")


def reconstruct_multiple_objects(input_paths, output_folder="output_mesh",
                                 base_name="scene_recon", seed=42):
    """Reconstruct each object PLY into a (non-watertight) Poisson mesh.

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
            [sys.executable, _RECONS_WORKER, path, recon_ply, "--seed", str(seed)],
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

    print("\nMerging reconstruction meshes...")
    final_mesh = merge_meshes(meshes)
    final_mesh = clean_merged_scene(final_mesh)
    print(f"Scene triangles: {len(final_mesh.triangles):,}")

    scene_ply = os.path.join(output_folder, f"{base_name}.ply")
    scene_stl = os.path.join(output_folder, f"{base_name}.stl")
    o3d.io.write_triangle_mesh(scene_ply, final_mesh)
    o3d.io.write_triangle_mesh(scene_stl, final_mesh)

    return scene_ply, scene_stl, recon_paths


def reconstruct_mesh_stage(object_paths, output_dir, seed=42):
    """Pipeline wrapper. Returns (scene_recon_path, recon_mesh_paths)."""
    print()
    print("=" * 60)
    print("STAGE 4: Reconstructing mesh (Poisson)")
    print("=" * 60)

    mesh_output_dir = os.path.join(output_dir, "mesh")
    os.makedirs(mesh_output_dir, exist_ok=True)

    try:
        scene_recon, _, recon_paths = reconstruct_multiple_objects(
            input_paths=object_paths,
            output_folder=mesh_output_dir,
            base_name="scene_recon",
            seed=seed,
        )
        print(f"  Scene recon mesh: {scene_recon}")
        for p in recon_paths:
            print(f"  Object recon mesh: {p}")
        return scene_recon, recon_paths
    except Exception as e:
        print(f"  ERROR during reconstruction: {e}")
        return None, []


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


# numpy import kept for type-consistency with original recons.py
_ = np
