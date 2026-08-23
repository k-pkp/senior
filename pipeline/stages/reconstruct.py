"""Stage 4 — Surface reconstruction for each object PLY."""
import os
import numpy as np
import subprocess
import sys

import open3d as o3d

from pipeline.core.mesh import merge_meshes, clean_merged_scene


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_RECONS_WORKER = os.path.join(_PROJECT_ROOT, "workers", "recons_methods_worker.py")
# Poisson for both objects, chosen 2026-08-23 after Stage 5 was fixed to call
# PyMeshFix's `repair()` rather than `fill_holes()` alone.
#
# The earlier evidence against Poisson was an artefact of that weaker repair.
# Re-measured with it fixed, on inputs/small_leg:
#
#   arm                     limb cm3   box chi   limb chi   p95 to the cloud
#   alpha / alpha            1081.94      2         2         2.39 mm
#   alpha cube + PSR limb    1070.85      2         2         1.30 mm
#   PSR / PSR                1074.32      2         2         1.30 mm
#
# All three are valid solids; Poisson fits the points 1.8x closer and lands
# within 1% of alpha. Across depth 8-11 and trim 0.01-0.10 its answer spans
# 0.32%, so it is not sensitive to its own parameters.
#
# WHAT THIS COSTS, stated because it is real: alpha shape's ladder GUARANTEES
# chi = 2 by selecting on it, and Poisson has no such guarantee. On
# inputs/short_leg -- an uncut whole leg including the foot, which is genuinely
# awkward topology -- Poisson closes at chi = -18 and reports 22% below the
# alpha answer. Stage 5 now prints a loud warning whenever the final mesh is not
# chi = 2, because a closed-but-holed mesh is still `is_watertight` and Stage 6
# would otherwise integrate it in silence.
#
# `--recon-method alpha_shape` restores the guaranteed path for any capture the
# warning fires on.
_DEFAULT_METHOD = "poisson"
_DEFAULT_BOX_METHOD = "poisson"


def _recon_name(input_path):
    """Derive output base name from input PLY: box.ply → box_recon, obj.ply → obj_recon."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    return f"{base}_recon"


def _pick_method(path, method, box_method, obj_method):
    """Determine recon method for a single object path."""
    basename = os.path.basename(path).lower()
    if "box" in basename:
        return box_method if box_method is not None else _DEFAULT_BOX_METHOD
    if obj_method is not None and ("obj" in basename or "leg" in basename):
        return obj_method
    if method is not None:
        return method
    return _DEFAULT_METHOD


ALPHA_FALLBACK = "alpha_shape"


def _survives_repair(recon_ply):
    """Would Stage 5 turn this mesh into a single closed solid? (chi == 2)

    Runs the same repair Stage 5 does and asks the question Stage 6 actually
    needs answered. `is_watertight` is NOT that question: a surface with tunnels
    is closed, returns True, and its signed volume is not the volume of a solid.
    Measured case: Poisson on inputs/short_leg repairs to watertight at chi=-18,
    about ten handles, and reads 22% below the alpha-shape answer for the same
    cloud.

    Returns (ok: bool, euler: int | None).
    """
    try:
        import trimesh
        from workers.meshfix_worker import _pymeshfix_repair, _o3d_fill_holes

        mesh = o3d.io.read_triangle_mesh(recon_ply)
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        candidate = trimesh.Trimesh(verts, faces, process=False)
        if not candidate.is_watertight:
            verts, faces = _pymeshfix_repair(verts, faces)
            candidate = trimesh.Trimesh(verts, faces, process=False)
            if not candidate.is_watertight:
                verts, faces = _o3d_fill_holes(verts, faces)
                candidate = trimesh.Trimesh(verts, faces, process=False)
        if not candidate.is_watertight:
            return False, None
        euler = int(candidate.euler_number)
        return euler == 2, euler
    except Exception as exc:
        # Do not fall back on an error in the CHECK -- that would silently swap
        # the method because trimesh hiccupped. Say so and keep what we have.
        print(f"  validity check failed ({type(exc).__name__}: {exc}); "
              f"keeping the mesh as reconstructed")
        return True, None


def reconstruct_multiple_objects(input_paths, output_folder="output_mesh",
                                  base_name="scene_recon", seed=42,
                                  method=None, box_method=None, obj_method=None):
    """Reconstruct each object PLY into a mesh using the chosen method.

    Returns (scene_ply, scene_stl, recon_mesh_paths).
    Files saved as box_recon.ply / obj_recon.ply based on input names.
    """
    os.makedirs(output_folder, exist_ok=True)

    meshes = []
    recon_paths = []
    fallback_used = []

    for i, path in enumerate(input_paths):
        obj_method_name = _pick_method(path, method, box_method, obj_method)
        print(f"\nProcessing: {path}  [{obj_method_name}]")

        name = _recon_name(path)
        recon_ply = os.path.join(output_folder, f"{name}.ply")
        recon_stl = os.path.join(output_folder, f"{name}.stl")

        # Reuse a mesh that is already newer than the cloud it came from.
        #
        # The reference cube is the case this exists for: the cut does not
        # touch it, so a run that measures the cube first and cuts the limb
        # afterwards would otherwise reconstruct the same cube twice. Alpha
        # shapes are deterministic under a fixed seed, so the reused mesh is
        # the one that would have been rebuilt.
        if (os.path.exists(recon_ply)
                and os.path.getmtime(recon_ply) >= os.path.getmtime(path)):
            mesh = o3d.io.read_triangle_mesh(recon_ply)
            mesh.compute_vertex_normals()
            print(f"  Reusing {name}: {len(mesh.vertices):,} verts, "
                  f"{len(mesh.triangles):,} faces (cloud unchanged)")
            recon_paths.append(recon_ply)
            meshes.append(mesh)
            continue

        result = subprocess.run(
            [sys.executable, _RECONS_WORKER, path, recon_ply,
             "--method", obj_method_name, "--seed", str(seed)],
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
        print(f"  Recon {name}: {len(mesh.vertices):,} verts, "
              f"{len(mesh.triangles):,} faces ({size_mb:.1f} MB)")

        # Poisson has no topological guarantee. Alpha shape does, because its
        # ladder selects on chi = 2 -- so when Poisson produces something Stage 5
        # cannot turn into a solid, rebuild THIS OBJECT with alpha shape rather
        # than reporting a volume that is not a volume.
        if obj_method_name != ALPHA_FALLBACK:
            ok, euler = _survives_repair(recon_ply)
            if not ok:
                print(f"  {name}: {obj_method_name} gives "
                      f"{'chi=' + str(euler) if euler is not None else 'a mesh Stage 5 cannot close'}"
                      f", not a single closed solid — rebuilding with "
                      f"{ALPHA_FALLBACK}, whose ladder guarantees chi=2")
                fb = subprocess.run(
                    [sys.executable, _RECONS_WORKER, path, recon_ply,
                     "--method", ALPHA_FALLBACK, "--seed", str(seed)],
                    capture_output=True, text=True, timeout=600,
                )
                for line in fb.stdout.strip().split("\n"):
                    if line.strip():
                        print(f"    {line}")
                if fb.returncode == 0 and os.path.exists(recon_ply):
                    mesh = o3d.io.read_triangle_mesh(recon_ply)
                    mesh.compute_vertex_normals()
                    o3d.io.write_triangle_mesh(recon_stl, mesh)
                    ok2, euler2 = _survives_repair(recon_ply)
                    print(f"  {name}: fallback gives "
                          f"{'chi=' + str(euler2) if euler2 is not None else 'unknown chi'}"
                          f" — {'a valid solid' if ok2 else 'STILL NOT a valid solid'}")
                    fallback_used.append(name)
                else:
                    print(f"  {name}: {ALPHA_FALLBACK} fallback FAILED, keeping "
                          f"the {obj_method_name} mesh — its volume is not "
                          f"trustworthy")

        recon_paths.append(recon_ply)
        meshes.append(mesh)

    if fallback_used:
        print(f"\n  Fell back to {ALPHA_FALLBACK} for: {', '.join(fallback_used)}")

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


def reconstruct_mesh_stage(object_paths, output_dir, seed=42, method=None,
                           box_method=None, obj_method=None):
    """Pipeline wrapper. Returns (scene_recon_path, recon_mesh_paths)."""
    if method is None:
        method = _DEFAULT_METHOD
    parts = [method.replace("_", " ").title()]
    if box_method and box_method != method:
        parts.append(f"box={box_method}")
    if obj_method and obj_method != method:
        parts.append(f"obj={obj_method}")
    method_label = " + ".join(parts)
    print()
    print("=" * 60)
    print(f"STAGE 4: Reconstructing mesh ({method_label})")
    print("=" * 60)

    mesh_output_dir = os.path.join(output_dir, "mesh")
    os.makedirs(mesh_output_dir, exist_ok=True)

    try:
        scene_recon, _, recon_paths = reconstruct_multiple_objects(
            input_paths=object_paths,
            output_folder=mesh_output_dir,
            base_name="scene_recon",
            seed=seed,
            method=method,
            box_method=box_method,
            obj_method=obj_method,
        )
        print(f"  Scene recon mesh: {scene_recon}")
        for p in recon_paths:
            print(f"  Object recon mesh: {p}")
        return scene_recon, recon_paths
    except Exception as e:
        print(f"  ERROR during reconstruction: {e}")
        return None, []
