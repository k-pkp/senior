"""Stage 6 — Capture multi-perspective screenshots of outputs via viewer.py."""
import os
import subprocess
import sys


def _capture_multi_view(viewer_script, file_path, output_dir, label, extra_args=None):
    """Capture multi-perspective screenshots of a 3D file using viewer.py --multi-view."""
    cmd = [
        sys.executable, viewer_script,
        file_path,
        "--multi-view",
        "--screenshot", output_dir,
        "--bg-color", "0.05", "0.05", "0.05",
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode == 0:
        count = result.stdout.strip().count("Saved:")
        print(f"    ✓ {label}: {count} views → {output_dir}")
        return True
    err = result.stderr[:200] if result.stderr else "unknown error"
    print(f"    ✗ {label} failed: {err}")
    return False


def evaluate_with_viewer(output_dir, ply_path, mesh_path=None,
                         object_mesh_paths=None, viewer_script=None):
    """Capture multi-perspective screenshots of PLY, scene mesh, and each object mesh."""
    print()
    print("=" * 60)
    print("STAGE 6: Evaluation — multi-perspective screenshots")
    print("=" * 60)

    if viewer_script is None:
        viewer_script = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
            "viewer.py",
        )
    if not os.path.exists(viewer_script):
        print("  WARNING: viewer.py not found. Skipping evaluation.")
        return

    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    if ply_path and os.path.exists(ply_path):
        print("  Point cloud (6 views)...")
        _capture_multi_view(viewer_script, ply_path,
                            os.path.join(eval_dir, "pointcloud"),
                            "Point cloud", ["--point-size", "2.0"])

    if mesh_path and os.path.exists(mesh_path):
        print("  Scene mesh (6 views)...")
        _capture_multi_view(viewer_script, mesh_path,
                            os.path.join(eval_dir, "scene"),
                            "Scene mesh")

    if object_mesh_paths:
        for obj_path in object_mesh_paths:
            if os.path.exists(obj_path):
                name = os.path.splitext(os.path.basename(obj_path))[0]
                print(f"  {name} mesh (6 views)...")
                _capture_multi_view(viewer_script, obj_path,
                                     os.path.join(eval_dir, name),
                                     name)

    print()
    print("  --- Output Summary ---")
    all_outputs = [("Point Cloud", ply_path), ("Scene Mesh", mesh_path)]
    if object_mesh_paths:
        for p in object_mesh_paths:
            name = os.path.basename(p)
            all_outputs.append((f"{name}", p))
    for label, path in all_outputs:
        if path and os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {label}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"  {label}: (not generated)")
