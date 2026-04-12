#!/usr/bin/env python3
"""
VGGT Viewer — view .ply and .stl files for result evaluation.

Usage:
    python viewer.py path/to/points.ply
    python viewer.py path/to/mesh.stl
    python viewer.py path/to/points.ply path/to/mesh.stl   # overlay both
    python viewer.py path/to/points.ply --screenshot out.png
    python viewer.py path/to/points.ply --auto-screenshot   # auto-save and exit

Supports: .ply (point clouds & meshes), .stl, .obj, .glb, .gltf
"""

import os
import sys
import argparse
import numpy as np

try:
    import trimesh
except ImportError:
    print("ERROR: trimesh is required.  pip install trimesh")
    sys.exit(1)

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for screenshot
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def parse_args():
    p = argparse.ArgumentParser(description="View .ply / .stl / .obj files")
    p.add_argument("files", nargs="+", help="One or more 3D files to view")
    p.add_argument("--screenshot", type=str, default=None,
                   help="Save a screenshot to this path (requires Open3D or matplotlib)")
    p.add_argument("--auto-screenshot", action="store_true",
                   help="Auto-capture screenshot and exit (no interactive window)")
    p.add_argument("--point-size", type=float, default=1.0,
                   help="Point size for point cloud rendering (Open3D)")
    p.add_argument("--bg-color", type=float, nargs=3, default=[0.1, 0.1, 0.1],
                   help="Background color as three floats 0-1 (default: dark gray)")
    p.add_argument("--subsample", type=int, default=None,
                   help="Subsample point cloud to at most N points for faster viewing")
    p.add_argument("--info", action="store_true",
                   help="Print file info (vertex count, bounds, etc.) and exit")
    p.add_argument("--multi-view", action="store_true",
                   help="Capture 38 screenshots: 36 orbital views (3 elevations x 12 angles) + top + bottom")
    return p.parse_args()


def load_geometry(filepath):
    """Load a 3D file and return (trimesh object, description string)."""
    scene_or_mesh = trimesh.load(filepath)

    if isinstance(scene_or_mesh, trimesh.Scene):
        geometries = list(scene_or_mesh.geometry.values())
        if not geometries:
            print(f"  Warning: {filepath} contains an empty scene")
            return None, "empty scene"
        combined = trimesh.util.concatenate(geometries)
        desc = f"scene with {len(geometries)} geometries"
    else:
        combined = scene_or_mesh
        desc = type(combined).__name__

    return combined, desc


def print_info(filepath, geom):
    """Print geometry information."""
    print(f"\n--- {os.path.basename(filepath)} ---")
    if isinstance(geom, trimesh.PointCloud):
        pts = np.asarray(geom.vertices)
        print(f"  Type      : PointCloud")
        print(f"  Vertices  : {len(pts):,}")
        if geom.colors is not None and len(geom.colors) > 0:
            print(f"  Colored   : Yes")
        print(f"  Bounds min: {pts.min(axis=0)}")
        print(f"  Bounds max: {pts.max(axis=0)}")
        print(f"  Center    : {pts.mean(axis=0)}")
    elif isinstance(geom, trimesh.Trimesh):
        print(f"  Type      : Trimesh")
        print(f"  Vertices  : {len(geom.vertices):,}")
        print(f"  Faces     : {len(geom.faces):,}")
        print(f"  Watertight: {geom.is_watertight}")
        print(f"  Bounds min: {geom.bounds[0]}")
        print(f"  Bounds max: {geom.bounds[1]}")
        print(f"  Center    : {geom.centroid}")
    else:
        print(f"  Type      : {type(geom).__name__}")
        if hasattr(geom, "vertices"):
            print(f"  Vertices  : {len(geom.vertices):,}")


def _trimesh_to_o3d(geom, args):
    """Convert a trimesh geometry to Open3D geometry."""
    if isinstance(geom, trimesh.PointCloud):
        pts = np.asarray(geom.vertices)
        pcd = o3d.geometry.PointCloud()

        if args.subsample and len(pts) > args.subsample:
            indices = np.random.choice(len(pts), args.subsample, replace=False)
            pts = pts[indices]
            colors = np.asarray(geom.colors)[indices] if geom.colors is not None and len(geom.colors) > 0 else None
        else:
            colors = np.asarray(geom.colors) if geom.colors is not None and len(geom.colors) > 0 else None

        pcd.points = o3d.utility.Vector3dVector(pts)
        if colors is not None:
            c = colors[:, :3].astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(c)

        return pcd

    elif isinstance(geom, trimesh.Trimesh):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(geom.vertices))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(geom.faces))

        try:
            vc = np.asarray(geom.visual.vertex_colors)
            if vc.ndim == 2 and vc.shape[0] == len(geom.vertices) and vc.shape[1] >= 3:
                vc = vc[:, :3].astype(np.float64) / 255.0
                mesh.vertex_colors = o3d.utility.Vector3dVector(vc)
        except (AttributeError, TypeError, ValueError):
            pass

        mesh.compute_vertex_normals()
        return mesh

    return None


def _screenshot_offscreen(o3d_geoms, screenshot_path, args):
    """Capture screenshot using Open3D OffscreenRenderer (works headless on Linux)."""
    import open3d.visualization.rendering as rendering

    width, height = 1920, 1080
    renderer = rendering.OffscreenRenderer(width, height)

    # Setup material
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = args.point_size * 3.0  # OffscreenRenderer uses different scale

    mat_mesh = rendering.MaterialRecord()
    mat_mesh.shader = "defaultLit"

    # Set background
    renderer.scene.set_background(np.array([*args.bg_color, 1.0]))

    # Add geometries
    bbox = None
    for i, g in enumerate(o3d_geoms):
        name = f"geom_{i}"
        if isinstance(g, o3d.geometry.PointCloud):
            renderer.scene.add_geometry(name, g, mat)
        else:
            renderer.scene.add_geometry(name, g, mat_mesh)

        gb = g.get_axis_aligned_bounding_box()
        if bbox is None:
            bbox = gb
        else:
            bbox = bbox + gb

    if bbox is None:
        print("  No geometry to render")
        return False

    # Setup camera to look at the geometry from a good angle
    center = bbox.get_center()
    extent = bbox.get_extent()
    max_extent = max(extent)

    # Position camera at 45 degrees looking at center
    eye = center + np.array([max_extent * 0.8, max_extent * 0.5, max_extent * 0.8])
    up = np.array([0.0, -1.0, 0.0])

    renderer.scene.camera.look_at(center, eye, up)

    img = renderer.render_to_image()
    o3d.io.write_image(screenshot_path, img)
    return True


def _screenshot_multi_view(o3d_geoms, screenshot_dir, base_name, args):
    """Capture screenshots from 16 camera perspectives around the geometry.

    Orbital ring at 3 elevations (high/mid/low, 45-degree steps) + top + bottom.
    """
    import open3d.visualization.rendering as rendering

    width, height = 1920, 1080
    os.makedirs(screenshot_dir, exist_ok=True)

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = args.point_size * 3.0

    mat_mesh = rendering.MaterialRecord()
    mat_mesh.shader = "defaultLit"

    bbox = None
    for g in o3d_geoms:
        gb = g.get_axis_aligned_bounding_box()
        bbox = gb if bbox is None else bbox + gb

    if bbox is None:
        print("  No geometry to render")
        return []

    center = bbox.get_center()
    extent = bbox.get_extent()
    dist = max(extent) * 1.3

    up = np.array([0.0, -1.0, 0.0])
    views = []

    # 3 elevation rings x 12 azimuth steps = 36 orbital views
    elevations = [
        ("mid", 0.3),     # slight downward
        ("high", 0.7),    # steep downward
        ("low", -0.15),   # slightly upward
    ]
    for elev_name, elev in elevations:
        for angle_deg in range(0, 360, 30):
            angle = np.radians(angle_deg)
            r = dist * (0.7 if elev_name == "high" else 1.0)
            eye = np.array([
                np.sin(angle) * r,
                -dist * elev,
                np.cos(angle) * r,
            ])
            views.append((f"{elev_name}_{angle_deg:03d}", eye, up))

    # Top and bottom
    views.append(("top", np.array([0.0, -dist, 0.0]), np.array([0.0, 0.0, 1.0])))
    views.append(("bottom", np.array([0.0, dist, 0.0]), np.array([0.0, 0.0, -1.0])))

    saved = []
    for view_name, eye_offset, view_up in views:
        renderer = rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background(np.array([*args.bg_color, 1.0]))

        for i, g in enumerate(o3d_geoms):
            name = f"geom_{i}"
            if isinstance(g, o3d.geometry.PointCloud):
                renderer.scene.add_geometry(name, g, mat)
            else:
                renderer.scene.add_geometry(name, g, mat_mesh)

        eye = center + eye_offset
        renderer.scene.camera.look_at(center, eye, view_up)

        out_path = os.path.join(screenshot_dir, f"{base_name}_{view_name}.png")
        img = renderer.render_to_image()
        o3d.io.write_image(out_path, img)
        saved.append(out_path)
        del renderer

    return saved


def view_with_open3d(geometries_info, args):
    """View using Open3D (preferred — supports interactive rotation, screenshots)."""
    o3d_geoms = []
    for filepath, geom in geometries_info:
        g = _trimesh_to_o3d(geom, args)
        if g is not None:
            o3d_geoms.append(g)

    if not o3d_geoms:
        print("No viewable geometry found.")
        return

    if args.multi_view:
        screenshot_dir = args.screenshot or "multi_view"
        base_name = os.path.splitext(os.path.basename(args.files[0]))[0]
        try:
            saved = _screenshot_multi_view(o3d_geoms, screenshot_dir, base_name, args)
            for p in saved:
                print(f"  Saved: {p}")
            print(f"  Total: {len(saved)} views saved to {screenshot_dir}")
        except Exception as e:
            print(f"  Multi-view failed: {e}")
        return

    if args.auto_screenshot or args.screenshot:
        screenshot_path = args.screenshot or "screenshot.png"
        # Try OffscreenRenderer first (works headless on Linux)
        try:
            success = _screenshot_offscreen(o3d_geoms, screenshot_path, args)
            if success:
                print(f"Screenshot saved: {screenshot_path}")
                return
        except Exception as e:
            print(f"  OffscreenRenderer failed ({e}), trying legacy Visualizer...")

        # Fallback to legacy Visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1920, height=1080)

        opt = vis.get_render_option()
        opt.background_color = np.array(args.bg_color)
        opt.point_size = args.point_size

        for g in o3d_geoms:
            vis.add_geometry(g)

        vis.poll_events()
        vis.update_renderer()
        # Reset view to see all geometry
        ctr = vis.get_view_control()
        ctr.set_zoom(0.7)

        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image(screenshot_path)
        vis.destroy_window()
        print(f"Screenshot saved: {screenshot_path}")
    else:
        # Interactive viewer
        print("\nOpen3D viewer controls:")
        print("  Mouse drag  : rotate")
        print("  Scroll      : zoom")
        print("  Shift+drag  : pan")
        print("  Q / Esc     : quit")

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="VGGT Viewer", width=1280, height=720)

        opt = vis.get_render_option()
        opt.background_color = np.array(args.bg_color)
        opt.point_size = args.point_size

        for g in o3d_geoms:
            vis.add_geometry(g)

        vis.run()
        vis.destroy_window()


def view_with_matplotlib(geometries_info, args):
    """Fallback viewer using matplotlib (limited but always available)."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for filepath, geom in geometries_info:
        if isinstance(geom, trimesh.PointCloud):
            pts = np.asarray(geom.vertices)
            if args.subsample and len(pts) > args.subsample:
                indices = np.random.choice(len(pts), args.subsample, replace=False)
                pts = pts[indices]
                colors = np.asarray(geom.colors)[indices] if geom.colors is not None and len(geom.colors) > 0 else None
            else:
                colors = np.asarray(geom.colors) if geom.colors is not None and len(geom.colors) > 0 else None

            if colors is not None:
                c = colors[:, :3].astype(np.float64) / 255.0
            else:
                c = "steelblue"

            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, s=0.2, alpha=0.6)

        elif isinstance(geom, trimesh.Trimesh):
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            verts = np.asarray(geom.vertices)
            faces = np.asarray(geom.faces)
            # Subsample faces for performance
            max_faces = 50000
            if len(faces) > max_faces:
                indices = np.random.choice(len(faces), max_faces, replace=False)
                faces = faces[indices]

            poly = Poly3DCollection(verts[faces], alpha=0.3, edgecolor="gray", linewidth=0.1)
            poly.set_facecolor("lightblue")
            ax.add_collection3d(poly)

            # Set limits
            ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
            ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
            ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("VGGT Viewer")

    if args.auto_screenshot or args.screenshot:
        screenshot_path = args.screenshot or "screenshot.png"
        plt.savefig(screenshot_path, dpi=150, bbox_inches="tight")
        print(f"Screenshot saved: {screenshot_path}")
        plt.close()
    else:
        # Interactive display — need to switch backend before plt.show()
        # matplotlib.use() must happen before figure creation to work reliably,
        # so we save the figure, switch backend, and re-display.
        print("\nMatplotlib viewer (interactive).")
        print("  Drag to rotate, scroll to zoom. Close window to exit.")
        try:
            plt.switch_backend("TkAgg")
            plt.show()
        except Exception as e:
            print(f"  Interactive display failed: {e}")
            print(f"  Saving to screenshot.png instead")
            plt.savefig("screenshot.png", dpi=150, bbox_inches="tight")
            plt.close()


def main():
    args = parse_args()

    # Load all files
    geometries_info = []
    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"ERROR: File not found: {filepath}")
            continue
        geom, desc = load_geometry(filepath)
        if geom is not None:
            print(f"Loaded {filepath}: {desc}")
            geometries_info.append((filepath, geom))

    if not geometries_info:
        print("No valid geometry files loaded. Exiting.")
        sys.exit(1)

    # Info-only mode
    if args.info:
        for filepath, geom in geometries_info:
            print_info(filepath, geom)
        return

    # Multi-view requires Open3D
    if args.multi_view and not HAS_O3D:
        print("ERROR: --multi-view requires Open3D. Install with: pip install open3d")
        sys.exit(1)

    # Try Open3D first (best experience), fall back to matplotlib
    if HAS_O3D:
        view_with_open3d(geometries_info, args)
    elif HAS_MPL:
        print("Open3D not available. Falling back to matplotlib viewer.")
        print("  For better experience: pip install open3d")
        view_with_matplotlib(geometries_info, args)
    else:
        print("ERROR: Neither Open3D nor matplotlib is available.")
        print("  Install one:  pip install open3d   OR   pip install matplotlib")
        # Still print info as a fallback
        for filepath, geom in geometries_info:
            print_info(filepath, geom)


if __name__ == "__main__":
    main()
