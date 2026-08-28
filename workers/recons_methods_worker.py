#!/usr/bin/env python3
"""Subprocess worker: multiple reconstruction methods for determinism testing.

Usage:
    python recons_methods_worker.py input.ply output.ply --method METHOD [--seed SEED]
"""
import os
import sys
import time
import traceback
import numpy as np
import open3d as o3d
import trimesh


SPARSE_POINT_LIMIT = 5000


# Ceiling on points handed to a reconstruction method. Above this, Delaunay
# tetrahedralisation time grows faster than the extra detail is worth.
MAX_RECON_POINTS = 90000


def _load_and_prep(input_path, seed=None):
    """Load pcd, downsample if huge, estimate+orient normals. Returns (pcd, num_points)."""
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        o3d.utility.random.seed(seed)

    pcd = o3d.io.read_point_cloud(input_path)
    num_points = len(pcd.points)
    print(f"Points: {num_points:,}")

    # Only decimate genuinely oversized clouds, so the density chosen upstream
    # survives instead of being silently undone here.
    if num_points > MAX_RECON_POINTS:
        bbox_extent = pcd.get_axis_aligned_bounding_box().get_max_extent()
        voxel_size = bbox_extent / 350
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        attempts = 0
        while len(pcd.points) > MAX_RECON_POINTS and attempts < 10:
            voxel_size *= 1.15
            pcd = o3d.io.read_point_cloud(input_path)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            attempts += 1
        print(f"Downsampled: {num_points:,} -> {len(pcd.points):,}")

    n = len(pcd.points)
    _dbg_t = time.time()
    if n <= SPARSE_POINT_LIMIT:
        # Radius search on a sparse cloud spans a large fraction of the object,
        # which flattens normals across real curvature. kNN stays scale-free.
        k = int(np.clip(round(n * 0.02), 12, 40))
        search_param = o3d.geometry.KDTreeSearchParamKNN(knn=k)
        param_label = f"knn={k}"
    else:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = max(avg_dist * 4.0, 0.005)
        max_nn = min(max(30, int(n * 0.01)), 100)
        search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        param_label = f"hybrid max_nn={max_nn}"
    pcd.estimate_normals(search_param=search_param)
    print(f"[DBG-prep] estimate_normals({param_label}): {time.time() - _dbg_t:.2f}s")

    oriented = False
    for k in [min(len(pcd.points) // 100, 15), 10]:
        if k < 5:
            break
        try:
            _dbg_t = time.time()
            pcd.orient_normals_consistent_tangent_plane(k)
            print(f"[DBG-prep] orient_tangent_plane(k={k}): {time.time() - _dbg_t:.2f}s")
            print(f"Normals: tangent_plane(k={k})")
            oriented = True
            break
        except Exception:
            continue
    if not oriented:
        centroid = np.mean(np.asarray(pcd.points), axis=0)
        pcd.orient_normals_towards_camera_location(centroid)
        print("Normals: camera_location")

    return pcd


def _is_closed(mesh):
    """Watertight by the definition the rest of the pipeline uses (trimesh)."""
    try:
        t = trimesh.Trimesh(np.asarray(mesh.vertices),
                            np.asarray(mesh.triangles), process=False)
        return bool(t.is_watertight)
    except Exception as exc:
        # Falling back to Open3D changes the DEFINITION, and this is the
        # criterion the alpha ladder selects on. Open3D is stricter about
        # vertex-manifoldness and disagrees with trimesh on meshes Stages 4-6
        # all treat as closed, so a silent fallback can quietly change which
        # mesh gets chosen -- and Stage 6 then integrates one that is not
        # closed by the definition it uses.
        print(f"  WARNING: trimesh watertight check failed "
              f"({type(exc).__name__}: {exc}) — falling back to Open3D's "
              f"is_watertight, which is a DIFFERENT definition from the one "
              f"Stages 4-6 use")
        return bool(mesh.is_watertight())


def _post_process(mesh, densities=None, n_input=None):
    """Cleanup + density filter + largest component. Returns clean mesh."""
    if densities is not None and len(densities) > 0:
        # Sparse clouds have genuinely low density everywhere, so the usual 5%
        # cut trims real surface (thin limb tips) rather than Poisson spill.
        q = 0.02 if (n_input is not None and n_input <= SPARSE_POINT_LIMIT) else 0.05
        threshold = np.quantile(densities, q)
        before = len(mesh.vertices)
        try:
            mesh.remove_vertices_by_mask(densities < threshold)
        except Exception:
            pass
        if len(mesh.vertices) != before:
            print(f"Density filter ({q*100:.0f}%): {before:,} -> {len(mesh.vertices):,}")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()

    # remove_non_manifold_edges deletes every triangle touching a non-manifold
    # edge, which can tear a closed surface open. The alpha search selects its
    # alpha *because* the mesh is watertight with euler 2, so letting cleanup
    # undo that discards the whole point of the search — on small_leg it turned
    # the selected alpha=25x mesh from watertight into an open shell with a hole
    # in the back of the calf, and Stage 6 then fell back to a leaking flood
    # fill. Snapshot first and revert if these steps open it.
    # Use trimesh's definition, not Open3D's. They disagree — on small_leg the
    # selected alpha=25x mesh is watertight to trimesh (euler 2) and not to
    # Open3D, which is stricter about vertex-manifoldness. The alpha search,
    # Stage 5 and Stage 6 all use trimesh, so guarding on Open3D would protect
    # a property nothing downstream reads and miss the one that matters.
    _closed_before = _is_closed(mesh)
    _snapshot = o3d.geometry.TriangleMesh(mesh) if _closed_before else None

    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    print(f"Cleanup: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} faces")

    try:
        cluster_ids, cluster_n_tris, _ = mesh.cluster_connected_triangles()
        cluster_ids = np.asarray(cluster_ids)
        cluster_n_tris = np.asarray(cluster_n_tris)
        if len(cluster_n_tris) > 1:
            largest = int(np.argmax(cluster_n_tris))
            remove_mask = cluster_ids != largest
            before = len(mesh.triangles)
            mesh.remove_triangles_by_mask(remove_mask)
            mesh.remove_unreferenced_vertices()
            mesh.compute_vertex_normals()
            print(f"Largest component: kept {len(mesh.triangles):,}/{before:,} faces "
                  f"({len(mesh.triangles)/before*100:.1f}%), "
                  f"{len(cluster_n_tris)} components total")
    except Exception as e:
        print(f"Component filtering skipped: {e}")

    if _closed_before and not _is_closed(mesh):
        print("Cleanup opened a watertight mesh — reverting to the pre-cleanup "
              "version (a closed surface has one component, so the component "
              "filter had nothing to remove anyway)")
        mesh = _snapshot
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

    return mesh


def _adaptive_poisson_depth(n_points):
    """Octree depth matched to sample count.

    Depth d resolves a 2^d grid; at depth 9 that is 512^3 cells for a cloud of
    ~1k points, so the solver interpolates far more detail than the data
    supports and the surface wobbles.
    """
    if n_points < 2000:
        return 6
    if n_points < 10000:
        return 7
    if n_points < 50000:
        return 8
    return 9


def method_poisson(pcd, output_path, seed=None):
    import tempfile
    import subprocess as sp

    mesh, densities = None, None
    n_input = len(pcd.points)
    d_start = _adaptive_poisson_depth(n_input)
    print(f"Poisson depth start: {d_start} (for {n_input:,} points)")

    with tempfile.TemporaryDirectory() as tmpdir:
        pcd_path = os.path.join(tmpdir, "pcd_with_normals.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)

        for d in range(d_start, max(d_start - 3, 4), -1):
            mesh_out = os.path.join(tmpdir, f"mesh_d{d}.ply")
            dens_out = os.path.join(tmpdir, f"dens_d{d}.npy")

            seed_lines = ""
            if seed is not None:
                seed_lines = f"""
import random
random.seed({seed})
np.random.seed({seed})
o3d.utility.random.seed({seed})
os.environ.setdefault('PYTHONHASHSEED', '0')
"""
            script = f"""
import sys, os, numpy as np, open3d as o3d
{seed_lines}
pcd = o3d.io.read_point_cloud({pcd_path!r})
try:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth={d})
    densities = np.asarray(densities)
    if len(mesh.triangles) == 0:
        sys.exit(2)
    o3d.io.write_triangle_mesh({mesh_out!r}, mesh)
    np.save({dens_out!r}, densities)
    sys.exit(0)
except Exception as e:
    print(f'failed: {{e}}', file=sys.stderr)
    sys.exit(1)
"""
            _dbg_t = time.time()
            r = sp.run([sys.executable, "-c", script],
                        capture_output=True, text=True, timeout=300)
            print(f"[DBG-poisson] depth={d}: {time.time() - _dbg_t:.2f}s rc={r.returncode}")
            if r.returncode == 0 and os.path.exists(mesh_out):
                mesh = o3d.io.read_triangle_mesh(mesh_out)
                densities = np.load(dens_out)
                print(f"Poisson: depth={d}, {len(mesh.vertices):,} verts, "
                      f"{len(mesh.triangles):,} faces")
                break
            err = r.stderr.strip()[:100] if r.stderr.strip() else f"exit {r.returncode}"
            print(f"Poisson depth={d} failed: {err}")

    if mesh is None or len(mesh.triangles) == 0:
        print("ERROR: Poisson failed at all depths")
        sys.exit(1)

    mesh = _post_process(mesh, densities, n_input=n_input)
    return mesh


# Alpha is searched as multiples of mean nearest-neighbour spacing, smallest
# first, and the first value that yields a sound closed surface wins.
#
# The ladder used to stop at 90x, and that ceiling was reached in practice: a
# limb cloud that closes cleanly at 140x reported "no watertight alpha found"
# and fell through to a repair path, one rung short of the answer. It was not
# under-sampled or holed -- its Euler number fell monotonically 2202, 868, 654,
# 398, 222, 102, 16, 2 with no boundary edges at any step, which is a surface
# converging, not a surface failing.
#
# Extended so that case terminates. The order still matters more than the
# ceiling: since the smallest sound alpha wins, adding coarse rungs cannot make
# a cloud that already closes finely choose a coarse one. A cloud that only
# closes out here is a real result but a blunt one -- at 140x the alpha ball is
# wider than the limb, and the surface it returns sits at 67% of the convex hull
# against 50% for a cloud that closes at 25x, meaning genuine concavities have
# been bridged. Closing coarsely is better than not closing; it is not as good
# as closing finely.
ALPHA_MULTIPLIERS = [8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 25.0, 30.0,
                     40.0, 55.0, 70.0, 90.0, 115.0, 140.0, 170.0, 200.0]


def method_alpha_shape(pcd, output_path, seed=None):
    """Interpolating reconstruction — surface passes through the actual points.

    Poisson fits a smooth *approximating* implicit surface, and that smoothness
    prior rounds off flat faces and sharp rims, losing real volume. Alpha shape
    interpolates instead, so sharp features survive.

    Alpha is selected as the smallest multiplier producing a mesh that is both
    watertight AND topologically sound (Euler number 2, i.e. a simple closed
    surface with no tunnels or cavities).

    Watertightness alone is not sufficient. A surface riddled with tunnels is
    still closed, and the signed-volume integral faithfully subtracts those
    tunnels — so the mesh looks like a perfect cube from outside while reporting
    31% too little volume. Measured on a reference cube: alpha 30 gave the first
    watertight mesh at Euler -1 and 1898 cm3; alpha 40 was the first at Euler 2
    and gave 2467 cm3 against a 2744 cm3 nominal.

    Selecting on "first alpha returning any triangles" is worse still: it picks
    the smallest alpha and returns a shredded non-manifold shell (-91% on a
    known can).
    """
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        o3d.utility.random.seed(seed)

    avg_dist = float(np.mean(pcd.compute_nearest_neighbor_distance()))

    # Build the Delaunay tetrahedralisation ONCE and reuse it for every alpha.
    # Open3D otherwise rebuilds it on each call and alpha merely filters which
    # tetrahedra survive — so a 12-value sweep paid for 12 tetrahedralisations.
    # That dominates the stage: the points lie on a surface, which is the
    # pathological case for 3D Delaunay (huge numbers of sliver tetrahedra), so
    # cost grows faster than quadratically in point count. Measured 4.3x faster
    # on 30k points over 5 alphas, and the saving grows with both.
    _t0 = time.time()
    tetra, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    print(f"[DBG-alpha] tetrahedralisation ({len(pcd.points):,} pts): "
          f"{time.time() - _t0:.2f}s — reused across all alphas")

    best = None
    fallback = None
    for mul in ALPHA_MULTIPLIERS:
        alpha = avg_dist * mul
        if alpha <= 0.00001:
            continue
        try:
            m = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha, tetra, pt_map)
        except Exception as e:
            print(f"Alpha Shape alpha={mul:.0f}x nn failed: {e}")
            continue
        if len(m.triangles) == 0:
            continue

        m.remove_degenerate_triangles()
        m.remove_duplicated_triangles()
        m.remove_duplicated_vertices()
        m.remove_non_manifold_edges()

        # Drop detached fragments BEFORE judging topology. Alpha shape on a
        # dense cloud often leaves a small satellite blob; two closed surfaces
        # give euler 4, so an otherwise perfect mesh would be rejected for a
        # fragment that _post_process is about to delete anyway.
        try:
            cid, ntri, _ = m.cluster_connected_triangles()
            ntri = np.asarray(ntri)
            if len(ntri) > 1:
                keep = int(np.argmax(ntri))
                m.remove_triangles_by_mask(np.asarray(cid) != keep)
                m.remove_unreferenced_vertices()
        except Exception:
            pass

        t = trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles),
                            process=False)
        wt = bool(t.is_watertight)
        euler = int(t.euler_number)
        sound = wt and euler == 2
        vol = abs(float(t.volume)) if wt else 0.0
        note = ""
        if wt and not sound:
            note = f"  <- closed but euler={euler}: tunnels/cavities, volume invalid"
        print(f"Alpha Shape alpha={mul:>4.0f}x nn: {len(m.triangles):>6,} faces, "
              f"watertight={wt}, euler={euler}"
              + (f", volume={vol:.6f}" if wt else "") + note)

        # Keep the best unsound mesh as a last resort. The inner guard here used
        # to be `if fallback is None`, which meant the fallback was fixed on the
        # FIRST iteration and never revisited -- and the search starts at the
        # smallest alpha, which is the most shredded candidate of all. A run
        # that failed to close then reported the worst mesh it had built rather
        # than the best, turning a partial failure into a total one.
        #
        # Rank instead: closed beats open; among those, the Euler number nearest
        # 2 is nearest to a valid solid; ties go to the mesh carrying more
        # surface, which is the less fragmented one.
        score = (wt, -abs(euler - 2), len(m.triangles))
        if fallback is None or score > fallback[2]:
            fallback = (m, mul, score)
        if sound:
            best = (m, mul, vol)
            break

    if best is not None:
        mesh, mul, vol = best
        print(f"Alpha Shape: selected alpha={mul:.0f}x nn "
              f"(smallest watertight AND euler=2), volume={vol:.6f}")
    elif fallback is not None:
        mesh, mul, score = fallback
        print(f"WARNING: no watertight alpha found — using the best candidate, "
              f"alpha={mul:.0f}x nn (watertight={bool(score[0])}, "
              f"euler={2 - score[1] if score[1] <= 0 else 2 + score[1]}, "
              f"{score[2]:,} faces; Stage 5 will attempt repair)")
    else:
        print("ERROR: Alpha Shape failed at all alphas")
        sys.exit(1)

    mesh = _post_process(mesh)
    return mesh


BOX_SQUARENESS_TOL = 0.20


def _fit_yaw_footprint(xy):
    """Minimum-area rectangle over yaw. Returns (yaw_deg, edges, min_corner)."""
    def measure(deg):
        t = np.radians(deg)
        c, s = np.cos(t), np.sin(t)
        aligned = xy @ np.array([[c, s], [-s, c]])
        lo, hi = aligned.min(0), aligned.max(0)
        e = hi - lo
        return e[0] * e[1], e, lo

    best = None
    for deg in np.arange(0.0, 90.0, 0.5):
        cand = (measure(deg), deg)
        if best is None or cand[0][0] < best[0][0]:
            best = cand
    centre_deg = best[1]
    for deg in np.arange(centre_deg - 0.5, centre_deg + 0.5, 0.02):
        cand = (measure(deg), deg)
        if cand[0][0] < best[0][0]:
            best = cand

    (_, edges, lo), yaw = best
    return float(yaw), edges, lo


def method_box_primitive(pcd, output_path, seed=None):
    """Reconstruct the ArUco reference as a rectangular prism.

    The reference is a known box, so fitting the primitive beats a generic
    surface solver: Poisson rounds off every corner and edge, which biases the
    volume that the whole Stage 6 scale factor is derived from.

    All three extents are measured, never forced equal. Forcing a cube would
    make the downstream "reference measures 14 cm" check circular — it would
    hold by construction however bad the input is. Leaving Z free keeps the
    deviation from 1:1:1 usable as a genuine accuracy estimate.
    """
    pts = np.asarray(pcd.points, dtype=np.float64)
    if len(pts) < 20:
        print(f"ERROR: only {len(pts)} points — too few to fit a box")
        sys.exit(1)

    yaw, edges, lo_xy = _fit_yaw_footprint(pts[:, :2])
    fx, fy = float(edges[0]), float(edges[1])
    z_min, z_max = float(pts[:, 2].min()), float(pts[:, 2].max())
    height = z_max - z_min

    squareness = abs(fx - fy) / max(fx, fy)
    print(f"Footprint fit: yaw={yaw:.2f}deg  {fx:.4f} x {fy:.4f}  "
          f"(squareness err {squareness*100:.1f}%)")

    if squareness > BOX_SQUARENESS_TOL:
        print(f"WARNING: footprint not square (>{BOX_SQUARENESS_TOL*100:.0f}%) — "
              f"falling back to poisson")
        return method_poisson(pcd, output_path, seed=seed)

    mean_side = (fx + fy) / 2.0
    cubeness = abs(height - mean_side) / mean_side
    print(f"Prism fit: {fx:.4f} x {fy:.4f} x {height:.4f}  "
          f"(Z/XY={height/mean_side:.3f}, cubeness err {cubeness*100:.1f}%)")

    box = o3d.geometry.TriangleMesh.create_box(width=fx, height=fy, depth=height)
    box.translate((lo_xy[0], lo_xy[1], z_min))

    Rz = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, -np.radians(yaw)))
    box.rotate(Rz, center=(0, 0, 0))

    if pcd.has_colors():
        box.paint_uniform_color(np.asarray(pcd.colors).mean(axis=0))

    box.compute_vertex_normals()
    box.compute_triangle_normals()
    print(f"Box primitive: {len(box.vertices)} verts, {len(box.triangles)} faces, "
          f"volume={box.get_volume():.6f}")
    return box


def method_poisson_omp1(pcd, output_path, seed=None):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    return method_poisson(pcd, output_path, seed=seed)


METHODS = {
    "poisson": method_poisson,
    "alpha_shape": method_alpha_shape,
    "poisson_omp1": method_poisson_omp1,
    "box_primitive": method_box_primitive,
}


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    method_name = "poisson"
    seed = None
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--method" and i + 1 < len(sys.argv):
            method_name = sys.argv[i + 1]
            i += 2
        elif arg == "--seed" and i + 1 < len(sys.argv):
            seed = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    if method_name not in METHODS:
        print(f"ERROR: unknown method '{method_name}'. Choices: {list(METHODS)}")
        sys.exit(1)

    print(f"Method: {method_name}")
    if seed is not None:
        print(f"Seed: {seed}")

    _dbg_t = time.time()
    pcd = _load_and_prep(input_path, seed=None)  # seed worker-level rng
    print(f"[DBG-worker] load_and_prep: {time.time() - _dbg_t:.2f}s")

    _dbg_t = time.time()
    mesh = METHODS[method_name](pcd, output_path, seed=seed)
    print(f"[DBG-worker] method {method_name}: {time.time() - _dbg_t:.2f}s")

    o3d.io.write_triangle_mesh(output_path, mesh)
    # _is_closed, not mesh.is_watertight(): Open3D is stricter about
    # vertex-manifoldness and calls meshes open that Stages 4-6 treat as
    # closed, so reporting its answer here contradicts what the pipeline
    # goes on to do with the mesh.
    print(f"Watertight: {_is_closed(mesh)}")
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
