"""Stage 6 — Compute real-world volumes using the ArUco reference (box).

Scale derivation (from a measured LENGTH, not a volume ratio):
    linear_scale = REFERENCE_REAL_SIZE_CM / mean(ref OBB edges)   # cm per unit
    k            = linear_scale ** 3                              # cm³ per unit³
    real_vol     = mesh_vol * k
    size_cm      = mesh OBB extents * linear_scale

Deriving scale as (real_vol / mesh_vol)^(1/3) instead uses mesh_vol^(1/3) as the
reference edge, which only holds for a perfect cube; the cube root compounds any
deviation three times. At 2.2% off cubic that under-read the edge by 3.1% and
inflated every volume ~10%. Using the edge directly also leaves the reference's
own volume free to disagree with nominal — that gap is a real error bar rather
than something forced to zero.

Volume method priority (per mesh):
    1. watertight        — exact signed volume, no discretisation error
    2. warp+floodfill    — GPU BVH surface mark + CPU flood-fill (leaks on open meshes)
    3. trimesh voxel     — CPU fallback
    4. convex_hull       — unreliable; ignores the surface entirely

Voxel occupancy is also computed alongside the exact value as an independent
cross-check. It over-reads by a few percent (boundary voxels counted whole) and
converges downward onto exact; a result below exact indicates a self-intersecting
or inverted surface.

The reference is a 14 × 14 × 14 cm ArUco cube identified by 'box' in its filename.
"""
import os
import numpy as np
import pandas as pd
import trimesh
from scipy import ndimage

from pipeline.config import REFERENCE_REAL_SIZE_CM

DEFAULT_VOXEL_RES = 150

# warp optional — GPU voxelization
try:
    import warp as wp
    _WARP_AVAILABLE = True
except ImportError:
    _WARP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Warp GPU kernel (must be at module level — warp requires file scope)
# ---------------------------------------------------------------------------

if _WARP_AVAILABLE:
    wp.init()

    @wp.kernel
    def _warp_mark_surface(
        mesh_id:   wp.uint64,
        surface:   wp.array3d(dtype=wp.int32),
        b_min:     wp.vec3,
        pitch:     float,
        threshold: float,
    ):
        i, j, k = wp.tid()
        cx = b_min[0] + (float(i) + 0.5) * pitch
        cy = b_min[1] + (float(j) + 0.5) * pitch
        cz = b_min[2] + (float(k) + 0.5) * pitch
        q = wp.mesh_query_point_sign_normal(mesh_id, wp.vec3(cx, cy, cz), threshold)
        if q.result:
            surface[i, j, k] = 1


# ---------------------------------------------------------------------------
# Volume methods
# ---------------------------------------------------------------------------

def _volume_voxel_warp(mesh: trimesh.Trimesh, resolution: int,
                        device: str = "cuda:0") -> float:
    """GPU surface-mark (warp BVH) + CPU flood-fill (scipy)."""
    verts = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)

    wp_mesh = wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device=device),
        indices=wp.array(faces.flatten(), dtype=wp.int32, device=device),
    )

    b_min  = verts.min(axis=0)
    b_max  = verts.max(axis=0)
    pitch  = float((b_max - b_min).max()) / resolution
    nx = max(1, int(np.ceil((b_max[0] - b_min[0]) / pitch)))
    ny = max(1, int(np.ceil((b_max[1] - b_min[1]) / pitch)))
    nz = max(1, int(np.ceil((b_max[2] - b_min[2]) / pitch)))

    threshold   = pitch * (3.0 ** 0.5) / 2.0
    surface_gpu = wp.zeros((nx, ny, nz), dtype=wp.int32, device=device)
    b_min_wp    = wp.vec3(float(b_min[0]), float(b_min[1]), float(b_min[2]))

    wp.launch(_warp_mark_surface, dim=(nx, ny, nz),
              inputs=[wp_mesh.id, surface_gpu, b_min_wp, pitch, threshold],
              device=device)
    wp.synchronize()

    surface_np     = surface_gpu.numpy().astype(bool)
    struct         = ndimage.generate_binary_structure(3, 1)
    padded         = np.pad(surface_np, 1, constant_values=False)
    labeled, _     = ndimage.label(~padded, structure=struct)
    exterior_label = int(labeled[0, 0, 0])
    interior       = (~padded) & (labeled != exterior_label)
    interior       = interior[1:-1, 1:-1, 1:-1]

    return float((interior.sum() + surface_np.sum()) * pitch**3)


def _volume_voxel(mesh: trimesh.Trimesh, resolution: int) -> float:
    pitch = float(mesh.extents.max()) / resolution
    return float(mesh.voxelized(pitch=pitch).fill().volume)


def _volume_convex_hull(mesh: trimesh.Trimesh) -> float:
    return float(abs(mesh.convex_hull.volume))


def auto_tune_voxel_res(
    mesh: trimesh.Trimesh,
    start: int = 50,
    stop: int = 300,
    step: int = 50,
    tol: float = 0.015,
    use_warp: bool = False,
) -> tuple[int, float]:
    """Increase resolution until volume converges. Returns (best_res, volume)."""
    backend = "warp+floodfill" if use_warp else "trimesh"
    print(f"  {'res':>6}  {'pitch':>10}  {'volume':>14}  {'Δ%':>8}  [{backend}]")
    print(f"  {'-'*6}  {'-'*10}  {'-'*14}  {'-'*8}")

    prev_vol = None
    best_res = start
    best_vol = None

    for res in range(start, stop + 1, step):
        pitch = float(mesh.extents.max()) / res
        try:
            if use_warp:
                vol = _volume_voxel_warp(mesh, res)
            else:
                vol = float(mesh.voxelized(pitch=pitch).fill().volume)
        except Exception as e:
            print(f"  {res:>6}  {pitch:>10.5f}  {'ERROR':>14}  {'—':>8}  (skipped: {e})")
            continue

        if prev_vol is not None and prev_vol > 0:
            delta_pct = abs(vol - prev_vol) / prev_vol * 100
            print(f"  {res:>6}  {pitch:>10.5f}  {vol:>14.6f}  {delta_pct:>7.2f}%")
            if delta_pct < tol * 100:
                best_res = res
                best_vol = vol
                print(f"\n  converged at res={res}  (Δ < {tol*100:.1f}%)")
                return best_res, best_vol
        else:
            print(f"  {res:>6}  {pitch:>10.5f}  {vol:>14.6f}  {'—':>8}")

        prev_vol = vol
        best_res = res
        best_vol = vol

    print(f"\n  [warn] did not converge by res={stop}. using res={best_res}.")
    return best_res, best_vol


def _measure_volume(mesh: trimesh.Trimesh,
                    voxel_res: int = DEFAULT_VOXEL_RES,
                    auto_res: bool = False) -> tuple[float, str]:
    """Return (volume_mesh_units3, method_label) using best available method."""
    if mesh.is_watertight:
        return float(abs(mesh.volume)), "watertight"

    # Not closed: everything below is an approximation, and flood-fill in
    # particular leaks through holes and can under-read by an order of
    # magnitude while still returning a plausible-looking number.
    print("  WARNING: mesh is NOT watertight — falling back to voxel "
          "approximation. Flood fill leaks through open surfaces; treat this "
          "volume as unreliable.")

    if _WARP_AVAILABLE:
        try:
            if auto_res:
                best_res, vol = auto_tune_voxel_res(mesh, use_warp=True)
                method = f"warp+floodfill (auto res={best_res})"
            else:
                vol = _volume_voxel_warp(mesh, voxel_res)
                method = f"warp+floodfill (res={voxel_res})"
            if vol > 0:
                return vol, method
        except Exception as e:
            print(f"  [warn] warp failed: {e} — falling back to trimesh")

    try:
        if auto_res:
            best_res, vol = auto_tune_voxel_res(mesh, use_warp=False)
            method = f"trimesh voxel (auto res={best_res})"
        else:
            vol = _volume_voxel(mesh, voxel_res)
            method = f"trimesh voxel (res={voxel_res})"
        if vol > 0:
            return vol, method
    except Exception as e:
        print(f"  [warn] trimesh voxel failed: {e}")

    try:
        print("  WARNING: convex hull fallback — ignores the surface entirely and "
              "only uses vertex positions. A broken mesh scores the same as a good "
              "one. Do not report this as a measurement.")
        return _volume_convex_hull(mesh), "convex_hull (UNRELIABLE)"
    except Exception as e:
        print(f"  [ERROR] all volume methods failed: {e}")
        return 0.0, "failed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_ref_mesh(path: str) -> bool:
    name = os.path.splitext(os.path.basename(path).lower())[0]
    return "box" in name


def _load_mesh_info(path: str, voxel_res: int, auto_res: bool = False) -> dict | None:
    if not os.path.exists(path):
        print(f"  skipping (not found): {path}")
        return None

    mesh = trimesh.load(path, force="mesh", process=False)
    if len(mesh.vertices) == 0:
        print(f"  skipping (empty): {path}")
        return None

    mesh.merge_vertices()
    volume, method = _measure_volume(mesh, voxel_res, auto_res=auto_res)

    # Oriented extents, not axis-aligned. An AABB around a yaw-rotated object
    # reports its diagonal, not its size — the can measures 6.09 cm wide by
    # AABB and 5.75 cm by OBB, and the cube 14.95 vs 14.0.
    # Keep the OBB extents ordered by ORIENTATION, not magnitude: index 0 is the
    # axis most aligned with world up. Stage 3 levels the scene, so that axis is
    # the one the floor truncates — identifying it by geometry beats guessing
    # from which edges happen to agree, which picks the wrong edge whenever two
    # are short in the same direction.
    _ob = mesh.bounding_box_oriented
    _ext = np.asarray(_ob.primitive.extents, dtype=float)
    _R = np.asarray(_ob.primitive.transform)[:3, :3]
    _align = [abs(float(np.dot(_R[:, i] / (np.linalg.norm(_R[:, i]) + 1e-12),
                               np.array([0.0, 0.0, 1.0])))) for i in range(3)]
    _vi = int(np.argmax(_align))
    _hi = [i for i in range(3) if i != _vi]
    obb = np.array([_ext[_vi], _ext[_hi[0]], _ext[_hi[1]]])  # [vertical, horiz, horiz]

    # Euler number is the decisive integrity check. A closed surface with
    # tunnels bounds a region perfectly well, so is_watertight passes and the
    # volume integral correctly subtracts the tunnels — the mesh looks right
    # from outside while reporting far too little volume. Only 2 means a simple
    # closed surface.
    euler = int(mesh.euler_number)
    if euler != 2:
        print(f"  WARNING: {os.path.basename(path)} has euler number {euler} "
              f"(expected 2) — the surface has tunnels or cavities and its "
              f"volume is NOT reliable.")

    # Independent cross-check: voxel occupancy of the same mesh. It over-reads
    # by a predictable few percent (boundary voxels counted whole) and converges
    # down onto the exact value. A voxel result *below* exact, or wildly above,
    # means the surface is self-intersecting or wrongly wound.
    voxel_check = None
    if method == "watertight":
        try:
            v = (_volume_voxel_warp(mesh, voxel_res) if _WARP_AVAILABLE
                 else _volume_voxel(mesh, voxel_res))
            voxel_check = float(v)
        except Exception:
            pass

    return {
        "name":     os.path.basename(path),
        "is_ref":   _is_ref_mesh(path),
        "volume":   volume,
        "method":   method,
        "obb_a":    float(obb[0]),
        "obb_b":    float(obb[1]),
        "obb_c":    float(obb[2]),
        "voxel":    voxel_check,
        "euler":    euler,
    }


# ---------------------------------------------------------------------------
# Main stage entry-point
# ---------------------------------------------------------------------------

def compute_volumes(object_mesh_paths: list[str],
                    voxel_res: int = DEFAULT_VOXEL_RES,
                    auto_res: bool = True):
    """Compute real-world volume of each object using ArUco box for scale."""
    res_label = "auto" if auto_res else str(voxel_res)
    print()
    print("=" * 60)
    print(f"STAGE 6: real-world volumes  "
          f"(ref = {REFERENCE_REAL_SIZE_CM} cm ArUco cube  |  voxel_res={res_label})")
    print("=" * 60)

    rows = [_load_mesh_info(p, voxel_res, auto_res=auto_res) for p in object_mesh_paths]
    rows = [r for r in rows if r is not None]
    if not rows:
        print("  No meshes loaded.")
        return

    df = pd.DataFrame(rows)

    raw_cols = ["name", "method", "euler", "volume", "obb_a", "obb_b", "obb_c"]
    print("\n  Raw mesh measurements (mesh units; obb_a = vertical axis, "
          "obb_b/c horizontal):")
    print(df[raw_cols].to_string(index=False))

    # Voxel occupancy cross-check. Discretisation counts boundary voxels whole,
    # so voxel should sit a few percent ABOVE exact and converge down onto it.
    # Below exact, or far above, means a self-intersecting or inverted surface.
    if df["voxel"].notna().any():
        print("\n  Voxel cross-check (independent; expect +1..8% over exact):")
        for _, r in df.iterrows():
            if r["voxel"] is None or not np.isfinite(r["voxel"]) or r["volume"] <= 0:
                continue
            dev = (r["voxel"] - r["volume"]) / r["volume"] * 100
            flag = "" if 0 <= dev <= 15 else "   <-- SUSPECT: mesh may be self-intersecting"
            print(f"    {r['name']:<16} exact {r['volume']:.6f}  "
                  f"voxel {r['voxel']:.6f}  {dev:+.2f}%{flag}")

    ref_rows = df[df["is_ref"]]
    if ref_rows.empty:
        print("\n  No box (reference) mesh found — cannot scale. Aborting.")
        return

    ref = ref_rows.iloc[0]
    if ref["volume"] <= 0:
        print(f"\n  Reference mesh '{ref['name']}' has zero volume. Aborting.")
        return

    # Scale from a measured LENGTH, not a volume ratio.
    #
    # The old route was linear_scale = (real_vol / mesh_vol)^(1/3), which uses
    # mesh_vol^(1/3) as the reference's edge — only valid if the mesh is a
    # perfect cube. Any deviation from cubic is compounded three times by the
    # cube root: at 2.2% off cubic the edge came out 3.1% short and inflated
    # every reported volume by ~10%.
    #
    # Scale comes from the two HORIZONTAL edges. obb_a is the vertical one (see
    # _load_mesh_info) and is the axis the floor truncates, so it is excluded
    # rather than averaged in — including it drags the estimate small and
    # inflates every downstream volume.
    #
    # The two horizontal edges are then independent measurements of the same
    # physical 14 cm length, and how far they disagree is a genuine error bar.
    ref_v = float(ref["obb_a"])                       # vertical
    ref_h = np.array([ref["obb_b"], ref["obb_c"]], dtype=float)
    ref_edge = float(ref_h.mean())
    h_disagree = float(abs(ref_h[0] - ref_h[1]) / ref_edge * 100)
    v_deficit = (ref_edge - ref_v) / ref_edge * 100

    linear_scale = REFERENCE_REAL_SIZE_CM / ref_edge
    k = linear_scale ** 3

    print(f"\n  Scale factor (from the reference's two horizontal edges):")
    print(f"    horizontal    = {ref_h[0]:.4f}, {ref_h[1]:.4f} units "
          f"— disagree by {h_disagree:.2f}%")
    print(f"    vertical      = {ref_v:.4f} units ({ref_v * REFERENCE_REAL_SIZE_CM / ref_edge:.2f} cm "
          f"of an expected {REFERENCE_REAL_SIZE_CM:.2f}), {v_deficit:+.2f}% short")
    print(f"    ref edge      = {ref_edge:.6f} units  (vertical excluded — it is "
          f"the axis the floor truncates)")
    print(f"    linear_scale  = {REFERENCE_REAL_SIZE_CM} / {ref_edge:.6f} "
          f"= {linear_scale:.6f} cm/unit")
    print(f"    k             = linear_scale³ = {k:.2f} cm³/unit³")
    print(f"    residual height deficit = {(ref_edge - ref_v) * linear_scale:.2f} cm "
          f"— what Stage 3's floor extend did not recover")
    if h_disagree > 3:
        print(f"    WARNING: the two horizontal edges differ by {h_disagree:.1f}% "
              f"— scale is poorly constrained")

    df["real_vol_cm3"] = df["volume"] * k
    df["real_vol_L"]   = df["real_vol_cm3"] / 1000.0
    df["height_cm"]    = df["obb_a"] * linear_scale   # vertical axis
    df["width_cm"]     = df["obb_b"] * linear_scale
    df["depth_cm"]     = df["obb_c"] * linear_scale

    result_cols = ["name", "height_cm", "width_cm", "depth_cm",
                   "real_vol_cm3", "real_vol_L", "method"]
    print("\n  Real-world dimensions (OBB) and volumes:")
    print(df[result_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\n  Reference now reads {df[df['is_ref']].iloc[0]['real_vol_cm3']:.1f} cm³ "
          f"vs {REFERENCE_REAL_SIZE_CM**3:.0f} cm³ nominal — the gap is the "
          f"reference's own reconstruction error, no longer forced to zero.")

    return df
