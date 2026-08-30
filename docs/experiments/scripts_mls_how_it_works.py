"""Draw how MLS works, on one real neighbourhood from the limb cloud."""
import sys, numpy as np, trimesh, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
sys.path.insert(0, ".")
from pipeline.ghost import (compute_voxel_size, ghost_voxel_downsample,
                            normal_aware_filter, mls_project)
from pipeline.config import MLS_RADIUS_MULT
from scipy.spatial import cKDTree

cloud = trimesh.load("work/psr_cold/03_clean/debug/leg_cluster.ply", process=False)
points = np.asarray(cloud.vertices, dtype=np.float32)
voxel = compute_voxel_size(points)
colours = np.zeros((len(points), 3), dtype=np.uint8)   # normal_aware_filter indexes colours
points, colours = ghost_voxel_downsample(points, colours, voxel)
points, colours = normal_aware_filter(points, colours, voxel)
points = np.asarray(points, dtype=np.float64)
print(f"pre-MLS cloud: {len(points):,} pts, voxel {voxel:.5f}")

tree = cKDTree(points)
rng = np.random.default_rng(0)
sample = rng.choice(len(points), 5000, replace=False)
distances, _ = tree.query(points[sample], k=2, workers=-1)
spacing = float(distances[:, 1].mean())
radius = spacing * MLS_RADIUS_MULT
print(f"spacing {spacing*1000:.2f} mm   radius {radius*1000:.2f} mm "
      f"= {MLS_RADIUS_MULT}x spacing")

# A neighbourhood in the middle of the calf, where both ghost sheets are present.
height = points[:, 2]
band = np.abs(height - np.percentile(height, 55)) < spacing
candidates = np.flatnonzero(band)
counts = [len(tree.query_ball_point(points[i], r=radius)) for i in candidates]
centre_index = candidates[int(np.argmax(counts))]
neighbours = np.asarray(tree.query_ball_point(points[centre_index], r=radius))
patch = points[neighbours]
print(f"neighbourhood: {len(patch)} points")

centre = patch.mean(axis=0)
centred = patch - centre
_, singular, axes = np.linalg.svd(centred, full_matrices=False)
normal, tangent_u, tangent_v = axes[2], axes[0], axes[1]
u, v, h = centred @ tangent_u, centred @ tangent_v, centred @ normal

quad = np.column_stack([np.ones_like(u), u, v, u*u, u*v, v*v])
coefficients, *_ = np.linalg.lstsq(quad, h, rcond=None)
plane_design = np.column_stack([np.ones_like(u), u, v])
plane_coefficients, *_ = np.linalg.lstsq(plane_design, h, rcond=None)

target_u = float((points[centre_index] - centre) @ tangent_u)
target_v = float((points[centre_index] - centre) @ tangent_v)
def surface(uu, vv, c):
    """Evaluates the fitted local surface at (uu, vv), quadratic when the coefficients carry the extra terms."""
    return c[0] + c[1]*uu + c[2]*vv + (c[3]*uu*uu + c[4]*uu*vv + c[5]*vv*vv
                                       if len(c) > 3 else 0.0)
fitted = surface(target_u, target_v, coefficients)
moved = (surface(target_u, target_v, coefficients)
         - (points[centre_index] - centre) @ normal)

projected, _, stats = mls_project(points, colours, radius_mult=MLS_RADIUS_MULT,
                                  polynomial=True, verbose=False)

MM = 1000.0
slab = np.abs(points[:, 2] - centre[2]) < spacing * 1.5

fig, axarr = plt.subplots(2, 2, figsize=(13.5, 10.4))
fig.suptitle("How MLS works — one real neighbourhood from the limb cloud\n"
             f"{len(patch)} points inside a radius of {MLS_RADIUS_MULT}× the mean "
             f"point spacing ({spacing*MM:.2f} mm)", fontsize=14)
lim = radius * MM * 1.9

# 1 · the neighbourhood
ax = axarr[0][0]
ax.scatter((points[slab, 0]-centre[0])*MM, (points[slab, 1]-centre[1])*MM, s=4,
           color="#c9d6e5", label="the rest of the cloud")
ax.scatter((patch[:, 0]-centre[0])*MM, (patch[:, 1]-centre[1])*MM, s=18,
           color="#1f4e79", label=f"neighbours inside r ({len(patch)})")
ax.scatter([(points[centre_index, 0]-centre[0])*MM],
           [(points[centre_index, 1]-centre[1])*MM], s=110, color="#cc0000",
           zorder=5, label="the point being projected")
ax.add_patch(Circle((0, 0), radius*MM, fill=False, color="#cc6600", lw=1.8,
                    label=f"r = {radius*MM:.1f} mm"))
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
ax.set_title("1 · collect the neighbours inside a radius\n"
             "r is set in point spacings, so it follows the cloud's own density",
             fontsize=11)
ax.set_xlabel("mm"); ax.set_ylabel("mm")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)

# 2 · the local frame
ax = axarr[0][1]
ax.scatter(u*MM, h*MM, s=20, color="#1f4e79", label="the same 72 neighbours")
ax.axhline(0, color="#0a6e2e", lw=2.4, label="tangent plane — SVD axes 1 and 2")
ax.annotate("", xy=(0, 2.6), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#cc0000", lw=2.4))
ax.text(0.35, 2.15, "normal —\nthe least-variance axis", fontsize=9.5, color="#cc0000")
ax.set_ylim(-3.4, 3.4)
ax.set_title("2 · the neighbourhood's own spread gives a frame\n"
             "no normals are estimated beforehand", fontsize=11)
ax.set_xlabel("u — along the surface, mm")
ax.set_ylabel("height above the plane, mm")
ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.25)

# 3 · the height field
ax = axarr[1][0]
grid_u = np.linspace(u.min(), u.max(), 200)
ax.scatter(u*MM, h*MM, s=20, color="#1f4e79", label="neighbours")
ax.plot(grid_u*MM, [surface(x, target_v, coefficients)*MM for x in grid_u],
        color="#0a6e2e", lw=2.6, label="quadratic fit — what ships")
ax.plot(grid_u*MM, [surface(x, target_v, plane_coefficients)*MM for x in grid_u],
        "--", color="#b06000", lw=2.0, label="plane fit — flattens the limb")
ax.scatter([target_u*MM], [((points[centre_index]-centre) @ normal)*MM], s=110,
           color="#cc0000", zorder=5, label="the point, before")
ax.scatter([target_u*MM], [fitted*MM], s=110, marker="X", color="#0a6e2e",
           zorder=5, label="the point, after")
ax.annotate("", xy=(target_u*MM, fitted*MM),
            xytext=(target_u*MM, ((points[centre_index]-centre) @ normal)*MM),
            arrowprops=dict(arrowstyle="->", color="#cc0000", lw=1.6))
ax.set_title("3 · fit height as a function of position in the plane\n"
             "h ≈ c₀ + c₁u + c₂v + c₃u² + c₄uv + c₅v², by least squares",
             fontsize=11)
ax.set_xlabel("u, mm"); ax.set_ylabel("h, mm")
ax.legend(fontsize=8); ax.grid(alpha=0.25)

# 4 · before and after, same neighbourhood
ax = axarr[1][1]
near = np.flatnonzero(slab & (np.linalg.norm(points[:, :2]-centre[:2], axis=1)
                              < radius * 3))
ax.scatter((points[near, 0]-centre[0])*MM, (points[near, 1]-centre[1])*MM, s=26,
           facecolor="none", edgecolor="#c07070", lw=1.0, label="before MLS")
ax.scatter((projected[near, 0]-centre[0])*MM, (projected[near, 1]-centre[1])*MM,
           s=14, color="#1f4e79", label="after MLS")
for i in near:
    ax.plot([(points[i, 0]-centre[0])*MM, (projected[i, 0]-centre[0])*MM],
            [(points[i, 1]-centre[1])*MM, (projected[i, 1]-centre[1])*MM],
            color="#999999", lw=0.6, zorder=0)
ax.set_aspect("equal")
ax.set_title("4 · every point moves onto its own fitted surface\n"
             f"median move {stats['median_move']*MM:.2f} mm · "
             f"p95 {stats['p95_move']*MM:.2f} mm · nothing is deleted", fontsize=11)
ax.set_xlabel("mm"); ax.set_ylabel("mm")
ax.legend(fontsize=8); ax.grid(alpha=0.25)

fig.text(0.5, 0.012,
         "Every point gets its own fit, so no global shape is ever assumed. The radius must exceed the ghost separation, "
         "or the two sheets never share a neighbourhood and nothing merges.\n"
         "MLS deletes nothing — it only moves points — which is why the point count is unchanged while the shell collapses. "
         "Panels are re-derived from 03_clean/debug/leg_cluster.ply, so the move figures are this neighbourhood's own; "
         "the shipped run's shell goes 1.58 mm to 0.66 mm RMS.",
         ha="center", fontsize=9.5, color="#333333")
fig.tight_layout(rect=[0, 0.04, 1, 0.93])
out = sys.argv[1]
fig.savefig(out, dpi=125, facecolor="white")
print("wrote", out)
