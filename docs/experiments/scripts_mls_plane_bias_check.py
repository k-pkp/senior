"""Test the closed form for plane-MLS's inward bias, pointwise.

Prediction: fitting a plane where the surface is curved puts the fit at the
neighbourhood's MEAN height, which for a uniform disc of radius r is
    E[h] - c0 = (c3 + c5) * r^2 / 4 = H * r^2 / 4
with H the mean curvature. For a cylinder (one curvature 1/R) that is r^2/(8R).
Measured against the actual difference between the two projections.
"""
import sys, numpy as np, trimesh, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from pipeline.ghost import (compute_voxel_size, ghost_voxel_downsample,
                            normal_aware_filter)
from pipeline.config import MLS_RADIUS_MULT
from scipy.spatial import cKDTree

cloud = trimesh.load("work/psr_cold/03_clean/debug/leg_cluster.ply", process=False)
points = np.asarray(cloud.vertices, dtype=np.float32)
colours = np.zeros((len(points), 3), dtype=np.uint8)
voxel = compute_voxel_size(points)
points, colours = ghost_voxel_downsample(points, colours, voxel)
points, colours = normal_aware_filter(points, colours, voxel)
points = np.asarray(points, dtype=np.float64)

tree = cKDTree(points)
rng = np.random.default_rng(0)
sample_for_spacing = rng.choice(len(points), 5000, replace=False)
distances, _ = tree.query(points[sample_for_spacing], k=2, workers=-1)
spacing = float(distances[:, 1].mean())
radius = spacing * MLS_RADIUS_MULT
MM = 1000.0

test_points = rng.choice(len(points), 4000, replace=False)
measured, predicted, curvature_radius = [], [], []
for index in test_points:
    neighbours = tree.query_ball_point(points[index], r=radius)
    if len(neighbours) < 8:
        continue
    patch = points[neighbours]
    centre = patch.mean(axis=0)
    centred = patch - centre
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    normal, tu, tv = axes[2], axes[0], axes[1]
    u, v, h = centred @ tu, centred @ tv, centred @ normal
    quad_design = np.column_stack([np.ones_like(u), u, v, u*u, u*v, v*v])
    plane_design = np.column_stack([np.ones_like(u), u, v])
    cq, *_ = np.linalg.lstsq(quad_design, h, rcond=None)
    cp, *_ = np.linalg.lstsq(plane_design, h, rcond=None)
    u0 = float((points[index] - centre) @ tu)
    v0 = float((points[index] - centre) @ tv)
    h_quad = (cq[0] + cq[1]*u0 + cq[2]*v0
              + cq[3]*u0*u0 + cq[4]*u0*v0 + cq[5]*v0*v0)
    h_plane = cp[0] + cp[1]*u0 + cp[2]*v0
    measured.append(h_plane - h_quad)
    predicted.append((cq[3] + cq[5]) * radius**2 / 4
                     - (cq[3]*u0*u0 + cq[4]*u0*v0 + cq[5]*v0*v0))
    kappa = abs(cq[3] + cq[5])          # |H|, half the trace of the Hessian
    curvature_radius.append(1.0/kappa if kappa > 0 else np.inf)

measured = np.array(measured); predicted = np.array(predicted)
finite = np.isfinite(measured) & np.isfinite(predicted)
measured, predicted = measured[finite], predicted[finite]
correlation = np.corrcoef(measured, predicted)[0, 1]
slope = float(np.polyfit(predicted, measured, 1)[0])
print(f"n = {len(measured)}   spacing {spacing*MM:.2f} mm   r {radius*MM:.2f} mm")
print(f"median |R| from the fit  : {np.median(curvature_radius)*MM:.1f} mm")
print(f"measured  plane - quad   : median {np.median(measured)*MM:+.4f} mm  "
      f"mean {measured.mean()*MM:+.4f} mm")
print(f"predicted (c3+c5) r^2/4  : median {np.median(predicted)*MM:+.4f} mm  "
      f"mean {predicted.mean()*MM:+.4f} mm")
print(f"correlation {correlation:.4f}   slope {slope:.4f}")
R = np.median(curvature_radius)
print(f"cylinder form r^2/(8R)   : {radius**2/(8*R)*MM:.4f} mm  "
      f"({100*radius**2/(8*R)/R:.3f}% of R, so ~{2*100*radius**2/(8*R)/R:.2f}% of area)")

fig, ax = plt.subplots(figsize=(6.4, 6.0))
ax.scatter(predicted*MM, measured*MM, s=4, alpha=0.25, color="#1f4e79")
span = np.percentile(np.abs(np.r_[predicted, measured])*MM, 99)
ax.plot([-span, span], [-span, span], color="#cc0000", lw=1.5,
        label="y = x  (closed form exact)")
ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_aspect("equal")
ax.set_xlabel("predicted  (c₃+c₅)·r²/4 − (c₃u₀²+c₄u₀v₀+c₅v₀²),  mm")
ax.set_ylabel("measured  plane fit − quadratic fit,  mm")
ax.set_title("Plane-MLS bias against the closed form\n"
             f"{len(measured)} neighbourhoods · correlation {correlation:.3f} · "
             f"slope {slope:.3f}", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=130, facecolor="white")
print("wrote", sys.argv[1])
