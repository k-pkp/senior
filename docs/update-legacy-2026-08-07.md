> **SUPERSEDED — historical snapshot, 2026-08-07.**
>
> This describes the pipeline *before* the staged rework. Much of it is now
> wrong: stage numbering ran to 7 (evaluation stage since deleted), the box used
> a fitted primitive, Poisson was the default reconstructor, `--ghost-filter`
> existed, and Stage 3 clustered twice on two different clouds.
>
> Kept as a record of the earlier architecture and the reasoning at the time.
> For current behaviour see [`../pipeline.md`](../pipeline.md); for what changed
> and why see [`experiments.md`](experiments.md).

# VGGT Pipeline — Engineering Update

## Summary

Refactored the VGGT 3D reconstruction pipeline (images → point cloud → cleaned objects → watertight meshes → real-world volumes calibrated via ArUco reference cube). Key improvements: ghost reduction via voxel dedup + normal-aware filtering, adaptive median-normalized cluster scoring, independent two-flow clean stage in original VGGT space (no leveling before clustering), centroid-side marker cut (coordinate-system-agnostic), Grounding DINO + SAM detection integration, and per-step debug output.

## Project Structure

```
senior/
├── run.py                 entry point
├── requirements.txt
├── viewer.py              PLY/STL viewer (Open3D + Plotly)
├── volume.py              standalone volume CLI (GPU warp + trimesh)
├── docs/
│   ├── update.md          this document
│   └── pipeline.md        original pipeline documentation
├── inputs/                test input images
├── vggt/                  VGGT-1B model package (Meta, bundled)
├── workers/               subprocess workers (Poisson, meshfix)
├── tools/
│   └── com_vol.py         standalone mesh-vs-reference volume tool
└── pipeline/
    ├── cli.py             argparse
    ├── config.py          constants (ArUco size, model URL)
    ├── orchestrator.py    main() — runs Stages 1-7
    ├── detection.py       Grounding DINO + SAM
    ├── clean_debug.py     standalone two-flow debug (post-process)
    ├── core/
    │   ├── plane.py       GPU-batched RANSAC + leveling
    │   ├── cluster.py     DBSCAN + adaptive scoring + cubeness
    │   ├── fill.py        alpha-shape bottom cap
    │   ├── filters.py     confidence + spatial outlier
    │   ├── mesh.py        merge/verify/cleanup
    │   └── segmentation.py  marker detection + SVD plane + centroid-side cut
    ├── utils/
    │   ├── seeding.py     deterministic RNG seeding
    │   └── runlog.py      CPU/VRAM/RAM background logger
    └── stages/
        ├── inference.py   Stage 1 — VGGT inference
        ├── pointcloud.py  Stage 2 — dual PLY export
        ├── clean.py       Stage 3 — clean + cluster + marker cut
        ├── reconstruct.py Stage 4 — Poisson reconstruction
        ├── watertight.py  Stage 5 — PyMeshFix + Open3D
        └── volume.py      Stage 7 — real-world volume (warp GPU + trimesh)
```

## Output Structure (after full pipeline run)

```
output/
├── points.ply                     Stage 2 — dense cloud (baseline, ~885K pts)
├── points_clean.ply               Stage 2 — ghost-filtered clean (~8.7K pts)
├── predictions.npz                Stage 1 — raw VGGT predictions
├── labels.npz                     Stage 1b — detection seeds (with --use-detection)
│
├── dense/
│   ├── leg_cluster.ply            dense leg cluster (reference for marker detection)
│   └── cutting_line.json          marker planes: [{centroid, normal, npts}]
│
├── clean_output_from_vggt/
│   ├── leg_no_cut.ply             ghost-clean leg, pre marker cut (original colors)
│   ├── leg_cut.ply                ghost-clean leg, post marker cut (original colors)
│   ├── box.ply                    ghost-clean ArUco cube (original colors)
│   └── merged.ply                 leg_cut + box merged for viewing (original colors)
│
├── mesh/                          Stage 4 — Poisson reconstruction
│   ├── leg_cut_recon.ply/stl
│   ├── box_recon.ply/stl
│   └── scene_recon.ply/stl
│
├── segmented/                     Stage 3 — legacy segmentation (skipped in new flow)
└── target/                        demo_gradio compatible layout
    ├── predictions.npz
    └── images/
```

---

## Stage-by-Stage Technical Details

### Stage 1 — VGGT-1B Inference

**Model:** Meta VGGT-1B (Vision Geometry Grounded Transformer).

**Architecture:**
- **Backbone:** DINOv2 ViT-L/14 (1024-dim embeddings, 14×14 patches, 518×518 input)
- **Aggregator:** 24 blocks of alternating attention
  - Frame attention: within a single image, shape `(B*S, P, C)`
  - Global attention: cross-frame, shape `(B, S*P, C)`
  - 2D Rotary Position Embedding (RoPE), frequency = 100
  - 1 camera token + 4 register tokens per frame
- **Heads** (operating on layers [4, 11, 17, 23]):
  - `CameraHead` — 4-iteration AdaIN refinement, outputs `pose_enc` shape `(B, S, 9)`
  - `DPTHead` (depth) — Dense Prediction Transformer, multi-scale fusion, `output_dim=2` (depth + confidence)
  - `DPTHead` (point) — same architecture, `output_dim=4` (xyz + confidence)
  - `TrackHead` — correlation pyramid + iterative refinement tracker

**Pose encoding decomposition:**
```
pose_enc ∈ ℝ⁹ per frame:
  [t_x, t_y, t_z]           translation (3D)
  [q_x, q_y, q_z, q_w]      rotation quaternion (XYZW, scalar-last)
  [fov_h, fov_w]             field of view
```
Decoded to:
```
extrinsic ∈ ℝ^(3×4) = [R | t]   cam-from-world matrix (OpenCV convention)
intrinsic ∈ ℝ^(3×3):    K = [[f_x, 0, c_x],
                              [0, f_y, c_y],
                              [0,   0,   1]]
  where f = (size/2) / tan(fov/2),  c = size/2
```

**Point map activation:**
```
world_points = sign(x) · (exp(|x|) − 1)    inverse log transform
conf = 1 + exp(raw_conf)                    range [1, ∞)
```

**Prediction modes:**
- `pointmap` (default): direct 3D point regression via DPT head → `world_points ∈ ℝ^(S×H×W×3)`
- `depth`: depth map unprojection → `world_points_from_depth = f(depth, extrinsic, intrinsic)`

### Stage 2 — Dual Point Cloud Export

**Dense cloud** (`points.ply`):
```
confidence threshold:  P_c = percentile(conf_flat, args.conf_thres)
  if frac_at_min(conf) > 0.5:  P_c = max(P_c, percentile(conf_tail, 25))
mask = (conf ≥ P_c) ∧ (conf > 1e-5)
points_dense = world_points[mask]
```
Optional: black background mask (brightness > 15), white background mask (brightness < 240).

**Clean cloud** (`points_clean.ply`): dense cloud → voxel dedup → normal-aware filter.

---

### Stage 3 — Ghost Reduction + Clustering

#### 3.1 Voxel Deduplication

```
voxel_size = 2 · mean({NN_distance(p_i) | i ∈ random_sample(5000)})

voxel_idx = ⌊(p_i - origin) / voxel_size⌋           quantization

for each occupied voxel v:
  centroid_v = Σ(p_i · w_i) / Σ(w_i)                 weighted mean
  label_v = argmax_l Σ[lab_i = l] · w_i              majority vote
  color_v = round(Σ(c_i · w_i) / Σ(w_i))             average
```
Where `w_i = conf_i / mean(conf)` for confidence-weighted mode, or `w_i = 1` for naive centroid.

**Performance:** 1.18M points → 7.6K voxels (155:1 compression ratio).

#### 3.2 Normal-Aware Filtering

For each point `p_i` with estimated normal `n_i`:
```
N_k = k-nearest neighbors of p_i
mean_n = normalized(mean({n_j | j ∈ N_k}))
dev_i = 1 - |⟨n_i, mean_n⟩|

reject if dev_i > max_dev (default 0.3)
```

Points whose surface normal deviates > ~45° from the local mean are rejected as ghost artifacts.

#### 3.3 RANSAC Leveling (Phase B)

**GPU-batched deterministic RANSAC** (Phase B, applied to dense cloud):
```
Sample m = 1000 triplets (i₁, i₂, i₃) with deterministic seed.
For each triplet:
  n = cross(p₂ - p₁, p₃ - p₁)
  if |n| < 1e-12: reject (degenerate)
  n = n / |n|
  d = -⟨n, p₁⟩                                      plane: ⟨n, x⟩ + d = 0

Inlier count per hypothesis (batched):
  dist = |P @ N^T + d|                             (N, m) distance matrix
  inlier_mask = dist ≤ distance_threshold           (N, m) boolean
  inlier_counts = sum(inlier_mask, axis=0)          (m,) vector

Select hypothesis with max inlier count.
```

**Rodrigues rotation to Z-axis:**
```
v = cross(normal, [0, 0, 1])
s = |v|,  c = ⟨normal, [0, 0, 1]⟩
if s < 1e-8: return identity

v× = [[0,   -v_z,  v_y],
      [v_z,   0, -v_x],
      [-v_y, v_x,   0]]

R = I + v× + v×² · (1 - c)/s²                      Rodrigues formula
```

**Upside-down check:** After rotation, detect plane Z coordinate. If mean Z of non-plane points < plane Z, rotate 180° around X-axis.

#### 3.4 DBSCAN Clustering

```
eps = mean(NN_distance(pcd)) · 4.0                   adaptive epsilon
labels = cluster_dbscan(pcd, eps=eps, min_points=10)
```

#### 3.5 Adaptive Median-Normalized Cluster Scoring

All clusters ranked by three properties, each normalized by the cluster set's median:

```
For each cluster c_i:
  m_{pts}   = median({npts_j})                      median point count
  m_{dens}  = median({density_j})                    median density (pts/volume)
  d_{max}   = max({max_dim_j})                       maximum spatial extent

  s_i = (npts_i / m_{pts}) · 0.4
      + (density_i / m_{dens}) · 0.3
      + (1.0 - max_dim_i / d_{max}) · 0.3
```

No hardcoded `/1000` or `/1000000` constants. Works for any scene scale (500 pts to 500K pts).

#### 3.6 Cubeness Heuristic (Box vs Leg Identification)

```
cubeness = min(extent_x, extent_y, extent_z) / max(extent_x, extent_y, extent_z)  ∈ [0, 1]

bw_ratio = frac(points | brightness < 0.20) + frac(points | brightness > 0.80)    ∈ [0, 1]

aruco_score = cubeness · 0.3 + bw_ratio · 0.7

if any cluster aruco_score > 0.7:
  that cluster = ArUco marker (reference cube)
  other cluster = target leg
else:
  highest cubeness = reference cube
  other = target leg
```

#### 3.7 Marker Detection (HSV + ExG)

```
RGB ∈ [0, 255] → HSV:
  V = max(R, G, B)
  S = (V - min(R, G, B)) / V          if V > 0 else 0
  H = 0°                              if S = 0
      60° · (G - B)/(V - min)         if V = R
      60° · (B - R)/(V - min) + 120°  if V = G
      60° · (R - G)/(V - min) + 240°  if V = B
  (wrapped to [0, 360))

Marker mask:
  hsv_mask = (S > 15%) ∧ (H > 60°)
  exg_mask = (2·G - R - B > 10)                    Excess Green Index
  marker_mask = hsv_mask ∨ exg_mask
```

#### 3.8 SVD Plane Fitting

For each marker cluster with coordinates `P ∈ ℝ^(n×3)`:
```
C = P - mean(P, axis=0)                             centered coordinates
U, Σ, V_t = SVD(C)                                  V_t ∈ ℝ^(3×3)
normal = V_t[-1]                                    least-variance direction
```

#### 3.9 Centroid-Side Marker Cut

Coordinate-system-agnostic. No Z-up assumption. Works in any reference frame.

```
For each marker plane with centroid c and unit normal n:
  dist_centroid = ⟨leg_centroid - c, n⟩
  for each point p_i:
    dist_i = ⟨p_i - c, n⟩
    keep if sign(dist_i) = sign(dist_centroid)
```

**Correctness:** The leg centroid always lies on the leg-side of each marker plane. Points on the same signed-distance side as the centroid belong to the leg. Handles 1-marker (keep one side), 2-marker (keep between planes), and 3+ marker cases correctly.

---

### Stage 4 — Poisson Surface Reconstruction

Per-object subprocess workers. Four methods:

| Method | Algorithm | Details |
|--------|-----------|---------|
| `poisson` | Open3D Poisson | depths 9→6 descending, density filter (bottom 5%), largest component |
| `ball_pivot` | Ball Pivoting | 4 radius multipliers × 3-ball sequences |
| `alpha_shape` | Alpha Shapes | 5 alpha multipliers (0.5–8.0 × avg_NN) |
| `poisson_omp1` | Poisson + OMP=1 | Deterministic variant |

Preprocessing: downsample to ≤90K pts (`voxel = bbox_extent / 350`), normal estimation (`radius = max(avg_NN·4, 0.005)`), tangent-plane orientation.

---

### Stage 5 — Watertight Repair

Two-stage subprocess worker:

**Stage A: PyMeshFix `fill_holes()`**
- `joincomp=False, remove_smallest_components=False`
- Fills holes via boundary-loop detection, never deletes faces
- Shape-preserving: only closes existing holes

**Stage B: Open3D `fill_holes()` fallback**
- `hole_size=1e9` (effectively unlimited)
- Applied only if PyMeshFix didn't produce watertight mesh

**Color transfer:** `cKDTree` nearest-neighbor vertex colors from source (original recon mesh).

---

### Stage 7 — Real-World Volume Computation

**Scale calibration:**
```
k = V_real_ref / V_mesh_ref = 14.0³ / box_mesh_volume    cm³ per mesh-unit³
s = k^(1/3)                                               cm per mesh-unit

real_vol_i = V_mesh_i · k                                 cm³
real_size_i = extent_i · s                                cm
```

**Volume measurement (4-tier fallback):**

| Tier | Method | When | Precision |
|------|--------|------|-----------|
| 1 | Exact signed volume | `mesh.is_watertight` | Exact |
| 2 | GPU warp BVH + CPU flood-fill | Warp available, non-watertight | Voxel-limited |
| 3 | Trimesh voxel | CPU fallback | Voxel-limited |
| 4 | Convex hull | Last resort | Overestimates |

**Tier 2 details (warp+floodfill):**
```
pitch = max_extent / resolution                     voxel size
threshold = pitch · √3 / 2                          half-diagonal query radius

GPU kernel: per voxel cell center, query mesh BVH via
  wp.mesh_query_point_sign_normal() → marks surface voxels
  wp.copy() → GPU → CPU boolean grid

CPU flood-fill: ndimage.label(~grid) from corner seed
  exterior = largest label
  interior = ~padded & ~exterior
volume = sum(interior + surface) · pitch³
```

**Auto-resolution tuning:** iterates 50→300 in steps of 50 until volume change < 1.5% between steps. Convergence tolerance controls voxelization accuracy floor.

---

### Detection Module (`pipeline/detection.py`)

**Model pipeline:**
```
Grounding DINO Base → bounding boxes for ["leg", "cube with black and white pattern"]
    ↓ (per detection)
SAM ViT-B → segmentation mask from box prompt
    ↓ (center 5% crop of mask)
seeds = [(frame_idx, u, v, label)] pixel coordinates
    ↓ lookup in world_points[frame, v, u]
seed_xyz = 3D points in original VGGT space
```

**Enabled via:** `--use-detection` CLI flag. Models cached at `~/.cache/opencode/models/`.

**Cluster labeling (pending integration):** Seeds vote for cluster labels via majority. Currently falls back to cubeness heuristic.

---

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--image_folder` | `./inputs/baam/` | Input image folder |
| `--output_dir` | `./output/` | Output directory |
| `--conf_thres` | 45.0 | Confidence percentile threshold |
| `--prediction_mode` | `pointmap` | `pointmap` or `depth` |
| `--ghost-filter` | off | Enable ghost reduction (dual cloud + voxel dedup + normal-aware) |
| `--use-detection` | on | Grounding DINO + SAM for object labeling |
| `--skip_mesh` | off | PLY only, skip clean+reconstruct |
| `--num_objects` | 2 | Objects to extract |
| `--max_frames` | auto | Max frames (auto-limit to 6 on MPS) |
| `--no-watertight` | off | Skip watertight repair |
| `--no-fill` | off | Skip bottom cap fill |
| `--no-segment-leg` | off | Disable marker-based leg segmentation |
| `--segment-height-axis` | `z` | Height axis for legacy marker cut (`x`/`y`/`z`) |
| `--recon-method` | `poisson` | Reconstruction method for all objects |
| `--box-recon-method` | — | Override method for box (ArUco) |
| `--obj-recon-method` | — | Override method for object (leg) |
| `--voxel-res` | 150 | Voxel grid resolution for volume |
| `--no-auto-res` | off | Disable auto voxel resolution tuning |
| `--seed` | 42 | Random seed |
| `--log` | on | Append per-run metrics to log.csv |
