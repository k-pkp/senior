# VGGT Volume Measurement Pipeline

Measures the real-world volume of an object from a handful of phone photos,
using an ArUco-marked cube of known size as the scale reference.

```
Input images
  → [1] VGGT inference        predictions.npz
  → [2] Point cloud export    points.ply
  → [3] Segment & cut         objects/{box, leg_cut}.ply
  → [4] Surface reconstruct   mesh/*_recon.ply
  → [5] Watertight check      mesh/*.ply
  → [6] Real-world volume     volumes.csv
```

Six stages. Only Stage 1 runs a neural network; everything after is geometry.

---

## Stage 1 — VGGT inference

**File**: `pipeline/stages/inference.py` → `run_inference()`

One forward pass of VGGT-1B over all input images, then the model is freed.
Produces nine arrays:

| output | shape | used |
|---|---|---|
| `world_points` + `world_points_conf` | (S,518,518,3) | **yes** — the only 3D source |
| `images` | (S,3,518,518) | yes — colours |
| `depth` + `depth_conf` | (S,518,518,1) | no |
| `world_points_from_depth` | (S,518,518,3) | no |
| `extrinsic` / `intrinsic` / `pose_enc` | (S,3,4) / (S,3,3) / (S,9) | no |

### Checkpoint licensing

```python
VGGT_USE_COMMERCIAL = True                          # pipeline/config.py
VGGT_COMMERCIAL_REPO = "facebook/VGGT-1B-Commercial"
```

`facebook/VGGT-1B` is **CC BY-NC-SA 4.0 — non-commercial only**. The commercial
checkpoint is gated: accept the terms on its model page, then `hf auth login`.
Without a token the loader falls back **loudly** to the non-commercial weights —
silently shipping NC weights would be worse than failing.

The commercial licence's Acceptable Use Policy forbids unlicensed medical/health
professional practice and inferring health data without consent — directly
relevant to limb measurement — and requires acknowledgement in publications.

### Preprocessing

`--preprocess-mode crop` (default) resizes width to 518 and centre-crops the
height, discarding ~44% of a 9:16 phone photo. `pad` keeps the whole frame at
lower effective resolution. Neither wins outright: pad is better on est_325,
crop on small_leg.

**Input resolution is fixed at 518.** 1022 was tested and is far worse — the
DINOv2 backbone's positional embeddings are learned for a 37×37 patch grid, so
larger inputs are out of distribution. More effective resolution can only come
from making the subject fill more of the frame.

---

## Stage 2 — Point cloud export

**File**: `pipeline/stages/pointcloud.py` → `export_ply()`

1. Adaptive confidence filter — percentile, with a distribution-aware absolute
   floor for clouds where most points sit at minimum confidence
2. Optional background masking (`--mask_black_bg`, `--mask_white_bg`)
3. Statistical outlier removal (k=20, std_ratio=2.5)

**Output**: `points.ply`

`conf_thres=45` is the only threshold that filters floor and object at the same
rate (57.3% vs 57.6%). Higher values reduce noise but starve the subject — at 90
the object is kept at half the floor's rate.

---

## Stage 3 — Segment, cut, close

**File**: `pipeline/stages/clean.py` → `clean_and_extract()`

Clusters **once** on the dense cloud, then ghost-filters each identified cluster
separately. Filtering first would force a second DBSCAN on a ~14× sparser cloud
and let the two results disagree about which object is the reference, with
nothing checking them.

**Phase A — segmentation**
1. SOR, then voxel downsample if >100k points
2. Remove the dominant plane (`core/plane.py:remove_dominant_plane`) — without
   it DBSCAN links every object through the floor into one blob
3. DBSCAN → box / object, identified by cubeness + black-white ratio
4. Marker detection on the **dense** limb (`core/segmentation.py`) — HSV +
   Excess Green, DBSCAN, SVD plane fit. Density matters: the dense limb yields
   ~337 marker points where the filtered one yields ~10
5. Ghost filter each cluster (`pipeline/ghost.py`) — voxel dedup + normal-aware
   rejection. The pipeline's dominant decimation step, ~97% of points

**Phase B — levelling**
RANSAC ground plane → rotation to Z-up, plus an upside-down flip check. The
combined transform `R_total` is applied to the marker planes too; applying only
`R` mis-places the cut by ~99 mm whenever the flip fires.

**Phase C — cuts and closure**
1. Floor cut (below `floor_z + 8 mm`)
2. Centroid-side marker cut (`core/segmentation.py:apply_marker_cut`) — keeps
   points on the same signed-distance side as the limb centroid, so 1-marker,
   2-marker and n-marker cases use one rule in any coordinate frame
3. **Cut-plane cap** (`core/fill.py:cap_points_on_plane`) — fills the exposed
   cross-section with a grid built in the marker plane's own (u,v) basis, so it
   stays coplanar with the cut however the marker is tilted
4. **Extend to floor** (`core/fill.py:extend_point_cloud_to_floor`) — VGGT does
   not reconstruct the shadowed base where an object meets the ground, so each
   cluster floats 1.5–1.9 cm above the floor. The bottom band is swept down to
   the detected plane *before* capping; capping at the raw `z_min` seals the
   object above the ground and loses the base permanently
5. Bottom cap (alpha-shape disc)

**Output**: `objects/{box, leg_cut, leg_no_cut, merged}.ply`,
`debug/{leg_cluster.ply, cutting_line.json}`

---

## Stage 4 — Surface reconstruction

**File**: `pipeline/stages/reconstruct.py` → `workers/recons_methods_worker.py`

Default **`alpha_shape` for both objects**.

**alpha_shape** is interpolating — the surface passes through the actual points.
Alpha is swept over `ALPHA_MULTIPLIERS` (8–40 × mean NN distance) and the
**smallest value producing a watertight mesh** is selected: the tightest
enclosing surface. Selecting on "first alpha returning any triangles" instead
picks the smallest alpha and returns a shredded non-manifold shell (−91% volume
on a known can).

**Why not Poisson** — it fits a smooth *approximating* implicit surface whose
smoothness prior rounds flat faces and sharp rims inward, losing real volume
(−8.3% on a 325 ml can where the point cloud itself was within ±3%).
alpha_shape beat Poisson for the object at every fixed reference method and
under all four volume-measurement methods.

**Why not a fitted primitive for the box** — `box_primitive` is still available
but no longer default. It builds a *bounding* prism, so on a ~2 mm-noisy shell
it sits 4.8 mm outside the points and inflates the reference ~11%. alpha_shape
sits at median offset 0.000 from the cloud. One method for both objects is also
easier to defend than a primitive for the reference and a solver for the target.

`ball_pivot` was removed — it produced 5.5×10⁷ cm³ and non-watertight output.

---

## Stage 5 — Watertight check

**File**: `pipeline/stages/watertight.py` → `workers/meshfix_worker.py`

Skipped when the mesh is already closed, which with alpha_shape is always — it
selects for watertightness by construction. So Stage 5 is **insurance, not a
processing step**, and the log shows when repair actually fires (a signal that
Stage 4 struggled).

When it fires: PyMeshFix `fill_holes()`, then Open3D `fill_holes()` for residual
gaps, then cKDTree colour transfer, with three retries for transient C++ crashes.

Verification uses `trimesh.load(process=False)` deliberately — the default merge
welds PyMeshFix's intentional seam duplicates and reports a watertight mesh as
open.

---

## Stage 6 — Real-world volume

**File**: `pipeline/stages/volume.py` → `compute_volumes()`

### Scale from a measured length

```
linear_scale = REFERENCE_REAL_SIZE_CM / mean(reference's two horizontal
               FITTED-FACE edges)      # not a bounding box - see below
k            = linear_scale ** 3
```

Deriving scale as `(real_vol / mesh_vol)^(1/3)` instead uses `mesh_vol^(1/3)` as
the reference edge, which holds only for a perfect cube — and the cube root
compounds any deviation three times. At 2.2% off cubic that under-read the edge
3.1% and inflated every volume ~10%.

Using the edge directly also leaves the reference's own volume **free to
disagree with nominal**. Under the old scheme the box always printed exactly
2744 cm³ because it was the denominator; it now reads ~2500 cm³, and that gap is
a real error bar instead of one forced to zero.

### Volume measurement

1. **watertight** — exact signed volume, no discretisation error
2. warp + flood-fill (GPU) — leaks through open surfaces
3. trimesh voxel (CPU)
4. convex hull — unreliable, ignores the surface entirely

Tiers 2–4 warn loudly. A non-watertight mesh once returned 0.000825 instead of
0.0119 because the flood fill escaped through a hole.

### Voxel cross-check

Voxel occupancy is computed alongside the exact value. Boundary voxels are
counted whole, so voxel sits a few percent **above** exact and converges
downward onto it:

```
res 150   box +2.96%   can +6.30%
res 400   box +1.12%   can +2.36%
```

A voxel result *below* exact, or far above, flags a self-intersecting or
inverted surface. Thin objects are hit ~2× harder than compact ones — which is
why voxel is a verifier here, not the primary measurement.

### Dimensions are OBB, not AABB

An axis-aligned box around a tilted object reports its diagonal. The can reads
5.78 cm across by OBB and 6.09 cm by AABB — a 5.5% error, and volume goes as
diameter².

---

## Output structure

```
output/
  leg_mesh.ply / .stl          the measured object
  box_mesh.ply / .stl          the ArUco reference
  scene_mesh.ply / .stl        both merged, vertex colours in the PLY
  for_debug/
    01_inference/   predictions.npz, target/
    02_pointcloud/  points.ply
    03_clean/       objects/  debug/
    04_recon/       mesh/
    05_watertight/  mesh/
    06_volume/      volumes.csv
```

The three top-level meshes are the deliverables; everything intermediate stays
under `for_debug/`. With `--no-watertight` they are published from the Stage 4
recon instead, so they always exist if a mesh was produced.

---

## Usage

```bash
python run.py -i inputs/est_325 --no-segment-leg    # rigid object, no marker
python run.py -i inputs/small_leg                   # limb with marker band
python run.py -i inputs/est_325 --skip_mesh         # point cloud only
```

### Stage-by-stage runner

`stagerun.py` runs stages individually with inspectable output, caching Stage 1
so parameter sweeps cost nothing:

```bash
python3 stagerun.py 1 -i inputs/est_325 --name est_test
python3 stagerun.py 2-6 --name est_test
python3 stagerun.py 4-6 --name variant --src est_test --obj-recon-method poisson
```

`--src` redirects only the first stage of a range. Each stage writes a
`summary.txt`; Stage 1 also writes `raw/` — every model output as PNG, PLY and
JSON.

### Flags

| flag | default | note |
|---|---|---|
| `-i`, `--image_folder` | `./inputs/baam/` | |
| `--output_dir` | `./output/` | |
| `--conf_thres` | 45.0 | see Stage 2 |
| `--prediction_mode` | `pointmap` | `depth` measured worse on both datasets |
| `--num_objects` | 2 | |
| `--max_frames` | auto | 6 on MPS |
| `--no-fill` | off | skip bottom cap and floor extend |
| `--no-segment-leg` | off | required for objects with no marker band |
| `--no-watertight` | off | publish Stage 4 recon as the final meshes |
| `--recon-method` | `alpha_shape` | also `poisson`, `poisson_omp1`, `box_primitive` |
| `--box-recon-method` / `--obj-recon-method` | — | per-object override |
| `--voxel-res` | 150 | cross-check resolution |
| `--seed` | 42 | |
| `--preprocess-mode` | `crop` | `stagerun.py` only |
| `--input-res` | 518 | `stagerun.py` only; must be ÷14 |

---

## Known limitations

**No second reference — scale cannot be validated.** With one known object the
cube *defines* scale. Its shape can be checked (footprint edges agree to 0.3%;
Z/XY reveals floor truncation) but never its absolute size. Closing this needs a
second object of known dimensions: calibrate on the cube, *predict* the second
object's size, compare to truth. That prediction error is the accuracy figure
this project needs and currently cannot produce.

**`REFERENCE_REAL_SIZE_CM = 14.0` is unverified.** The cube is handmade. A 2 mm
build error is 1.4% linear = 4.3% volume on every result.

**Noise floor ~2 mm.** Floor planarity RMS is 1.9–2.3 mm and the can's radial
shell noise is 2.1 mm — the same magnitude, so this is VGGT's baseline surface
error, not something downstream adds. Since volume goes as r², a 1σ radius error
is ±16% volume. Most tuning below that is inside the noise.

**Corrected 2026-08-12:** that ±16% came from a pre-rework measurement on the
can using a metric that mixed shape error into the noise. Re-measured on the
current pipeline as local surface thickness, the floor is **~1-2% volume**
(0.29 mm shell on a 3.94 cm radius limb). Few-percent effects are measurable.

**Ground truth ambiguity.** 325 ml is the can's *fill* volume; the pipeline
measures *external displacement*, which is larger. Every reported percentage
depends on which is meant.

---

## Layout

```
run.py                    entry point
stagerun.py               stage-by-stage runner with per-stage diagnostics
viewer.py                 PLY/STL viewer
volume.py                 standalone volume CLI
pipeline/
  orchestrator.py         drives stages 1-6, publishes final meshes
  cli.py  config.py
  ghost.py                voxel dedup + normal-aware filter
  detection.py            Grounding DINO + SAM (available, not wired into scale)
  stages/                 inference pointcloud clean reconstruct watertight volume
  core/                   plane cluster segmentation fill filters mesh
  utils/                  seeding runlog
workers/
  recons_methods_worker.py   reconstruction subprocess
  meshfix_worker.py          PyMeshFix subprocess
tools/com_vol.py          standalone mesh-vs-reference volume tool
docs/
  experiments.md          every test, result and verdict
  prompt.md               website design brief
  update.md               historical snapshot (superseded)
work/                     stagerun outputs, one folder per stage's experiments
```

Determinism: `--seed` seeds `random`, NumPy, PyTorch and Open3D. Stage 3 is
reproducible bit-for-bit; PyMeshFix and Open3D `fill_holes` have no RNG.
