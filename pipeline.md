# VGGT Volume Measurement Pipeline

Measures the real-world volume of an object from a handful of phone photos,
using an ArUco-marked cube of known size as the scale reference.

```
Input images
  → [0] Framing gate          frame_NN.png (518²), framing.json
        ── refuses a capture it cannot frame ──
  → [1] VGGT inference        predictions.npz
  → [2] Point cloud export    points.ply
  → [3] Segment, detect       leg_open.ply, cutting_line_levelled.json
        ── stops here for the cut to be confirmed ──
  → [4] Surface reconstruct   mesh/*_recon.ply
  → [5] Watertight check      mesh/*.ply
  → [6] Real-world volume     volumes.csv
  → [3] Apply confirmed cut   objects/leg_cut.ply   (--cut-only)
  → [4..6] again, limb only
```

Seven stages, two of which take a human decision. Stages 0 and 1 run neural
networks; everything else is geometry.

Cold run on `inputs/small_leg`: **80 s** end to end (Stage 1 inference 19 s; the
rest is Stage 0's detectors and Stage 4's alpha ladder).

---

## Stage 0 — Framing gate

**File**: `pipeline/stages/prep.py` → `prepare_frames()`

VGGT is handed a 518×518 square, and its own centre crop discards 43.8% of a 9:16
phone photo with no regard for where the reference is — on `inputs/small_leg` that
clipped the cube's base in two of six frames. This stage chooses the crop instead.

| step | how |
|---|---|
| cube bounds | ArUco `DICT_5X5_250` face quads, expanded to whole faces by homography, unioned with a GroundingDINO box |
| limb + band | GroundingDINO boxes, SAM masks; the band box must sit on the selected limb |
| band colour | per-column max-deviation trace of the cord, **dilated ±3 rows** so it reports the cord's body rather than its darkest pixel |
| window | full frame width, sliding vertically; must hold cube and band with 5% margin |
| gate | accept if that window fits everything, **or** if VGGT's own centre crop would keep what is visible; reject only when neither holds |

The measured band colour goes to Stage 3 in place of the config's fixed thresholds,
which is what lets a marker of any colour work.

### Rejection reasons

| condition | verdict | what the frame is told |
|---|---|---|
| everything found and framed | **pass** | — |
| band missing | **warning** | marker missing — the cut must be placed by hand |
| band found but clipped | **warning** | marker out of window — the suggested cut may be off |
| cube found but clipped | **warning** | cube out of window — VGGT will centre-crop instead |
| cube not detected | **reject** | cube missing — the scale cannot be recovered |
| nothing detected | **reject** | nothing detected — no cube and no marker |
| file cannot be decoded | **reject** | file unreadable — cannot be decoded |

**A warning is a usable frame.** The distinction is what a defect *costs*. The
reference cube sets the scale of every number, so if it is not detected at all
there is nothing to recover from — that is a reject. Everything else degrades the
result without making it impossible: a clipped cube falls back to VGGT's own
centre crop, and a missing band only means the cut is placed by a person in the
review step, which it is anyway. Only rejects stop the run.

`--continue-on-rejected` lets a run proceed past a **reject**; warnings never
needed it.

Anything not croppable is written out **raw** for VGGT to handle, rejected frames
included, so a refused capture can still be inspected.
`--continue-on-rejected` decides whether a **rejected** frame stops the run, not
whether the frames are written.

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

Default **`poisson` for both objects, with `alpha_shape` as an automatic
per-object fallback.** Changed 2026-08-23; see the experiment log (not on this branch).

**Why Poisson.** It approximates rather than interpolates, and it tracks the
points about twice as closely: p95 point-to-surface 1.30 mm against alpha's
2.39 mm on `inputs/small_leg`, and 2.19 mm against 21.80 mm on `inputs/short_leg`.
Across depth 8-11 and trim 0.01-0.10 its answer spans **0.32%**, so it is not
sensitive to its own parameters.

**Why it was rejected before, and why that was wrong.** Poisson's meshes reached
Stage 6 closed but topologically invalid, and the cause was Stage 5: it called
`pymeshfix.fill_holes()` and never `repair()`, so self-intersections and
non-manifold edges survived. With `repair()` in place Poisson reaches χ = 2 on
both objects in **38 of 46** sweep configurations rather than 0 of 48.

**What Poisson does not give, and alpha does.** A *guarantee*. Alpha shape sweeps
α over 8-200 × mean NN distance and selects the smallest value that is both
watertight **and** χ = 2 — the tightest surface that is still a single closed
solid. Poisson has no such selection, and on `inputs/short_leg` — an uncut whole
leg including the foot — it closes at χ = −18, about ten handles, reading 22%
below the alpha answer.

**So the fallback is automatic.** After reconstructing an object, Stage 4 runs the
same repair Stage 5 will and checks the Euler characteristic. If the result is not
χ = 2 it rebuilds **that object** with `alpha_shape` and re-checks:

```
leg_cut_recon: poisson gives chi=-18, not a single closed solid
               — rebuilding with alpha_shape, whose ladder guarantees chi=2
leg_cut_recon: fallback gives chi=2 — a valid solid
```

The trigger is **χ ≠ 2, not "not watertight"** — a surface with tunnels is closed
and `is_watertight` returns True for it, so testing watertightness alone would let
exactly this case through. It is per object, so the cube can keep Poisson while
the limb falls back. If the *check itself* errors, no fallback happens and the
reason is printed: silently swapping methods because trimesh hiccupped would be
worse than the problem. Stage 5 independently warns if any final mesh is still
not χ = 2.

**Why not a fitted primitive for the box** — `box_primitive` is still available
but not default. It builds a *bounding* prism, so on a ~2 mm-noisy shell it sits
4.8 mm outside the points and inflates the reference ~11%.

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

> **Reverted to main's version**, pending review by the stage's author. What runs
> is `linear_scale = (2744 / V_ref_mesh)^(1/3)` with axis-aligned extents, so the
> reference cube reports exactly 2744.00 cm³ on every run — an identity, not a
> measurement — and the 14 cm cube reads 19.18 × 19.47 × 14.09 cm.
>
> The section below describes the **parked** method, kept as a commented block at
> the bottom of `volume.py`. See the full write-up (not on this branch).
>
> The CSV columns are `ext_x/ext_y/ext_z/size_*_cm` rather than the parked
> method's `obb_a/obb_b/obb_c/height_cm`. The web viewer reads both and mirrors
> whichever derivation produced the file, so runs display either way.

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
2744 cm³ because it was the denominator; under the parked method it reads
2694.2 cm³ on the leg scene (−1.8%), and that gap is a real error bar instead of
one forced to zero. An earlier figure of ~2500 cm³ quoted here predated the
fitted-face fix, which is what moved it from −10.3% to −1.8%.

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
    01_inference/   predictions.npz, raw/
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
python run.py -i inputs/small_leg --continue-on-rejected   # ignore the framing gate
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
| `--recon-method` | `poisson` | also `alpha_shape`, `poisson_omp1`, `box_primitive`. **Does not affect the reference cube** — use `--box-recon-method` for that |
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
  orchestrator.py         drives stages 0-6, publishes final meshes
  cli.py  config.py
  ghost.py                the whole ghost chain: voxel dedup, normal-aware
                          filter, MLS surface projection
  multiview.py            multi-view ghost filter (written, not wired into Stage 2)
  stages/                 prep inference pointcloud clean reconstruct watertight volume
  core/                   plane cluster segmentation fill filters mesh faces
                          markers3d vlm_detect
  utils/                  seeding runlog
workers/
  recons_methods_worker.py   reconstruction subprocess
  meshfix_worker.py          PyMeshFix subprocess
tools/com_vol.py          standalone mesh-vs-reference volume tool
work/                     stagerun outputs, one folder per stage's experiments
```

Determinism: `--seed` seeds `random`, NumPy, PyTorch and Open3D. Stage 3 is
reproducible bit-for-bit; PyMeshFix and Open3D `fill_holes` have no RNG.
