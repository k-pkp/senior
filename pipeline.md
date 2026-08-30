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
| limb + bands | GroundingDINO boxes, SAM masks; **every** band box that sits on the selected limb, up to two, duplicates suppressed by IoU and by a minimum separation |
| band colour | per-column max-deviation trace of the cord, **dilated ±3 rows** so it reports the cord's body rather than its darkest pixel |
| window | full frame width, sliding vertically; must hold cube and **every** band with 5% margin |
| gate | accept if that window fits everything, **or** if VGGT's own centre crop would keep what is visible; reject only when neither holds |

### How many bands the capture wears

The count is reported, not just the geometry: `framing.json` carries `bands`,
and a frame's record carries `bands_seen`. A capture wears two bands when at
least `BAND_MIN_FRAME_FRAC` of its frames saw two — the same corroboration rule
the band colour uses, for the same reason. One frame's duplicate detection must
not be able to turn a below-the-band measurement into a between-the-bands one.

Measured: `inputs/champ` reads 2 bands on 8 of 8 frames, `inputs/small_leg` 1 on
6 of 6, `inputs/est_325` (no cord) 0. **It under-counts on a hard capture** —
`inputs/sunshine2` wears an ankle cord and a below-knee cord, and GroundingDINO
returns the upper one on 1 frame of 8, below the bar. Two ways of forcing it out
were tried and rejected on measurement: re-running the detector on the limb
above the first band, and matching the primary band's own traced colour. Both
produce candidates on one-band captures at the same rate as on two-band ones
(`inputs/small_leg` scores a "second band" at 0.36–0.45 with contrasts of
95–214), so neither separates the cases.

That is why `--cut-mode auto` reads Stage 3's gated plane count rather than this
one; the band count is a cross-check that says so when the two disagree.

The boxes are not only used for framing. Stage 3 projects them through Stage 1's
pointmap into 3D and fits a plane to what comes back — see **Band projection**
below — so a band Stage 0 can see is a band the cut can use, whatever colour the
cord is.

The measured band colour goes to Stage 3 in place of the config's fixed
thresholds — **when it is usable**. Stage 3 refuses it when the band and the limb
are closer than `MARKER_MIN_AXIS` in chromaticity, which every Aug 2026 capture
is, so on those the config window does the detecting after all. The claim that
this "lets a marker of any colour work" is argued from the mechanism and is not
demonstrated by any capture in hand.

`dilate=3` is measured in **pixels** and was set on 2160 px-wide photographs
where the cord is 10–15 px across. At 1108 px it reaches into skin: the traced
band colour drifts from ExG +11.0 at `dilate=0` to +1.5 at 3, against a limb at
(129, 107, 103). That is what shortens the axis the refusal above then catches.

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
   ~337 marker points where the filtered one yields ~10.
   Where Stage 0 measured a band colour, a chromaticity contrast rule replaces
   those thresholds — but only if the band and limb are far enough apart to
   discriminate. `MARKER_MIN_AXIS = 0.05`; below it the learned colour is
   **refused** and detection falls back to the window above, which is what
   every Aug 2026 capture does. `MARKER_SCORE_MAX = 1.5` bounds the rule from
   above when it does run: a score of 1.0 *is* the band, so anything well past
   it cannot be the band
5. Ghost filter each cluster (`pipeline/ghost.py`) — voxel dedup + normal-aware
   rejection. The pipeline's dominant decimation step, ~97% of points

### Band projection — a second source for the cut

**File**: `pipeline/core/bands3d.py`

Colour detection needs the cord and the limb to be chromatically separable, and
refuses outright below `MARKER_MIN_AXIS`. Measured on a web job with two cords:
separation 0.0259 against a 0.05 floor, so the learned colour was refused, the
config's khaki window matched 130 points on the whole limb, and the lower cord
survived as an 11-point cluster that `MARKER_MIN_CLUSTER_PTS = 40` then dropped.
Stage 0 had detected that cord on 6 of 7 photographs.

The information was in the wrong space, not missing. Stage 0 records each band's
box in frame pixels; Stage 1's `world_points` is a 3D position for every pixel of
every frame. Mapping one through the other turns the 2D detection into thousands
of 3D points, with no colour question asked:

| step | how |
|---|---|
| pixel map | cropped frames map linearly from Stage 0's window; uncropped ones follow VGGT's own resize-and-centre-crop. Both branches verified: on `small_leg` the uncropped frame's points sit at −0.0045 from the fitted plane against −0.0060…+0.0054 for the cropped ones, with the same scatter |
| sampling | the middle `BOX_CORE_FRAC` of each box, pixels above the frame's `CONF_PCT` confidence percentile |
| grouping | DBSCAN in **3D**, not by box position per frame — a frame that saw only the lower cord makes that cord its first box, and pairing by list position would merge two bands into one plane |
| corroboration | a cluster must appear in at least half the frames that saw a band, and never fewer than two |

It is **additive**. Colour planes are never moved or dropped; a projected plane
within `SAME_BAND_DISTANCE` of one is taken to be the same band and the colour
fit wins, being fitted to the cord itself rather than to a slab of limb centred
on it. So a run can only gain a plane here.

Projected planes are exempt from the `MARKER_MIN_HEIGHT_CUBES` floor. That gate
rejects uncorroborated blobs near the ground — feet, arch shadows, the floor
junction — and a band seen on most of the photographs is the corroboration it
stands in for. The perpendicularity gate still applies to everything: it is
geometry, not corroboration.

Measured: on the two-cord web job, colour finds 1 plane and projection supplies
the second (1,842 points from 6 frames), giving cuts at 18.7% and 72.4% of the
limb's span. On `champ` both projected planes agree with colour planes and the
cut is unchanged, point for point. On `small_leg` the single plane agrees and
the cut is unchanged. It cannot rescue a band Stage 0 never saw: `sunshine2`
still reports one.

---

**Phase B — levelling**
RANSAC ground plane → rotation to Z-up, plus an upside-down flip check. The
combined transform `R_total` is applied to the marker planes too; applying only
`R` mis-places the cut by ~99 mm whenever the flip fires.

**Phase C — cuts and closure**
1. Floor cut (below `floor_z + 8 mm`)
2. **Gate the candidate planes**, then cut. A plane must sit at least
   `MARKER_MIN_HEIGHT_CUBES = 1.0` reference-cube heights above the floor — a
   physical length that transfers between captures, where a fraction of the
   limb's own span does not — and lie within
   `MARKER_MAX_AXIS_ANGLE_DEG = 35°` of perpendicular to the limb's local axis,
   fitted from slice centroids. A cord tied round a limb lies across it; a
   plane fitted to a blob of skin or clothing takes the blob's orientation, and
   this is the only test that sees the difference. Genuine bands measure
   2.4–27° off the axis, false planes 53–89°.
   Gating happens **before** the two-plane cap, not after: ranking candidates
   by point count and trimming first discarded the real band on two captures,
   because a real band is small (194 points) and a false one is not (5 265).
   `--cut-mode` then selects, defaulting to `MARKER_CUT_MODE` (`auto`) —
   `upper` keeps what is below the highest valid plane, `span` keeps what lies
   between the outermost two, and `auto` chooses span exactly when two planes
   survive the gates above. Auto reads the plane count and not Stage 0's band
   count, because the gates here are the measured discriminator and a plane is
   the only thing that can cut; Stage 0's count is printed as a cross-check and
   the run says so when the two disagree. Selecting does not discard: **every** gated plane is published
   as `candidates`, and only the selection is published as `markers`, so the
   review screen can offer both bands of a two-band capture even on a run that
   cut on one. This is a statement about what was physically measured, not
   something to infer from how many bands the detector happened to find, so it
   is a per-run flag rather than a per-capture guess: one capture set can hold
   a one-band subject and an upper-and-lower-band subject side by side. Asking
   for `span` when only one plane survives the gates cuts on that plane alone
   and says so — the result is a below-the-band volume, not a segment.
   The cut itself is `core/segmentation.py:apply_marker_cut`: 0 planes no cut,
   1 keeps below, 2 keep between, each normal flipped along world up first so
   the detected sign cannot change the outcome
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
`debug/{leg_cluster.ply, cutting_line.json, cutting_line_levelled.json, levelling.json}`

---

## Stage 4 — Surface reconstruction

**File**: `pipeline/stages/reconstruct.py` → `pipeline/workers/recons_methods_worker.py`

Default **`poisson` for both objects, with `alpha_shape` as an automatic
per-object fallback.** Changed 2026-08-23; see `experiments.md`, E-psr-adopted.

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

**File**: `pipeline/stages/watertight.py` → `pipeline/workers/meshfix_worker.py`

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

> ### RESOLVED 2026-08-27 — main's version stays
>
> What runs is `linear_scale = (REF³ / V_ref_mesh)^(1/3)` with axis-aligned
> extents, so the reference cube reports exactly `REF³` on every run — an
> identity, not a measurement.
>
> This was previously "pending review", with the parked method below
> recommended as its replacement. Scored against five water-displacement
> volumes, the recommendation loses: **1.7% mean absolute error for what ships,
> against 4.1% for the parked fitted-face method and 7.4% for its OBB form.**
> The parked block was deleted from `volume.py` on 2026-08-31; `git log`
> is now the record. It last appeared in commit 371d3da.
> Full table in `docs/stage06_experiments.md`.
>
> The *epistemic* objection is untouched and still correct — calibrating on the
> reference's own volume means it cannot report an error bar on itself. The
> error bar now comes from held-out objects instead, which is what the five
> displacement measurements are.
>
> Stage 6 also runs a **reference reconstruction check** before scaling: the
> cube fills 0.87–0.89 of its own oriented box on a sound capture, and a
> warning fires below `REFERENCE_FILL_MIN = 0.83`. `inputs/blue shirt` measures
> 0.787. That is the pipeline's only check on whether the *geometry* is right,
> as opposed to the capture, the cut or the topology.
>
> The section below describes the **parked** method, kept for the reasoning.
>
> The CSV columns are `ext_x/ext_y/ext_z/size_*_cm` rather than the parked
> method's `obb_a/obb_b/obb_c/height_cm`. The web viewer reads both and mirrors
> whichever derivation produced the file, so runs display either way.

### Circumference at the cut, printed here and shown live in the review

`core/crosssection.py` slices `leg_open.ply` in a ±4 mm slab at each cutting
plane, fits an ellipse (Halir-Flusser, direct least squares) and reports
Ramanujan II's perimeter, plus the diagnostics that say whether the number
means anything: angular coverage, radial residual, and an independent
median-radius polygon. It is the one limb dimension a tape measure can check
without water.

The review screen computes the **same measurement in the browser**, per plane,
while the plane is being dragged — `web/src/lib/crosssection.ts` is a port of
that file, verified against it on a real slice to every digit printed
(36.2753 cm, a 5.8960, b 5.6494, 147 slab points, polygon 36.5967) and against
a synthetic ellipse of known axes to 1e-8. A round-trip to the service would
describe a plane the user had already moved.

Two differences, both deliberate: the browser slices `leg_no_cut.ply`, the cloud
the review draws, which is the same points except near the floor where the
fabricated base lives — a plane low enough to cut that is flagged rather than
measured — and it reports a reason instead of raising, because a plane in
motion passes through positions where no ellipse exists.

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
python run.py -i inputs/small_leg                   # limb with one band: measure below it
python run.py -i inputs/champ                       # two bands: auto measures between them
python run.py -i inputs/champ --cut-mode upper     # ...unless the ruler measured foot-to-upper-band
python run.py -i inputs/est_325 --skip_mesh         # point cloud only
python run.py -i inputs/small_leg --continue-on-rejected   # ignore the framing gate
./serve.sh                                          # web app + compute service
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
pipeline/tools/viewer.py  PLY/STL viewer
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
service/                  HTTP front end: upload, run, serve artifacts
serve.sh                  starts the web app and the service together
web/                      viewer and review UI (Next.js)
docs/
  updates.md              what changed against main and why it is better
  full_flowchart.md       every stage and sub-process, in one chart
  progress.md             the full progress log: every finding, derivation and
                          measurement, in the order they were established
  experiments.md          every test, result and verdict
  pipeline_flowchart.md   diagrams and per-stage I/O
  repo_review.md          contract and silent-failure sweep, with outcomes
  running_the_web_app.md  how to run the web app and the compute service
  stage06_experiments.md  Stage 6 method history and what blocks accuracy
  web_explaination.md     the web app explained from first principles
  experiments/            figure-backed write-ups: ghost removal, MLS, main vs now
  experiments/FIGURES.md  which figures are reproduced from a verified run
work/                     stagerun outputs, one folder per stage's experiments
```

Determinism: `--seed` seeds `random`, NumPy, PyTorch and Open3D. Stage 3 is
reproducible bit-for-bit; PyMeshFix and Open3D `fill_holes` have no RNG.
