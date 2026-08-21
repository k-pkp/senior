# What changed, and why — this tree against `senior-ict/senior` `main`

Reference point: `origin/main` at `bab2bbc`. This tree is `keng-branch` at `38d0392`
plus uncommitted work. Two layers, described separately because they were done at
different times and carry different confidence:

- **Layer 1** — three commits already on `keng-branch`. +3440 / −1076 across 26 files.
- **Layer 2** — uncommitted work from the current session. +666 / −65 across 9 files,
  plus three new modules and one new pipeline stage.

Every number quoted below was measured on `inputs/small_leg` or `inputs/est_325`
unless stated otherwise. Where a change was made and later found to be wrong, that
is recorded too — the reasoning matters more than the diff.

> ## Current state, read this first
>
> **Stage 6 has been reverted to main's version.** `pipeline/stages/volume.py`
> runs main's code; everything this tree had done to it is parked below that code
> as a commented block under a `PARKED` header, for the stage's author to review.
> Sections **1.1**, **1.2** and **2.4** below therefore describe work that is
> *no longer active*. They are kept because the reasoning is what the review needs.
>
> Stages **0–5 are active and unchanged by that revert** — verified by re-running
> them afterwards and byte-comparing the meshes, which came out identical.
>
> One consequence, since fixed: main's Stage 6 writes different CSV columns
> (`ext_x/ext_y/ext_z/size_*_cm`) from the parked one
> (`obb_a/obb_b/obb_c/height_cm`). The viewer now reads both — see
> §3.5 — so runs display either way. What still differs is how scale is
> *derived*, and under main's method the reference cube reports exactly
> 2744.00 cm³ by construction.

---

# Layer 1 — committed

## 1.1 `pipeline/stages/volume.py` — how the scale is derived  ⟨PARKED⟩

**Functions touched:** `compute_volumes`, `_load_mesh_info`, `_measure_volume`

### Before

```python
real_ref_vol = REFERENCE_REAL_SIZE_CM ** 3      # 14³ = 2744 cm³
k            = real_ref_vol / ref["volume"]     # cm³ per mesh-unit³
linear_scale = k ** (1.0 / 3.0)
```

Scale came from the **ratio of volumes**: the reference cube's real volume divided
by its reconstructed mesh volume.

### Why that was wrong

It forces the reference's own error to zero. If the cube reconstructs 5% small,
`k` absorbs exactly that 5%, the cube reports 2744 cm³ on the nose, and the error is
silently transferred onto the limb instead. The pipeline could not report how well
it had reconstructed its own reference, because the answer was always "perfectly".

Worse, the correction it applies is **isotropic** — a single cube-root spread over
all three axes — while the actual error is not. The cube's vertical axis
reconstructs short and its horizontals do not, so a uniform correction is the wrong
shape of fix.

### After

```python
linear_scale = REFERENCE_REAL_SIZE_CM / ref_edge   # cm per unit, from fitted faces
k            = linear_scale ** 3
```

Scale comes from a **length**: the mean of the cube's two horizontal fitted-face
separations. The vertical is deliberately excluded, because it is the axis that
reconstructs short.

### What improved

The reference's residual is now a real, reportable measurement rather than a
definition. It reads −1.86% on the six-frame baseline and −1.54% in the current
configuration. That number is the pipeline's own honesty check, and it did not
exist before.

`compute_volumes` also gained a voxel-occupancy cross-check per mesh (an
independent volume estimate that should sit 1–8% above the exact signed volume),
and a squareness self-check that reports the cube's three edges as shares of their
sum — 33.33% each if perfectly cubic, and scale-free, so it needs no ground truth.

---

## 1.2 `pipeline/core/faces.py` — new file, 158 lines  ⟨PARKED — nothing imports it now⟩

**Functions:** `fit_box_faces`, `reference_edges`, `_triangle_normals_and_areas`

### The problem

The reference edge used to come from an oriented bounding box. An OBB must
*contain* the convex hull of the points it wraps, so any excess is pure fitting
error — and on `inputs/est_325` the OBB exceeded its own hull by 6.8% in volume,
consistent with the box being mis-rotated by roughly 1.3°.

```
face-to-face box       2505.7 cm³     the cube's real size
convex hull of points  2479.2 cm³     −1.1%
oriented bounding box  2647.8 cm³     +5.7%
```

### The fix

A cube does not need a bounding box to be measured. Its face normals *are* its
axes. `fit_box_faces` clusters triangle normals by direction (area-weighted, so
that numerous tiny corner triangles cannot outvote a real face), fits a plane to
each face, and pairs opposite faces.

Separation is measured **through the mesh centroid** rather than by differencing
plane offsets, because opposite faces splay by about 1° and differencing would
measure the gap extrapolated away from the object.

### What improved

Reference volume error **−10.7% → −4.7%** on that dataset. This module is for
box-shaped references only; it is never applied to a limb.

---

## 1.3 `pipeline/stages/clean.py` — floor removal and the cut rule

**Functions touched:** `clean_and_extract`, `clean_and_extract_objects`,
`_segment_leg`, `_count_markers_on_ply`. The file is the largest single change in
Layer 1 (+713 lines changed).

### Floor removal

VGGT reconstructs a flat floor as a **duplicated pair of sheets** — the same
ghosting it produces on the limb. The floor is therefore about two RANSAC
thresholds thick, and removing only the plane's inliers took the middle and left
both skins behind:

```
distance from fitted plane, points surviving a 1x removal
  [-0.010,-0.005)  20,080
  [-0.005,+0.005)     614      <- the band that was actually removed
  [+0.005,+0.010)  21,669
```

Those 41,749 leftover points are a full-size slab of floor. DBSCAN then linked the
limb, the cube and that slab into one cluster, and Stage 3 exported the entire
scene as the limb while labelling a 3,743-point sliver as the object.

`PLANE_REMOVAL_BAND_MULT = 2.0` removes a band two thresholds wide. At 2× the floor
stops reappearing — the next dominant plane is a wall, dot 0.19 with the floor
instead of 1.00 — and the box cluster becomes cube-shaped (cubeness 0.87). Past 2×
the gain flattens while the removal starts eating whatever rests on the floor.

### Marker detection and the cut

The old rule is described in 1.4. Its practical effect here was that the cut plane
was fitted to a shadow, and the pipeline measured a slab of ankle.

---

## 1.4 `pipeline/core/segmentation.py` — marker detection

**Functions touched:** `detect_markers`, `cluster_markers`, `compute_cluster_planes`,
`cut_surface_plane`, `segment_point_cloud`, `rgb_to_hsv`

### Before

```python
hsv_mask = (s > 15) & (h > 60)
exg_mask = (2 * g - r - b) > 10
```

Hardcoded, unparameterised. `hue > 60` accepts hue 60–360 — everything except red,
orange and yellow — with **no brightness floor at all**.

### Why that failed

On `inputs/small_leg` it classified two things as markers:

```
shadow  RGB(8,6,8)      V=3.1%    hue is arbitrary at that brightness
skin    RGB(139,87,89)  hue=358°  which `hue > 60` happily admits
```

The shadow cluster had **3349 supporting points against 197 for the real band**, so
it won the cut.

### After

Thresholds moved to `pipeline/config.py` and made overridable per call:

```python
MARKER_HUE_MIN = 70.0     # degrees; below this is yellow/orange/red
MARKER_HUE_MAX = 180.0    # above this is cyan/blue/magenta
MARKER_SAT_MIN = 25.0     # percent
MARKER_VAL_MIN = 15.0     # percent; below this hue is numerically unstable
MARKER_EXG_MIN = 10       # 2G-R-B
```

`MARKER_VAL_MIN` is the important one: hue is numerically unstable as value
approaches zero, so any dark pixel can land anywhere on the wheel.

A detail worth recording, because it explains the design: the `small_leg` band is
**khaki**, RGB(60,52,30), hue 44° — it fails every green hue test. What finds it is
excess green, +14 against skin at −54. The hue window's job is to *exclude* skin,
not to describe the band. Raising `MARKER_EXG_MIN` to 15 silently lost the only real
marker in the dataset.

`MARKER_MIN_CLUSTER_PTS = 40` replaced a hardcoded 150. A real band is small — 99
points out of 182k on `small_leg`, because only the part facing a camera
reconstructs — and 150 rejected it. The old loose colour rule only cleared 150 by
padding clusters with shadow, which is what produced the 3349-point "marker".

---

## 1.5 `workers/recons_methods_worker.py` — alpha shape selection

**Functions touched:** `method_alpha_shape`, `method_poisson`, `method_ball_pivot`,
`_load_and_prep`, `_post_process`

### Before

A list of absolute alpha values, taking the first that produced *any* mesh. No
topology test — a mesh with tunnels or disconnected blobs was accepted as long as
it was produced.

### After

Alpha is searched as **multiples of mean nearest-neighbour spacing**, so it adapts
to point density instead of assuming a scene scale, and the accepted mesh must be
watertight **and** Euler number 2 — a simple closed surface. A mesh that is closed
but has genus (tunnels) has an invalid volume and is now rejected as such.

`workers/recons_worker.py` (164 lines) was deleted; `recons_methods_worker.py`
supersedes it.

---

## 1.6 Other new modules in Layer 1

| file | purpose |
|---|---|
| `pipeline/detection.py` (200) | object detection seeds for segmentation |
| `pipeline/ghost.py` (99) | ghost-sheet dedup — VGGT's duplicated surfaces |
| `pipeline/mls.py` (109) | moving-least-squares surface projection |
| `pipeline/multiview.py` (94) | multi-view consistency filter |
| `stagerun.py` (593) | per-stage runner, so a stage can be re-run without redoing inference |

`MULTIVIEW_MIN_VIEWS` is set to 0 — disabled, and the reason is recorded in
`config.py`. It was built to remove ghost sheets and does not work: shell thickness
was unchanged at every setting while up to 41% of the cloud was discarded. The
ghost is the same model making the same mistake in every view, so the views agree
with each other and the ghost passes. It is kept as a genuine geometric
self-consistency diagnostic, just not as a ghost filter.

---

# Layer 2 — this session, uncommitted

## 2.1 `pipeline/stages/prep.py` — Stage 0, new stage

**Functions:** `prepare_frames`, `_cube_faces`, `_cube_bbox`, `_face_corners`,
`_leg_mask`, `_band_bbox`, `_square_window`, `_vggt_window`, `_grow`,
`_debug_overlay`

### The problem it exists for

VGGT is handed a 518×518 square. Its default `crop` mode resizes the width to 518
and centre-crops the height, which on a 9:16 phone photo **discards 43.8% of every
frame**:

```
2160x3840 -> width 518, height 924 -> centre-crop to 518
keeps original rows 844..2996
```

That is not a neutral loss. It clipped the reference cube's base on two of six
frames — IMG_4458 by 457 px and IMG_4462 by 182 px — and the cube is the object
that sets the scale for every number the pipeline reports. Nothing downstream can
detect or undo it.

### Locating the cube — `_face_corners`, `_cube_bbox`

The markers are not the cube. A 6.3 cm black square sits inside a 14 cm face, so
sizing a window on marker corners **understates the cube by 120–300 px per side**.

Scaling the marker box outward would be wrong, because perspective does not
preserve distance ratios. But the marker's own four corners are four known points
on the face plane, so they *define* the homography between that plane and the
image:

```python
src = [(-3.15,-3.15), (3.15,-3.15), (3.15,3.15), (-3.15,3.15)]   # marker, cm
H    = getPerspectiveTransform(src, detected_corners)
dst  = [(-7,-7), (7,-7), (7,7), (-7,7)]                          # face, cm
face_corners = perspectiveTransform(dst, H)
```

The face's corners have known coordinates in that plane, so mapping them through
`H` is **exact**, not approximate.

One face still bounds only itself — the cube continues behind it, and a single face
understates by up to 203 px. Two adjacent faces cover most of the silhouette, which
is why two were briefly required. That was dropped: the requirement was never "two
faces", it was "a box containing the cube". The cube box is now the **union of the
marker face quads and an open-vocabulary detection of the box**. Each under-covers
alone — the detector by 22–90 px on every frame — but they under-cover in different
directions, so the union does not.

`solvePnP` from a single marker was tried and rejected: a planar square has a
two-fold pose ambiguity that bites hardest when the face is seen edge-on, which is
exactly the case where only one marker is visible. On IMG_4463 it returned a 1378 px
cube from a 207 px face.

### The window — `_square_window`

Full frame width, square, slid vertically until the cube and the band both fit.

Nothing is lost horizontally. Vertically the loss is unavoidable — the square is as
tall as the frame is wide, which is the largest square that exists in a portrait
frame — but its *position* is chosen rather than fixed at centre. Measured offsets
from where VGGT would have cropped:

```
IMG_4459  +133 px      IMG_4462  +434 px     <- its cube was being cut by 178 px
IMG_4460   −81 px      IMG_4463  −300 px
IMG_4461   −47 px      IMG_4458   cannot fit at any offset
```

Sliding vertically also costs less than a freely-placed window. VGGT's pose
encoding is 9 values — 3 translation, 4 quaternion, 2 field-of-view — and carries
**no principal point**; it emits `cx = cy = image centre` on every frame. A
full-width window keeps the horizontal principal point exactly right and misstates
only the vertical, where a subject-centred square got both wrong.

### The gate

Every submitted frame must pass, and at least `MIN_FRAMES = 6` must pass, or the
pipeline does not start. Failures are reported by position in the submitted order
(`img1 (IMG_4458.jpg)`).

Each failure carries a reason and a severity, from `_reject_reasons()`. The two
questions are kept apart because a person acts on them differently: was the object
**seen**, and did it **survive the window**.

| condition | reason | severity | rejected? |
|---|---|---|---|
| band missing, cube seen | `marker missing, not crucial` | not crucial | no |
| cube missing, band seen | `cube missing, crucial` | crucial | yes |
| both missing | `marker and cube missing, very crucial` | very crucial | yes |
| detected, does not fit | `objects out of window` | crucial / very crucial | yes |

A missing band costs the cut but not the scale, and the cut only needs the band on
some frames, so that frame is reported and kept. A missing or clipped cube costs the
scale of every number the run reports, silently, which is why this stage refuses
rather than warns.

The out-of-window reason deliberately does **not** name which object was clipped.
The window is not something a person can adjust — it is the largest square the photo
allows, placed by this stage — so the remedy is identical either way: step back and
re-take. `cube_ok` / `band_ok` remain in `manifest.json` for debugging, and severity
still tracks the consequence even though the message does not.

**Anything not croppable goes to VGGT untouched**, including rejected frames, which
are now written rather than skipped so they can be inspected. If the window could
not hold the cube and the band, any crop it emitted would cut something that
matters, so VGGT applies its own preprocessing instead. A viewpoint reconstructed
from a worse crop is still a viewpoint; a deleted one is not.
`--continue-on-rejected` controls whether the run proceeds, not whether the frames
exist.

Two bugs fixed here. `"band not found"` was unreachable — `band_ok` is `True`
whenever the band is `None`, so a missing band could never produce it. And
`stagerun.py` re-derived its own copy of the reasons, which had drifted: its summary
still said `"cube not contained"` after the stage had begun distinguishing a cube
that was never seen from one the window cuts. Both now read one helper.

### Output

518×518 directly, `INTER_AREA`. For a square input VGGT's own arithmetic is
`round(518/14)*14 = 518` exactly, so its centre-crop branch never fires and nothing
is discarded twice. Doing the reduction here also means the filter is ours, and it
structurally removes a 0.34% vertical stretch that the 921→924 patch-size rounding
introduces on non-square input.

`for_debug/` carries an annotated copy of every frame — crop window in yellow, cube
in magenta, band in green, with the verdict and reason — including rejected ones,
which is precisely where they are needed.

---

## 2.2 `pipeline/core/vlm_detect.py` — new file

**Functions:** `detect`, `segment`, `trace_band_colour`, `_models`, `release`

Open-vocabulary detection (GroundingDINO `IDEA-Research/grounding-dino-tiny`) and
box-prompted segmentation (`facebook/sam-vit-base`).

### Why a model here and not for the cube

The cube carries printed markers, so its silhouette is recoverable exactly by
geometry — no model involved, and a detector would be strictly worse (it falls
inside the true silhouette on every frame).

The limb and the band were class-based and colour-based rules, and both were
brittle for the same reason: they encoded one particular capture.

- The limb came from a COCO `person` mask, which contains the whole body. Its union
  with the cube demanded a square **98–161% of the frame** — larger than the frame
  itself. A box cannot separate a limb from the body it belongs to.
- The band came from `2G − R − B > 10`, tuned to one khaki band, which fired on a
  **houseplant** instead of the band on two frames.

Naming them instead removes both limitations. Measured: the band is found on **6 of
6 frames at 0.81–0.84 confidence**, including the two where the colour rule failed
outright.

### `trace_band_colour` — reading the colour off the band

This took three attempts and the failures are instructive.

The band is a **thin diagonal cord occupying 3–5% of its own bounding box**. So:

1. A median over the box returns skin. It confidently reported the *limb's* colour.
2. Taking the pixels least like skin returns **shadow** — they came out at excess
   green −14 where the band is known to be +14.
3. What is true of the cord is that it **crosses every column of the box exactly
   once**. Scanning column by column and taking the pixel that departs furthest
   from that column's own median traces the cord itself.

Result: RGB(37,30,9), excess green **+14** — exactly the value `MARKER_EXG_MIN` was
hand-tuned to, but derived from the image.

The same trace records the **limb's** colour from the surrounding columns, which
turns out to matter more than the band's (see 2.3).

---

## 2.3 `pipeline/core/segmentation.py` — marker detection by learned contrast

**New function:** `marker_mask_by_contrast`. **Modified:** `segment_point_cloud`
(now takes `marker_colour`).

### The wrong version, recorded because it failed loudly

The first attempt derived a hue/saturation/value window centred on the measured
band colour. It was catastrophic:

```
config window (hand-tuned)      hue  70-180   sat>25   val>15   ExG>10
derived colour window           hue   15-75   sat>34   val>5.1  ExG>7.0
marker cluster selected         102,988 points        (a real band is ~210)
```

Two errors. `val > 5.1%` ignored the config's own warning that 15% is the floor
that matters. And centring the hue window **on** the band put it at 15–75° while
skin sits at 11–20° — so the window admitted the entire limb.

The lesson: the hand-tuned window works because it **excludes skin**, not because
it describes the band. Learning the band's colour and centring a window on it
throws that away.

### The right version

The discriminator is the **contrast between band and limb**, both of which Stage 0
now measures. They are two points in colour space; the line between them is the
direction along which they differ, and everything else is irrelevant:

```python
score = ((chroma(pixel) - chroma(limb)) @ axis) / (axis @ axis)   # axis = band - limb
mask  = score > 0.5        # closer to the band than to the limb
```

Working in **chromaticity** (brightness divided out) rather than raw RGB matters:
this band is darker as well as differently coloured, so the raw axis points partly
along brightness and shadowed skin scores as band — 3,244 points selected. Dividing
intensity out keeps only the chromatic difference, which is exactly why the
hand-tuned rule leaned on excess green rather than value.

The threshold is 0.5 because that is the decision boundary the projection already
defines, not a number tuned until this dataset gave a pleasing count.

### What improved

```
                                   points selected
config window (hand-tuned)              13
derived colour window (the bug)     17,726
chromaticity contrast @ 0.5             12
```

Back in line with the hand-tuned rule, from colours the pipeline measured.

And it generalises, which was the point. On synthetic red and blue bands:

```
                learned            derived thresholds       old rule
khaki (real)    RGB(37,30,9)  +14   hue  14-74, ExG 7        works
red             RGB(203,40,42) −165  hue 329-360, ExG off     FAILS (needs ExG > +10)
blue            RGB(41,60,203) −124  hue 203-263, ExG off     FAILS
```

Excess green is kept only when the marker is actually green-dominant; otherwise it
is meaningless and hue carries the detection.

---

## 2.4 `pipeline/core/markers3d.py` — new file, an independent scale check  ⟨PARKED — nothing imports it now⟩

**Functions:** `detect_marker_quads`, `quad_metrics`, `edge_lengths_by_axis`,
`infer_up_axis`

The reference cube is the pipeline's only ruler, so it cannot check itself: scale
comes from its own fitted faces, so it measures 14.00 cm whatever it really is. The
squareness check has the same blind spot — inflate all three edges equally and the
shares do not move.

The printed markers are a second structure the calibration never touches. ArUco
locates their corners sub-pixel, and those corner pixels index
`predictions.npz["world_points"]` directly, so a marker lifts into 3D with no colour
thresholding, no meshing and no image-to-cloud mapping.

Most of the value needs **no physical constant at all**. A printed marker is flat
and square, so departures are pure reconstruction error:

```
flatness      0.04-0.45 mm    the surface is locally accurate
aspect        1.077-1.079     a square reconstructs ~8% out of square
size spread   3.7-3.9%        one physical square, measured on five faces
```

Reproducibility is what makes it usable: **0.4518 and 0.4507** on two independent
captures with different scenes and scale factors — agreement to 0.24%.

`REFERENCE_MARKER_CM` is deliberately left unset. The sheet was designed at 7.00 cm
but did not print at that size — rectifying the photographs through the marker's own
homography and holding the box at its measured 14.00 cm puts the printed square at
**6.49–6.58 cm**, about 93% of design and consistent with "fit to page" scaling.
Setting 7.00 would bake in a false 10% discrepancy.

---

## 2.5 `workers/recons_methods_worker.py` — two fixes

### The alpha ladder was one rung too short

```python
ALPHA_MULTIPLIERS = [..., 70.0, 90.0]                       # before
ALPHA_MULTIPLIERS = [..., 90.0, 115.0, 140.0, 170.0, 200.0] # after
```

A limb cloud that closes cleanly at 140× reported "no watertight alpha found" and
fell through to a repair path. It was not under-sampled or holed — its Euler number
fell monotonically 2202, 868, 654, 398, 222, 102, 16, 2 with **no boundary edges at
any step**, which is a surface converging, not failing.

Order still matters more than the ceiling: since the smallest sound alpha wins,
adding coarse rungs cannot make a cloud that already closes finely choose a coarse
one.

### The fallback was pinned to the worst candidate

```python
if fallback is None or (wt and not isinstance(fallback[0], type(None))):
    if fallback is None:          # <- inner guard: never updates after the first
        fallback = (m, mul)
```

The outer condition could never do anything either — `fallback[0]` is a mesh, so
`isinstance(..., type(None))` is always False. So the fallback was fixed on the
**first** iteration, and the search starts at the smallest alpha, which is the most
shredded candidate of all. A run that failed to close reported the worst mesh it had
built rather than the best. Measured effect when it fired: limb volume 399.13 cm³
with Euler 430, against a 1074 cm³ expectation.

Now ranked: closed beats open; among those, Euler nearest 2; ties to the mesh with
more surface. Same failing run then produced 850.42 cm³.

---

## 2.6 Smaller corrections

| file | change |
|---|---|
| `pipeline/stages/clean.py` | persists `R_total` to `debug/levelling.json` — Stage 1's pointmap is in the unlevelled frame while every export is in the levelled one, and nothing could relate the two |
| `pipeline/stages/volume.py` | three comments attributed the vertical deficit to the floor. Measurement contradicts it: the floor is correct to ~1 mm (marker centroid reads 7.03 ± 0.14 cm against a physical 7.00). The deficit is a lid seen at grazing incidence plus rim rounding |
| `pipeline/orchestrator.py` | `target/images` copied the *submitted* folder, which stopped being what VGGT saw once Stage 0 could rewrite frames. Now copies the actual inference input |
| `pipeline/cli.py`, `stagerun.py` | `--prep-*` flags, `--continue-on-rejected`, `--no-prep-crop`, `--inference` |

---

# Layer 3 — after this document was first written

Four things, in descending order of how much they change the numbers.

## 3.1 `pipeline/core/vlm_detect.py` — the band colour was measured off the wrong pixels

The cut was landing visibly tilted. Root cause was not the plane fit: it was that
`trace_band_colour` reported a colour the reconstruction never contains.

The trace locates the cord by taking, for each column of the band box, the pixel
departing furthest from that column's median. That pixel is the cord's *most
extreme* one — its darkest, most shadowed core. But a 3D point takes the colour of
whatever cord pixel its ray happened to land on, which is a typical one:

| | band RGB | chromatic B |
|---|---|---|
| reported by the trace | (37.5, 30.0, 9.0) | 0.118 |
| what the point cloud actually holds | (69.7, 61.6, 35.7) | 0.214 |

Stage 3's contrast axis was therefore calibrated on one colour and applied to
another. True band points scored a median of **0.298** along an axis thresholded at
**0.50**, so 85% of the band was discarded — 193 points became 40 — and the
surviving sliver measured 0.21 × 0.04 × 0.03 cm. A strip that thin does not
determine a plane normal (its second and third singular values are within 20% of
each other), so SVD returned an essentially arbitrary one.

The fix is one parameter: `dilate=3`, sampling three rows either side of the traced
cord so the colour describes the cord's body.

| | marker points | plane tilt | off perpendicular to the limb |
|---|---|---|---|
| before | 40 | 30.5° | **20.8°** |
| main's hardcoded detector, for reference | 193 | 19.7° | 3.2° |
| after | **302** | **18.4°** | **1.3°** |

Corroboration, three independent ways: the limb's own axis fitted at that height is
18.0°; main's colour-window detector puts the plane at 19.7°; and the volume it
produces agrees with main's detector to **0.33%** (1071.46 against 1075.04 cm³).
The value of the fix is that it keeps the colour *learned*, so a marker of any
colour works, while matching a detector that only worked because its hue window
happened to exclude this particular skin.

Three alternatives were tested and rejected, recorded because each looked plausible:

- **Tuning the threshold.** Not an operating point: 0.40 → 1.3°, 0.35 → 20.4°,
  0.30 → 82°. True band median is 0.298 and limb p99 is 0.300, so the two
  populations overlap and no threshold separates them.
- **SAM on the band box.** Box-prompted SAM returns the dominant object in the box,
  and inside a band box that is the *limb*. Its mask covers ~13% of the box where
  the cord is ~1.5%, giving ExG +5.0 against the cord's +13.5. Result: 816 "band"
  points and **8.6°** off perpendicular — worse than the trace. SAM stays for the
  leg and cube masks, which are objects; a 2–3 px cord is not.
- **Measuring at VGGT's own 518 resolution**, on the theory that it would match the
  cloud better. It does not: 1.8° at `dilate=0` and **87.6°** at `dilate=1`, which
  swallows the whole limb (14,162 points). At 518 the cord is 2–3 px and every
  pixel is already blended with skin, so there is no clean population left to
  sample. At full resolution it is 10–15 px wide, which is what makes dilation work.

Also confirmed while chasing this: point colours are **not** averaged across views.
`pipeline/stages/pointcloud.py:61` assigns each point the colour of its own pixel
in its own view.

## 3.2 The cut is deferred until a person confirms it

Stage 3 always had `apply_cut=False`, which stops after the uncut cloud and
publishes the detected planes. Nothing used it. Now the measuring pass does.

`stagerun.py --no-cut` runs Stage 3 in that mode: it detects the cutting planes and
publishes them, and deliberately **does not write `leg_cut.ply`**, so Stages 4–6
measure only the reference cube. The limb is not reconstructed at all until the cut
is agreed. Cutting on the first pass instead would put a volume on screen derived
from a cut nobody had approved, and then ask the user to confirm it — the
confirmation would be theatre.

`stagerun.py --cut-only` is the second half. It reads back `leg_open.ply` — the
levelled, filtered, floor-cut limb saved in exactly the state the cut operates on —
applies the confirmed planes, caps the cross-section, and closes to the floor. SOR,
RANSAC, DBSCAN, ghost filtering, MLS and levelling are not repeated.

Two details this depends on, both of which were bugs first:

- **`leg_open.ply` had to be added.** The full path cuts the *pre-floor-close* cloud
  and closes whatever the cut leaves open. `leg_no_cut.ply` is the post-close cloud
  and carries a fabricated floor skirt, so cutting *it* would silently produce
  different numbers. `floor_z` is now persisted in `levelling.json` for the same
  reason.
- **Stage 4 and 5 left stale meshes behind.** When Stage 3 stopped writing
  `leg_cut.ply`, the previous run's `leg_cut_recon.ply` survived and Stage 5 globbed
  it — so a deferred cut still reported a cut limb. Stage 4 now prunes meshes with
  no object behind them, and reuses a mesh newer than its source cloud, so the
  reference cube is reconstructed once rather than twice.

Correctness check: applying the *detected* plane through the split path reproduces
the un-split run at `1126.1505024938106 cm³` — identical to the last digit. The
split changed when the cut happens, not what it computes.

Cost, measured: the confirm pass went from 23.5 s to **9.1 s** (Stage 3 8.7 → 1.2 s,
Stage 4 12.2 → 5.2 s with the cube reused). Whole measurement 124 s → 107 s. Stage 1
is ~50 s of that and runs once either way, so this is not primarily a speed change.

## 3.3 `service/` — a compute service, and the web app driving it

New: `service/app.py` and `service/jobs.py`, plus `serve.sh`. Documented in
[`running_the_web_app.md`](running_the_web_app.md).

A job is a `work/<job_id>/` directory — the same layout every manual run uses — so
`stagerun.py` *is* the backend rather than a second implementation of it, and any
job can be inspected or re-run from a terminal. Stages run as subprocesses: the
service learns which stage is running without parsing logs, and a process that exits
guarantees its VRAM is released.

The flow mirrors the pipeline's own shape rather than flattening it. An upload runs
Stage 0 alone and stops at the framing gate; the user sees which photos failed and
why; Stages 1–6 then run with `--no-cut`; the user confirms or moves the plane;
Stages 3–6 re-run with `--cut-only`.

## 3.4 Stage 0 reports *how* a frame passed, and the overlay stops contradicting it

`framing.json` frames gained a `mode` field. "Accepted" was covering two different
outcomes — a frame this stage could crop itself, versus one that only survived
because VGGT's own centre crop happens to keep the reference. The second loses
whatever that crop removes, and a reviewer could not tell them apart.

The overlay was also drawing the *proposed* window while the verdict was reached
against VGGT's. A frame could be labelled "cube not contained" while the drawn box
plainly contained the cube. It now draws the window the verdict was actually reached
on, with the unused proposal in grey.


## 3.5 The viewer reads both Stage 6 schemas, and can no longer be blanked by a bad scale

Reverting Stage 6 changed the CSV's column names, and the web app read the old
ones. The failure was much worse than a missing number.

`linearScale` computed `REFERENCE_CM / ((obb_b + obb_c) / 2)`. With those columns
absent the parser yielded 0 for each, so the expression evaluated to
`14.0 / 0` — **Infinity**. That was handed to `usePly`, which multiplies every
vertex by it, producing a geometry of `NaN`. The result was a blank viewport with
**no error anywhere**: the fetch had succeeded, the PLY had parsed, and the scene
was simply destroyed afterwards. Samples kept working because their CSV predates
the revert, which made it look like a problem with live data transport rather than
with interpretation.

Three changes:

- `parseVolumesCsv` detects which stage wrote the file — `obb_b` present or not —
  and normalises both shapes, keeping an `aabb` flag so the difference is not
  silently lost. Main's vertical axis is `ext_z`, the scene having been levelled.
- `linearScale` mirrors the derivation of whichever stage produced the file. For
  the parked method that is the fitted-face horizontals; for main's it is
  `k = real_vol_cm3 / volume`, `linear_scale = k^(1/3)`. It does **not** use
  main's extents, which are axis-aligned and on a tilted cube measure the
  diagonal: they give 44.0 cm/unit against a true 59.8, a 26% error.
- It returns `null` rather than a non-finite number, and the viewport renders a
  message instead of nothing.

Mirroring matters more than picking the better method here. If the viewer scaled
geometry differently from the stage that computed the volumes, an object would
appear at a size contradicting its own printed measurement, with no way to tell
which was wrong.

| CSV | derivation | cm/unit |
|---|---|---|
| parked Stage 6 (sample) | fitted-face horizontals | 59.79 |
| main's Stage 6 (live job) | `k^(1/3)` volume ratio | 60.86 |
| main's Stage 6, via its extents | *rejected* — AABB diagonal | 44.01 |

Verified on a real `awaiting-cut` job: 296 marker points, 14,257 kept, cut at
21.45 cm, no error box — and the sample still renders unchanged.

Related, from the same investigation: `usePly` had always captured its load errors
at line 99 and nothing rendered them, so a 404 on a mesh also produced a silent
blank. `CutReview` and `MeshView` draw inside the Canvas and cannot show DOM, so
they now pass the failure up through `onLoadError` to the `Viewport`, which also
detects a missing WebGL context — the state an editor preview pane produces, and
which looks identical to every other blank.


---

# Measured effect, end to end

```
                              main      keng-branch     this session
limb alpha closure              —             —          140x -> 25x
marker colour             hard-coded     hard-coded      measured per capture
cut squareness to limb          —             —          20.8 deg -> 1.3 deg off
capture validation            none          none         gate, per-frame reasons
```

The scale rows that used to head this table have been removed, because Stage 6 is
back on main's derivation and they no longer describe what runs. For the record,
they read: reference volume error −10.7% (main) → −4.7% → −1.54%, and an independent
marker cross-check reproducible to 0.24% across two captures. Recovering those means
adopting some part of the parked block; the case for it is §1.1, §1.2 and §2.4.

With main's Stage 6 active, the reference cube reports **exactly 2744.00 cm³** on
every run, because scale is derived as `(real_vol / mesh_vol)^(1/3)` from that same
cube. That is not an accuracy result — it is an identity. The stage currently emits
no error signal about itself.

Cold full run, `inputs/small_leg`, after the revert: 91 s wall, 5 of 6 frames
accepted, 302 marker points, plane 18.4°, alpha 55× watertight with euler 2, limb
1091.79 cm³.

---

# What did NOT change

**Stages 0–5, across the Stage 6 revert.** Verified rather than assumed: Stages 3–5
were re-run after the revert and their meshes byte-compared against copies taken
before it — `leg_mesh.ply`, `box_mesh.ply`, `scene_mesh.ply` and `leg_no_cut.ply`
all identical, same cut, same 302 marker points. A cold run from an empty `work/`
reproduces the same meshes again.

**The volume arithmetic.** Layer 2 added a cross-check and a diagnostic block to
`compute_volumes` and left the integration alone; limb volumes were bit-identical
across those edits. That whole file is now main's anyway.

---

# Open, and not fixed

1. **Stage 6 is unresolved by design** — reverted to main's version pending review
   by the stage's author. While it is, scale comes from the volume ratio, the
   reference agrees with itself by construction, and dimensions come from an
   axis-aligned box, which reports the 14 cm cube as 18.80 × 19.30 × 14.06 cm
   because an AABB around a tilted cube measures its diagonal.
2. **Two Stage 6 CSV schemas are in circulation.** The viewer handles both
   (§3.5), but they are not equivalent: main's scale comes from a volume ratio
   and its extents are axis-aligned, so displayed centimetres differ by ~2% and
   reported dimensions by much more — the 14 cm cube reads 18.80 × 19.30 × 14.06.
   This resolves when Stage 6 does.
3. **The two entry points disagree on identical input.** `run.py` keeps 12,375
   points and reports 1083.54 cm³; `stagerun.py` keeps 11,864 and reports
   1091.79 cm³ — 4% and 0.8%. Pre-existing, unrelated to the revert, and it bounds
   how precisely any Stage 6 number can be quoted. Most likely a Stage 0 or Stage 3
   argument one caller passes and the other does not. Not chased down.
4. **The marker-colour fix is validated on one dataset.** `small_leg` only —
   `est_325` has no band. The claim that a learned colour generalises to a red or
   blue marker is argued but not demonstrated; a capture with a differently
   coloured band is the test.
5. **No error bar on reported volume.** The horizontal-edge disagreement (~1%) is a
   real scale uncertainty and cubes to ~3% on volume. Nothing propagates it.
6. **No held-out object with independent ground truth.** Every accuracy number in
   this document describes the reference cube measuring itself, or internal
   consistency. Nothing here demonstrates the accuracy of a limb measurement, which
   is what the project exists to do.

Carried over from the parked Stage 6 work, and still true of it if adopted: marker
size spread 6.73% with the full-width window against 3.86% without;
`REFERENCE_MARKER_CM` unset pending a ruler measurement of the black square
(predicted 6.5 ± 0.1 cm); limb voxel cross-check at +9.34%, outside the +1–8%
expectation.


---

# Appendix — the mathematics, stage by stage

Merged in from `update.md` (written 2026-08-12), which answered the same question
as this document and is now folded into it rather than kept alongside. It goes
deeper than the sections above on the **committed** work — every constant with the
measurement that produced it — and it is the reference to read when you need the
derivation rather than the summary.

Two things to hold in mind while reading it:

- It **predates Layers 2 and 3**. Stage 0, the compute service, the deferred cut and
  the band-colour fix are not in it; those are covered above.
- Its **Stage 6 section describes the parked method**, not what runs. Stage 6 has
  since been reverted to main's version — see the STATUS banner at the top of this
  document and `stage06_experiments.md`.

### 0. Change inventory

```
 README.md                        | 206 +++++-----      rewritten
 pipeline.md                      | 576 ++++++-------    rewritten
 pipeline/cli.py                  |  15 +-              new flags
 pipeline/config.py               | 165 ++++++-         13 new constants
 pipeline/core/cluster.py         |  26 +-              adaptive scoring
 pipeline/core/fill.py            | 167 ++++++-         floor extend, plane cap
 pipeline/core/plane.py           |  52 ++-             band removal
 pipeline/core/segmentation.py    | 124 ++++++-         marker rule, cut rule
 pipeline/orchestrator.py         | 100 +++++-          staged output layout
 pipeline/stages/clean.py         | 713 +++++++++-----   restructured
 pipeline/stages/inference.py     |  72 +++-            licence, preprocess mode
 pipeline/stages/pointcloud.py    |  99 ++++-           single output
 pipeline/stages/reconstruct.py   |  50 +--             method routing
 pipeline/stages/volume.py        | 172 ++++++--        scale derivation
 pipeline/stages/watertight.py    |  18 +-              honest repair reporting
 workers/recons_methods_worker.py | 341 +++++++++--      alpha selection
 workers/recons_worker.py         | 164 -----           deleted
 18 files, +1991 / -1074
```

New, untracked on `main`:

| path | what |
|---|---|
| `pipeline/ghost.py` | voxel dedup + normal-aware ghost filter |
| `pipeline/mls.py` | moving-least-squares surface projection |
| `pipeline/multiview.py` | multi-view consistency (**disabled** — documented failure) |
| `pipeline/detection.py` | Grounding DINO + SAM seed detection |
| `stagerun.py` | per-stage runner with caching and metrics |
| `web/` | Next.js review/result front end |
| `docs/` | this file, experiments, web brief |

Stage numbering changed: **7 stages → 6**. The old Stage 6 (evaluation) was
deleted; volume moved from Stage 7 to Stage 6.

---

### 1. Stage 1 — VGGT inference

### 1.1 Checkpoint licensing

`main` unconditionally fetched `facebook/VGGT-1B`, which is **CC BY-NC-SA 4.0 —
non-commercial only**. For a project that may be presented or deployed, that is
a licensing defect, not a preference.

Now `_load_weights()` prefers `facebook/VGGT-1B-Commercial`
(`vggt_1B_commercial.pt`) under the `vggt-aup-license`, falling back only on
failure — and the fallback is **loud**, because silently continuing on the
non-commercial checkpoint is exactly the failure mode that matters:

```python
if VGGT_USE_COMMERCIAL:
    try:
        path = hf_hub_download(VGGT_COMMERCIAL_REPO, VGGT_COMMERCIAL_FILE)
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        print("  WARNING: falling back to facebook/VGGT-1B — CC BY-NC-SA 4.0, "
              "NOT licensed for commercial use.")
```

The commercial repo is **gated**: it needs the licence accepted on the model
page plus `HF_TOKEN` in the environment.

The commercial checkpoint's Acceptable Use Policy forbids unlicensed medical or
health-professional practice and inferring health data without consent. That is
directly relevant to limb measurement and is recorded in `config.py`.

### 1.2 Preprocessing mode

VGGT takes a fixed **518 × 518** input. `main` used only `crop`: scale width to
518, centre-crop the height.

On a 9:16 phone photo, cropping to square discards

$$1 - \frac{9/16}{1} \;\approx\; 44\%$$

of every frame's vertical field of view. On a standing limb that routinely
amputates the subject. `run_inference()` now accepts
`preprocess_mode ∈ {crop, pad}` and `input_res`, exposed as `--preprocess-mode`
and `--input-res`.

**Input:** folder of JPG/PNG/HEIC.
**Output:** `predictions.npz` — `world_points (S,H,W,3)`, `world_points_conf
(S,H,W)`, `depth`, `depth_conf`, `images`, `extrinsic`, `intrinsic`.

---

### 2. Stage 2 — point cloud export

`main` wrote two PLYs (raw and pre-filtered). Now it writes **one**,
`points.ply`, because ghost reduction moved into Stage 3 where it runs *per
identified cluster* — the box and the limb have different densities and must not
share a voxel size derived from their union.

### 2.1 Adaptive confidence threshold

Rather than a fixed cutoff, the threshold is a **percentile of the observed
confidence distribution**:

$$\tau = \operatorname{percentile}\big(\mathrm{conf},\; p\big), \qquad
\text{keep } \{i : \mathrm{conf}_i \ge \tau \;\wedge\; \mathrm{conf}_i > 10^{-5}\}$$

with `p = --conf_thres` (default 45). A fixed absolute threshold does not
transfer between scenes because VGGT's confidence scale shifts with texture and
lighting.

### 2.2 Multi-view consistency — built, measured, disabled

`pipeline/multiview.py` implements the standard idea: reproject each 3D point
into every other view and require its depth to be corroborated by at least
`MULTIVIEW_MIN_VIEWS` others within `MULTIVIEW_REL_THRESHOLD` relative error.

**It does not work here, and the reason is worth keeping.** Measured on
`small_leg`, using local shell thickness (std of point distance to a locally
fitted plane) as the ghost metric:

| `min_views` | points | shell std |
|---|---|---|
| 0 | 527,769 | 0.72 mm |
| 2 | 446,218 | 0.73 mm |
| 3 | 311,760 | 0.72 mm |

Discarding **41%** of the cloud changed shell thickness by nothing. The method
assumes each view's errors are independent, so a wrong point fails
corroboration. VGGT's ghost sheet is *the same model making the same mistake in
every view* — the views agree with each other and the ghost passes cleanly.

`MULTIVIEW_MIN_VIEWS = 0`. Kept in the tree as a genuine self-consistency
diagnostic, not as a ghost filter.

---

### 3. Stage 3 — segment, detect, cut, close

This stage absorbed the largest change (+713 lines). It is now three explicit
phases.

### Phase A — cluster in original VGGT space

`main` levelled the scene first, then clustered. Levelling depends on a RANSAC
floor fit, so a bad fit corrupted the clustering that everything downstream
depends on. Clustering now runs **before** levelling, in raw VGGT coordinates,
and is therefore coordinate-system agnostic.

#### 3.1 Dominant-plane removal — the band fix

Objects rest on the floor, so DBSCAN links every object *through the ground* and
returns one blob. The floor must be removed first.

`main` removed the RANSAC **inliers** — a band one threshold thick:

$$\text{remove } \{p : |\,\hat{n}\cdot p + d\,| \le \tau\}, \qquad
\tau = 3 \cdot \operatorname{median}(\text{NN distance})$$

**This is the single most consequential bug fixed.** VGGT ghosts the floor the
same way it ghosts the limb, so the floor is a *sandwich* roughly two thresholds
thick. Removing the middle leaves both skins. Measured on `small_leg`:

```
signed distance from the fitted plane, points surviving a 1x removal
  [-0.010, -0.005)   20,080
  [-0.005, +0.005)      614      <- the band that was removed
  [+0.005, +0.010)   21,669
```

41,749 leftover points form a full-size slab of floor. DBSCAN then welded the
limb, the cube and that slab into one cluster, which Stage 3 exported *as the
limb*. Every symptom downstream followed from this: the "limb" cloud was
182,234 points, deeper (0.86) than it was tall (0.76); the cube appeared to
float 0.45 units above the floor; the reference's two horizontal edges disagreed
by **122%**; no reconstruction method could close the mesh at any α.

The fix removes a **band around the fitted plane**, not the inlier set:

$$\text{remove } \{p : |\,\hat{n}\cdot p + d\,| \le \tau \cdot \beta\},
\qquad \beta = \texttt{PLANE\_REMOVAL\_BAND\_MULT} = 2.0$$

Evidence for β = 2:

| β | removed | next dominant plane |
|---|---|---|
| 1 | 260,450 | **the same floor**, \|n·n_floor\| = 1.000 |
| 2 | 302,565 | a wall, \|n·n_floor\| = 0.194 |
| 3 | 309,767 (+7,202 only) | — |
| 4 | 312,524 (+2,757 only) | — |

At β = 2 the floor stops reappearing and the gain flattens; past that it starts
eating whatever rests on the floor.

The RANSAC itself is GPU-batched and deterministic — all 1000 hypotheses are
scored in one chunked matmul under a seeded `torch.Generator`, so runs are
reproducible.

#### 3.2 Adaptive cluster scoring

`main` scored clusters with magic constants:

```python
score = npts/1000 * 0.5 + min(density, 5e6)/1e6 * 0.3 - max_dim * 0.2
```

Those constants encode an assumed scale. A cloud twice as dense scores twice as
high for no physical reason. Scoring is now **normalised by the data's own
medians**, so it is scale-free:

$$
s_i = 0.4\,\frac{N_i}{\operatorname{med}(N)}
    + 0.3\,\frac{\rho_i}{\operatorname{med}(\rho)}
    + 0.3\,\left(1 - \frac{D_i}{\max_j D_j}\right)
$$

where $N$ = point count, $\rho$ = density, $D$ = max extent. The third term
rewards compactness. Note the sign change: `main` *subtracted* raw `max_dim`
(units-dependent); this adds a normalised compactness score in [0, 1].

Box identification uses **cubeness** $= \min(\text{extent}) / \max(\text{extent})$
plus an ArUco-likeness score. After the band fix, on `small_leg`:

```
cluster #1: 112,468 pts, cubeness=0.7315  ->  OBJ
cluster #2:  60,651 pts, cubeness=0.8577  ->  BOX   extent (0.313, 0.328, 0.365)
```

#### 3.3 Ghost reduction — `pipeline/ghost.py` (new)

Two steps, per cluster.

**Voxel dedup.** Voxel size is derived from the cloud, not fixed:

$$v = f \cdot \operatorname{mean}\big(\text{NN distance}\big), \qquad
f = \texttt{GHOST\_VOXEL\_FACTOR} = 0.65$$

Points are hashed to $\lfloor (p - p_{\min}) / v \rfloor$ and one representative
kept per cell. This is the pipeline's dominant decimation step. Measured
combined output across box and limb:

```
1.5  -> ~13k pts   (original main behaviour — faceted meshes)
0.75 -> ~26k
0.65 -> current
0.5  -> ~59k
0.35 -> ~120k
0    -> disabled
```

**Normal-aware filter.** A point whose normal disagrees with its neighbourhood's
mean normal sits on a different sheet from its neighbours. For point $i$ with
unit normal $n_i$ and $k = 20$ nearest neighbours:

$$\bar{n}_i = \frac{1}{k}\sum_{j \in \mathcal{N}(i)} n_j, \qquad
\hat{\bar{n}}_i = \frac{\bar{n}_i}{\|\bar{n}_i\|}, \qquad
\delta_i = 1 - \big|\,n_i \cdot \hat{\bar{n}}_i\,\big|$$

Reject when $\delta_i > 0.3$. The absolute value makes it orientation-blind, so
a consistently-flipped normal is not punished.

This was rewritten from a Python loop to a fully vectorised form — one batched
KD-tree query for all points, then `einsum` for the dot products:

```python
_, idx = cKDTree(points).query(points, k=k, workers=-1)   # (N, k)
mean_n = normals[idx].mean(axis=1)
dev = 1.0 - np.abs(np.einsum("ij,ij->i", normals, mean_n))
```

**Known limitation:** this cannot remove the ghost sheet. The ghost is
*parallel* to the true surface, so its normals agree with the neighbourhood and
δ stays small. It removes edge noise and stragglers only.

#### 3.4 MLS surface projection — `pipeline/mls.py` (new)

Since neither multi-view consistency nor normal filtering can see a parallel
ghost, the ghost is collapsed rather than deleted. Each point is projected onto
a locally fitted quadratic surface.

For point $p_i$, take neighbours within radius $r = m \cdot \bar{s}$ where
$\bar{s}$ is mean NN spacing and $m = \texttt{MLS\_RADIUS\_MULT} = 4.0$:

1. Centroid $c$; SVD of the centred neighbourhood $Q - c = U\Sigma V^\top$.
   Local frame: $u = v_1$, $w = v_2$, normal $n = v_3$ (least-variance
   direction).
2. Tangent coordinates $a = (q-c)\cdot u$, $b = (q-c)\cdot w$, height
   $h = (q-c)\cdot n$.
3. Least-squares fit of a degree-2 height field:

   $$h \approx c_0 + c_1 a + c_2 b + c_3 a^2 + c_4 ab + c_5 b^2$$

4. Evaluate at the point's own $(a_i, b_i)$ and move it there:

   $$p_i' = c + a_i u + b_i w + h(a_i, b_i)\, n$$

Both sheets collapse onto one surface; where they are equally populated the
result lands between them.

**The radius must exceed the ghost separation** or the two sheets never share a
neighbourhood — at $m = 2.0$ nothing moved at all. This *moves* points, so it is
a genuine change to geometry, not a cleanup, and it costs volume. Measured on
`small_leg` (shell std / convex hull volume):

| `MLS_RADIUS_MULT` | shell std | hull vol | Δ |
|---|---|---|---|
| off | 0.93 mm | 1922 cm³ | — |
| 3.0 | 0.60 mm | 1887 cm³ | −1.8% |
| **4.0** | **0.41 mm** | 1833 cm³ | −4.6% |
| 6.0 | 0.29 mm | 1788 cm³ | −7.0% |

Whether that shrinkage removes noise or real surface is **unresolved** — it
depends on whether the true surface is the inner or the outer sheet, which we
cannot currently determine.

### 3.5 Marker detection — rewritten

The marker band defines where the measurement stops. `main`'s rule was:

```python
hsv_mask = (s > 15) & (h > 60)        # h in degrees, 0..360
exg_mask = (2*g - r - b) > 10
marker   = hsv_mask | exg_mask
```

`h > 60` is **upper-open**: it accepts hue 60–360°, every colour except red,
orange and yellow — 83% of the wheel. There is no brightness floor at all. Hue
is computed as a ratio of channel differences to the channel maximum,

$$H = 60 \cdot \left(\frac{G-B}{\Delta} \bmod 6\right) \text{ etc.}, \qquad
S = \frac{\Delta}{\max}, \qquad V = \max$$

so as $V \to 0$ the hue becomes **numerically arbitrary**. On `small_leg` this
classified:

| what | RGB | ExG | hue | V | old rule |
|---|---|---|---|---|---|
| shadow | (8, 6, 8) | −4 | 300° | **3.1%** | passes |
| shadow | (15, 11, 12) | −5 | 345° | 5.9% | passes |
| skin | (139, 87, 89) | −54 | **358°** | 54.5% | passes |
| **band** | (60, 52, 30) | **+14** | 44° | 23.5% | passes (via ExG) |

The shadow cluster reached 3,349 supporting points against 197 for the real
band, so it won the cut and the pipeline measured a slab of ankle.

The new rule is an actual green test with a brightness floor on **both** paths:

```python
bright   = v > MARKER_VAL_MIN                                    # 15%
hsv_mask = bright & (s > MARKER_SAT_MIN) & (hue_min < h < hue_max)   # 70..180°
exg_mask = bright & (exg > MARKER_EXG_MIN)                       # 10
```

**Critical detail:** the hue window is *not* what finds the marker. The
`small_leg` band is khaki, hue 44°, and fails every green hue test. **ExG** finds
it — $2G - R - B = +14$ against skin at −54. Raising `MARKER_EXG_MIN` to 15
silently lost the only real marker in the dataset. It stays at 10.

Two geometric plausibility gates were added:

- **`MARKER_MIN_CLUSTER_PTS = 40`** (was hardcoded 150). A real band is small —
  99 points out of 182k, because only the camera-facing arc reconstructs. 150
  rejected it; the old loose colour rule only ever cleared 150 by padding
  clusters with shadow.
- **`MARKER_MIN_HEIGHT_FRAC = 0.20`** — reject planes in the bottom 20% of the
  object's height. Feet, arch shadows and the floor junction live there.

  This gate runs in `clean.py` **after** `R_total`, not inside
  `segment_point_cloud`. Detection runs in original VGGT space, where the
  vertical axis is whatever the camera gave — on `small_leg` the limb's long
  axis is **Y**, not Z — so a Z-based height test there measures sideways. Only
  after levelling does "height" mean height.

Result on `leg_conf45`:

| | old | new |
|---|---|---|
| marker points | 1,402 | 249 |
| via HSV rule | 1,298 | **0** |
| planes accepted | 2 (one spurious, 3349 pts) | **1** |

The surviving plane is 99 pts at 44.3% of height, normal
`[−0.206, −0.246, −0.947]` — matching the last known-good fixture (43.3%,
`[−0.203, −0.246, −0.948]`) to three decimals.

**Honest caveat:** ExG separates this khaki band from this skin tone by ~24
units. Real but not large, and it is one subject. A saturated green band would
move the margin to hundreds and let the hue rule earn its place.

### 3.6 Plane fitting

Each marker cluster is spatially separated by **DBSCAN** (ε = 0.03, min_samples
= 10), then a plane is fitted by **SVD**: for centred cluster points $Q - c$,
the unit normal is the **last right singular vector** $v_3$ — the direction of
least variance. This handles tilted and slanted bands correctly, which a
horizontal-slice cut cannot.

### Phase B — levelling

RANSAC the floor, build the rotation taking its normal to $+Z$ via Rodrigues:

$$R = I + [v]_\times + [v]_\times^2\,\frac{1-c}{s^2}, \qquad
v = \hat{n} \times \hat{z},\; s = \|v\|,\; c = \hat{n}\cdot\hat{z}$$

**Bug fixed (`R_total`).** An upside-down scene triggers a 180° flip *after* the
first rotation. `main` applied the flip to the cloud but kept the original `R`
for the marker planes, so the cut plane was rotated into the wrong frame — it
cut 99 mm off the wrong end. The combined rotation is now tracked:

```python
R_total = np.asarray(R, dtype=np.float64)
## ... in the flip branch:
R_total = np.asarray(flip_R, dtype=np.float64) @ R_total
```

`R_total` is applied to the cloud **and** the marker planes, and the levelled
planes are published to `debug/cutting_line_levelled.json` — because
`cutting_line.json` is written pre-levelling while `leg_no_cut.ply` is written
post-levelling, so the two do not share a frame.

### Phase C — cut and close

#### 3.7 The cut rule — rewritten

`main` used a **centroid-side** rule: keep points whose signed distance shares
the sign of the cloud centroid's. One rule for 1, 2 and n markers.

That is fine for a one-shot run but has three defects, all of which bite once a
user can move the plane in the web UI:

1. Dragging a plane past the centroid **inverts the entire selection** in one
   step.
2. Two planes only mean "keep between" when the centroid already sits between
   them. Put both above it and each independently keeps the below-side.
3. The user cannot overrule a centroid that lands on the wrong side of a
   lopsided cloud.

The rule is now stated in terms of **height**, which is a property of the scene
rather than of the cloud's mass distribution:

```
0 markers  ->  no cut
1 marker   ->  keep what is BELOW the plane
2 markers  ->  keep what is BETWEEN the two planes
>2         ->  trimmed to the 2 best-supported (by npts)
```

Each normal is flipped to point along world up first, so the detected normal's
sign cannot change the outcome:

$$\hat{n} \leftarrow \operatorname{sign}(\hat{n}\cdot \hat{u})\,\hat{n},
\qquad d_i = \hat{n}\cdot p_i - \hat{n}\cdot c$$

- 1 plane: keep $d_i \le 0$.
- 2 planes: keep $(d_i^{(1)} \le 0) \oplus (d_i^{(2)} \le 0)$ — **XOR**. A point
  between the planes is below the upper and above the lower, so exactly one test
  is true. No ordering of the planes is needed.

A plane with $|\hat{n}\cdot\hat{u}| < 10^{-3}$ stands vertical, has no above or
below, and is skipped rather than guessed at.

Verified on a synthetic 1001-point cloud: normal sign irrelevant, plane order
irrelevant, third marker trimmed, tilted normals correct.

`MAX_MARKERS = 2`, enforced in `clean.py` before the planes are published, so
the cut, the cross-section caps and the JSON all see the same set.

#### 3.8 Closing order — moved after the cut

The floor extension and bottom cap fabricate a base at floor level. That is
correct **only when the floor really is the bottom of the region being
measured**, which depends on the cut:

| markers | bottom of kept region | who closes it |
|---|---|---|
| 0 | the floor | floor extend + bottom cap |
| 1 | the floor (keep-below) | floor extend + bottom cap |
| 2 | the lower cut face | `cap_points_on_plane` |

`main` filled before cutting, which got the 2-marker case wrong. The fabricated
skirt was usually discarded by the cut and merely wasted — but a marker placed
low on the limb left part of it *inside* the kept segment: invented geometry in
the middle of a measurement, and a lower face that bulged instead of sitting
flat.

`leg_no_cut.ply` is exempt and always floor-closed — it is the review cloud and
what gets measured if the user declines to cut — so it is built from its own
copy while the cut runs on the unfilled cloud.

Verified on the real cloud, both branches:

```
2 markers: case_2_between -> 19,130 pts, both cut lines capped, NO floor  -> 28,951
1 marker:  case_1_below   -> 15,654 pts, cut line capped + floor closed   -> 86,299
```

#### 3.9 `extend_point_cloud_to_floor` (new)

Capping at the raw $z_{\min}$ would seal the object *above* the ground and lose
the shadowed base entirely. So the bottom band (lowest `band_frac = 10%` of
height) is swept downward in discrete levels to the detected floor $z_f$.

Refuses to run when the gap exceeds `max_gap_frac = 35%` of object height —
that means the floor fit is suspect, and extending would fabricate a long
skirt. This guard is what printed
`[box] floor gap 0.4479 > 35% of height 0.1381 — skipping extension` while the
floor bug was still present.

#### 3.10 `cap_points_on_plane` (new)

Fills the exposed cut cross-section. The 2-D basis is derived from the **marker
normal**, not from world axes:

$$u \perp n, \quad w = n \times u$$

so a tilted cut is capped in its own plane. Left open, the surface solver rounds
the cut into a dome instead of the flat face the cut actually produced.

#### 3.11 Bottom cap density

`cap_point_cloud_bottom` tiles a grid inside the bottom cross-section's alpha
hull. `main` sized that grid from `avg_spacing` — the cloud's own point density.

Reasonable at the time, but that is a **measurement, not a constant**. When
Stage 3 stopped downsampling, spacing went from ~0.007 to 0.00173. Grid count
scales as $1/s^2$, so the cap grew ~16× to **100,559 points** — 77% of the whole
cut cloud was fabricated floor, handed to Delaunay as a dense coplanar slab,
which is its degenerate input.

Fixed in two parts:

$$s_{\text{cap}} = \operatorname{clip}(\bar{s}, 0.0005, 0.005)\cdot
\texttt{CAP\_SPACING\_MULT}, \qquad \texttt{CAP\_SPACING\_MULT} = 3.0$$

Safe because the smallest α multiplier is 8× spacing, so a 3× gap is still
bridged.

Then the count is **bounded**, because the multiplier alone is not safe at both
ends — the cube's underside is pressed against the floor and barely
reconstructs, so 3× took its cap from 150 points to **16**:

$$n_{\text{est}} = \frac{A_{\text{hull}}}{s_{\text{cap}}^2}, \qquad
s_{\text{cap}} \leftarrow \sqrt{A_{\text{hull}} / n_{\text{target}}}
\;\text{ if } n_{\text{est}} \notin [200,\ 20000]$$

Result:

```
box:  hull area 0.00150   ->    203 pts   (was 16)
leg:  hull area 0.29934   -> 10,107 pts   (was 100,559)
```

The point-in-polygon test was also vectorised via `shapely.contains_xy`,
replacing one Python-level `hull.contains(Point(pt))` call per grid cell
(>100k calls).

---

### 4. Stage 4 — reconstruction

### 4.1 Default changed: Poisson → alpha shape

Poisson fits a smooth **approximating** implicit surface. That smoothness prior
rounds off flat faces and sharp rims, losing real volume — bad for a cube whose
edge length *is* the measurement.

Alpha shape **interpolates**: the surface passes through the actual points, so
sharp features survive.

### 4.2 Alpha selection on Euler number — not watertightness

This is the most important correctness change in Stage 4.

`main` had no α search. The naive fix is "first α that returns triangles", which
picks the smallest and returns a shredded non-manifold shell (**−91%** on a known
can).

The next-naive fix is "first watertight α". **Watertightness alone is not
sufficient.** A surface riddled with tunnels is still closed, and the
signed-volume integral faithfully subtracts those tunnels — so the mesh looks
like a perfect cube from outside while reporting far too little volume.

The **Euler characteristic** is the decisive test:

$$\chi = V - E + F$$

$\chi = 2$ is a simple closed surface (genus 0). $\chi = 2 - 2g$ for genus $g$,
so each tunnel costs 2. Measured on the reference cube:

| α | watertight | χ | volume |
|---|---|---|---|
| 30× | ✅ | **−1** | 1898 cm³ |
| 40× | ✅ | **2** | 2467 cm³ |

Against a 2744 cm³ nominal, selecting on watertightness alone would have
under-read by 31%. Selection is now **smallest α with watertight AND χ = 2**,
over `ALPHA_MULTIPLIERS = [8, 10, 12, 14, 16, 20, 25, 30, 40, 55, 70, 90]`.

### 4.3 Delaunay reuse — 4.3× speedup

Alpha shape is computed *from* a Delaunay tetrahedralisation. `main`'s structure
rebuilt it for every α. 3-D Delaunay on surface-sampled points is pathological
(a 42× slowdown was measured), so it is now built **once** and reused:

```python
tetra, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
m = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha, tetra, pt_map)
```

### 4.4 Largest-component filter before the Euler check

Stray fragments each contribute their own $\chi = 2$, so a mesh with debris can
score $\chi = 6$ and be rejected despite a perfectly good main body. The
component filter now runs **before** the topology test.

### 4.5 `_post_process` must not undo the selection

`_post_process` calls `remove_non_manifold_edges()`, which deletes every
triangle touching a non-manifold edge — and that can tear a closed surface open.
Since α was chosen *because* the mesh was watertight with χ = 2, letting cleanup
undo that discards the whole point of the search.

The mesh is now snapshotted before the destructive steps and restored if they
open it. Safe: a watertight surface has exactly one component, so the component
filter had nothing to remove anyway.

The guard uses **trimesh's** definition of watertight, not Open3D's. They
disagree — the `small_leg` mesh at α = 25× is watertight to trimesh (χ = 2) and
not to Open3D, which is stricter about vertex-manifoldness. The α search, Stage
5 and Stage 6 all use trimesh, so guarding on Open3D would protect a property
nothing downstream reads.

### 4.6 Other

- `ball_pivot` removed entirely — never produced a usable mesh.
- `MAX_RECON_POINTS = 90000` as a named constant, replacing two magic `150000`
  literals.
- `workers/recons_worker.py` deleted (−164 lines), superseded by
  `recons_methods_worker.py`.

---

### 5. Stage 5 — watertight repair

PyMeshFix repair, with one behavioural change: it now reports **honestly**
when repair was skipped because the mesh was already closed.

```
box: already watertight — repair skipped (5,645 verts, 11,286 faces)
```

That line is a useful signal in the other direction too: if repair *fires*,
Stage 4 struggled, and the result deserves suspicion.

---

### 6. Stage 6 — real-world volume

Renumbered from Stage 7.

### 6.0 The reference's edge comes from FITTED FACES, not a bounding box

**The largest accuracy change in Stage 6.** `pipeline/core/faces.py` (new).

An oriented bounding box has to guess the object's orientation. On the reference
cube it guessed ~1.3 degrees wrong — enough to enclose **6.8% more volume than
the convex hull of the same points**. An OBB must *contain* the hull, so that
excess is pure fitting error:

```
face-to-face box       2505.7 cm3    the cube's real size
convex hull of points  2479.2 cm3    -1.1%
oriented bounding box  2647.8 cm3    +5.7%
```

Both libraries fail: Open3D's `get_oriented_bounding_box` returned **2.1x the
AABB volume** on this cloud, trimesh's +5.7%. Spread over three axes that is
+2.2% per edge, and since `linear_scale = 14.0 / edge`, it under-read every
volume by ~6%.

A cube does not need a bounding box — its face normals *are* its axes. The new
module clusters mesh triangles by normal (area-weighted, so the many tiny corner
triangles cannot outvote a real face), pairs opposite faces, and measures their
separation **through the mesh centroid** — because opposite faces splay by about
a degree, and differencing plane offsets instead measures the gap extrapolated
away from the object (0.265 units against a true 0.252). Verified exact on a
synthetic cube rotated 30/20 degrees.

| | before | after |
|---|---|---|
| cube, leg scene | 2460.1 cm3 (-10.3%) | **2694.2 cm3 (-1.8%)** |
| cube, can scene | 2449.9 cm3 (-10.7%) | **2643.6 cm3 (-3.7%)** |
| horizontal edges agree to | 2.68% / 1.46% | **0.79% / 1.14%** |

Falls back to the OBB with an explicit warning below two face pairs. Reference
only — meaningless for a limb.

A **squareness gate** was added alongside: the three edges as shares of their sum
should be 33.33% each, and a horizontal deviating more than 2 pp now warns. It is
scale-free and needs no ground truth. It is deliberately *not* used as a
correction — normalising to 100% makes common-mode error invisible, which is the
exact defect above.

### 6.1 Scale derivation — the important maths

`main` derived scale from a **volume ratio**:

$$k = \frac{V_{\text{ref,real}}}{V_{\text{ref,mesh}}}, \qquad
\ell = k^{1/3}$$

This is wrong in a subtle and expensive way. It uses $V_{\text{mesh}}^{1/3}$ as
the reference edge length, which only holds for a **perfect cube**, and the cube
root compounds any deviation three times. Measured: at 2.2% off cubic it
under-read the edge by 3.1% and **inflated every volume by ~10%**.

It also **forces the reference's own error to zero** by construction — the
reference always "measures" exactly 2744 cm³, so the system can never report its
own accuracy.

Scale now comes from a measured **length**:

$$\ell = \frac{\texttt{REFERENCE\_REAL\_SIZE\_CM}}{\bar{e}_{\text{horiz}}},
\qquad k = \ell^3, \qquad
V_{\text{real}} = V_{\text{mesh}} \cdot k$$

**Only the two horizontal edges** are averaged. The vertical edge is the axis the
floor truncates — the cube's underside never reconstructs, so including it drags
the estimate small and inflates everything measured against it. Current run:

```
horizontal = 0.2264, 0.2325 units — disagree by 2.68%
vertical   = 0.2192 units (13.37 cm of an expected 14.00), 4.47% short
ref edge   = 0.229427  ->  linear_scale = 14.0 / 0.229427 = 61.02 cm/unit
```

The reference's volume is now **free to disagree** with nominal, and that gap is
a real error bar: **2346 vs 2744 cm³, −14.5%**.

> **A tautology that was found and removed.** An earlier iteration reported all
> three cube dimensions after forcing `side = mean(fx, fy)`. The mean of the
> three reported edges was then *always exactly 14.0000 by construction* — the
> number proved nothing. Only the **spread** between edges and the **volume** are
> real evidence. The current 13.37 × 13.81 × 14.19 cm is unforced.

### 6.2 OBB by orientation, not magnitude

`main` used axis-aligned bounds. An AABB around a yaw-rotated object reports its
**diagonal**, not its size: the can measured 6.09 cm wide by AABB against 5.75 cm
by OBB; the cube 14.95 vs 14.0.

Oriented extents are now used, and — importantly — they are ordered by
**orientation, not magnitude**. Index 0 is the axis most aligned with world up:

$$i_v = \arg\max_i \left| \frac{R_{:,i}}{\|R_{:,i}\|} \cdot \hat{z} \right|$$

Stage 3 levels the scene, so that axis is the one the floor truncates.
Identifying it geometrically beats guessing from which edges happen to agree,
which picks the wrong edge whenever two are short in the same direction.

CSV columns were renamed accordingly: `size_a/b/c_cm` → `height_cm` /
`width_cm` / `depth_cm`.

### 6.3 Volume by the divergence theorem

For a closed, consistently-oriented triangle mesh, the exact signed volume is

$$V = \frac{1}{6}\sum_{\text{triangles}} \big(v_0 \times v_1\big)\cdot v_2$$

No discretisation error. This is used whenever the mesh is watertight.

### 6.4 Independent voxel cross-check

Voxel occupancy is computed **alongside** the exact value. It over-reads by a few
percent because boundary voxels are counted whole, and converges downward onto
exact as resolution rises. Expected band: **+1% to +8%**.

A voxel result *below* exact indicates a self-intersecting or inverted surface.
Current run:

```
box.ply      exact 0.010323  voxel 0.010725  +3.89%
leg_cut.ply  exact 0.003746  voxel 0.004078  +8.84%
```

### 6.5 Honest failure reporting

- Non-watertight → explicit warning that flood fill **leaks** through holes and
  can under-read by an order of magnitude while returning a plausible number.
- Convex hull fallback → labelled `convex_hull (UNRELIABLE)` with a warning that
  it ignores the surface entirely, so a broken mesh scores the same as a good
  one.
- χ ≠ 2 → explicit warning that the volume is not reliable.

---

### 6.6 Other fixes found by auditing

**`clean_merged_scene()` tore closed meshes open.** It called
`remove_non_manifold_edges()` unguarded — the same function already guarded in
Stage 4's `_post_process`. Now snapshots and reverts if the mesh was closed
before and is not after.

**`stagerun.py` fed the scene mesh back into Stage 5.** Its glob was
`*_recon.ply`, which matched `scene_recon.ply` — already the merge of both
objects — so every object was merged twice. Coincident duplicate geometry turned
two closed surfaces into **7,554 fragments, euler 3113**. `orchestrator.py`
always passed `recon_mesh_paths` and excluded it, so `run.py` was never affected;
only the experiment harness was. The broken mesh did reach
`web/public/samples/small_leg/scene_mesh.ply` and has been regenerated.

**`MAX_RECON_POINTS = 90000`** replaces two magic `150000` literals in the worker.

**A CSV column broke the web app.** `face_h` is a list, and pandas wrote it as a
quoted comma-containing field; every consumer splits on commas without honouring
quotes, so the reference volume read as "2". Working values are now dropped
before the CSV is written.

### 7. Configuration — every new constant

All 13 constants live in `config.py`, each with the measurement that set it.

| constant | value | why |
|---|---|---|
| `PLANE_REMOVAL_BAND_MULT` | 2.0 | floor is a ghost sandwich ~2 thresholds thick |
| `MARKER_HUE_MIN` / `MAX` | 70 / 180° | bounded green window; old rule was upper-open |
| `MARKER_SAT_MIN` | 25% | washed-out pixels have no reliable hue |
| `MARKER_VAL_MIN` | 15% | hue is numerically arbitrary below this |
| `MARKER_EXG_MIN` | 10 | the rule that actually finds the khaki band |
| `MARKER_MIN_HEIGHT_FRAC` | 0.20 | reject feet / arch shadow / floor junction |
| `MARKER_MIN_CLUSTER_PTS` | 40 | a real band is 99 pts; 150 rejected it |
| `CAP_SPACING_MULT` | 3.0 | cap is flat, needs no surface resolution |
| `CAP_MIN_PTS` / `MAX_PTS` | 200 / 20000 | multiplier alone unsafe at both ends |
| `MLS_RADIUS_MULT` | 4.0 | must exceed ghost separation; 2.0 moved nothing |
| `GHOST_VOXEL_FACTOR` | 0.65 | dominant decimation step |
| `MULTIVIEW_MIN_VIEWS` | 0 | **disabled** — documented failure |
| `VGGT_USE_COMMERCIAL` | True | licensing |

---

### 8. Tooling added

### `stagerun.py` (new, ~550 lines)

Per-stage runner. Runs any stage or range (`3`, `4-6`), reads the previous
stage's output from a different run via `--src`, caches Stage 1, and prints
per-stage metrics: point counts, extents, colour presence, `border_contact`,
`floor_planarity`, `multiview_stats`.

This is what makes the pipeline debuggable — a full run is minutes, a single
stage is seconds.

### `web/` (new)

Next.js 15 / React 19 / three.js 0.172 front end. Three screens that matter:

- **Samples** — precomputed runs, no server needed.
- **Review** — interactive cut adjustment on the ghost-filtered cloud. Mirrors
  `apply_marker_cut` **exactly** client-side so the split updates while
  dragging. Shows the detected marker as a fixed yellow plane alongside the
  editable green cut disc.
- **Result** — measured volume, oriented dimensions, and the reference check.

Coordinate handling is the subtle part: the pipeline is **Z-up**, three.js is
**Y-up**, and the loader recentres the cloud. Anything positioned in the same
space (cut planes especially) must receive the identical transform, which is why
`transformToScene` stores `geom.userData.sceneOffset` and `pointToScene` /
`dirToScene` exist.

Slider angles are folded into the upper hemisphere before use. A plane is
unchanged by negating its *whole* normal but **not** by flipping z alone — reading
tilt as $\arccos|n_z|$ while keeping the original $x,y$ did exactly that, and the
plane visibly reversed the moment any slider moved. Verified: the round trip now
gives $|n \cdot n_0| = 1.0000$ against 0.7963 before.

### Output layout

`orchestrator.py` now publishes final deliverables to the top of `output/`
(`leg_mesh`, `box_mesh`, `scene_mesh` as `.ply` + `.stl`) with everything else
under `for_debug/<NN_stagename>/`.

---

### 9. End-to-end result

`inputs/small_leg`, 6 photos, stages 3–6:

```
       name     method  euler   volume    obb_a    obb_b    obb_c
    box.ply watertight      2 0.010323 0.219166 0.226351 0.232503
leg_cut.ply watertight      2 0.003746 0.508577 0.123537 0.258523

       name  height_cm  width_cm  depth_cm  real_vol_cm3
    box.ply      13.37     13.81     14.19       2345.55
leg_cut.ply      31.03      7.54     15.78        851.28
```

The 14 cm reference cube measures **13.37 × 13.81 × 14.19 cm** on a scale derived
from its own edges — the *shape* being right is independent evidence, not
circular. Both meshes reach Stage 5 already watertight with χ = 2.

---

### 10. Known open issues

1. **Reference error −14.5%** (2346 vs 2744 cm³). Genuine, unforced, and the
   dominant error bar on every other number. Vertical edge reads 13.37 cm —
   0.63 cm the floor extension does not recover.
2. **No compute backend.** The web Upload path probes `NEXT_PUBLIC_API_URL` and
   degrades honestly; the FastAPI service wrapping `stagerun.py` does not exist,
   so browser uploads cannot run the pipeline. Review sliders are display-only.
3. **`est_325` fixtures are stale** — only `small_leg` was regenerated.
4. **No caliper ground truth.** The can's nominal 325 ml is its *fill* volume;
   the pipeline measures external displacement. Different quantities — no error
   percentage is defensible until the can is physically measured.
5. **Review counts are not comparable to the pipeline's.** The browser cuts
   `leg_no_cut.ply` (floor-closed); the pipeline cuts the raw cloud then closes.
   Both correct, but the screen implies they match.
6. **MLS shrinkage is unresolved** — 4.6% of hull volume, and we cannot yet say
   whether it is noise or real surface.
7. **Marker detection rests on a ~24-unit ExG margin** on one subject. A
   saturated green band would make this robust.
