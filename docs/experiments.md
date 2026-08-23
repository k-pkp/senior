# Experiment Log

Every pipeline option starts as a flag defaulting to current behaviour. A flag
becomes the default only when it wins on ground truth, and the evidence is
recorded here alongside the decision.

## Decision rule

A change is adopted when it improves accuracy against a **known physical
quantity**, and the mechanism is understood. It is rejected — however good the
number looks — when:

- the metric is self-referential (true by construction), or
- the gain comes from an error that cancels another error, or
- it is only validated on an object with no ground truth.

Two changes failed this rule already:

| change | apparent result | why rejected |
|---|---|---|
| forced cube in `box_primitive` | est_325 error 0.0% | tautology — `k = 2744/side³` makes the reference read 14 cm regardless of input |
| `alpha_shape` for the box | can +3.3% (vs −5.9%) | reference under-measured by 11.4%, inflating everything scaled by it |

## Ground truth available

| object | quantity | value | confidence |
|---|---|---|---|
| ArUco cube | edge | 14.0 cm | high — but only enters as an assertion, see below |
| est_325 can | fill volume | 325 ml | **fill ≠ external displacement**; external likely 340–355 cm³, unmeasured |
| est_325 can | height × diameter | — | **NOT YET MEASURED — highest-value open item** |
| legs | — | none | shape plausibility only |

The cube cannot validate scale on its own: Stage 7 defines `k = 2744 / box_vol`,
so the box always reports 14 cm and 2744 cm³. Only the cube's *edge* is a real
measurement (two footprint edges agreeing to 0.23%).

## Known noise floor

> **CORRECTED 2026-08-12.** The ±16% figure below was measured on the *can*,
> on the *pre-rework* pipeline, before MLS existed, using radial spread about a
> fitted cylinder — a metric that includes shape error as well as noise. It was
> then quoted for years as the current floor, including in `stage06_experiments.md`
> and the README, where it was load-bearing for the claim that the reference's
> error "cannot be distinguished from noise". That claim was wrong.
>
> Re-measured on the current pipeline as local surface thickness (std of
> distance to a plane fitted through each point's 30 nearest neighbours):
>
> ```
> audit_leg   shell 0.29 mm on r = 3.94 cm  ->  volume floor ~1.5%
> audit_can   shell 0.15 mm on r = 2.66 cm  ->  volume floor ~1.1%
> ```
>
> The real floor is **~1–2% volume**, not 16%. Errors of a few percent ARE
> measurable and worth chasing. See `stage06_experiments.md`.

Historic figure, kept for provenance — point-shell radial noise on the can: std
**0.214 cm** on a 2.62 cm radius.
Since volume goes as r², that is **±16% volume**. Any change moving the result
by less than that is inside the noise and cannot be judged on one dataset.

```
radius p25 -> 260.2 cm3
radius med -> 286.4 cm3
radius p75 -> 333.1 cm3
```

## Baseline (2026-08-11)

Pipeline state at the start of the staged rework, est_325:

```
alpha_shape (obj) + box_primitive (box) + watertight volume
  can    305.7 cm3   -5.9% vs 325
  cube   cubeness err 2.7%   (unforced, so this is a real error bar)
```

## Results

| date | flag / change | dataset | result | verdict |
|---|---|---|---|---|
| 2026-08-11 | `GHOST_VOXEL_FACTOR` 1.5 → 0.75 | small_leg | 5,005 → 20,413 pts, meshes visibly smoother | adopted for appearance; **accuracy unverified** |
| 2026-08-11 | `alpha_shape` for object | est_325 | −5.9% vs poisson −10.6%; wins under all 4 volume methods | **adopted** |
| 2026-08-11 | `alpha_shape` for box | est_325 | can +3.3% | **rejected** — reference under-measured 11.4% |
| 2026-08-11 | forced cube | est_325 | 0.0% | **rejected** — tautology |
| 2026-08-11 | `ball_pivot` | est_325 / small_leg | 5.5e7 cm³, non-watertight | **reject; remove from METHODS** |
| 2026-08-11 | extend-to-floor | all | cube Z/XY 0.82–0.85 → 0.97–0.98 | **adopted** — validated against independent floor plane |
| 2026-08-11 | `R_total` flip fix | small_leg | cut edge 99 mm → 9 mm from marker band | **adopted** — bug fix, verified against marker colour |

## Stage 1 tests

### T1 — preprocess mode: crop vs pad (est_325, 8 frames, commercial ckpt)

| metric | crop | pad | |
|---|---|---|---|
| median rel depth error | 1.12% | **0.85%** | −24% |
| p90 rel depth error | 35.00% | 34.19% | ~same |
| mean agreeing views | 4.38 | **4.81** | better |
| pts ≥2 views @5% | 98.2% | **99.5%** | better |
| confident geom on border | 2.4% | **0.6%** | −75% |
| pointmap vs depth | 0.95% | 1.04% | slightly worse |

Pad wins every multi-view consistency metric. But on est_325 the crop was
**not** amputating the subject — both objects sit centred and fully inside the
cropped frame — so this gain is from added context, not from rescuing lost
content. On `small_leg` the crop cuts the leg at the knee, so the effect there
should be larger; that is the test that matters.

Scene scale shifts 0.561 → 0.642 under pad, so absolute downstream numbers are
not directly comparable across modes (the ArUco cube renormalises).

Status: **provisional** — self-consistency only. Needs a stage 2–6 accuracy run
before becoming default.

### T2 — preprocess mode: crop vs pad (small_leg, 6 frames)

| metric | crop | pad | |
|---|---|---|---|
| median rel depth error | **2.20%** | 2.37% | pad worse |
| p90 rel depth error | **33.53%** | 41.36% | pad worse |
| mean agreeing views | 2.47 | **2.72** | pad better |
| pts ≥2 views @5% | 80.7% | **82.3%** | pad better |
| border contact | 4.7% | 4.8% | same |

Predicted pad would win decisively here because crop amputates the leg at the
knee. **It did not** — mixed, and worse on both error metrics.

Reason: pad exposes the whole scene (person, chair, shorts, background
shelving), and that clutter is harder to reconstruct, raising median error. And
the content crop removes is *above the marker band*, which the marker cut
discards two stages later — so the amputation is real but harmless. The region
of interest (calf, ankle, foot, cube) is fully inside the cropped frame.

Conclusion: **crop vs pad is dataset-dependent, not a global default.** est_325
favours pad (centred subjects, added context); small_leg favours crop (less
clutter). Tie must be broken by a stage 2–6 accuracy run on est_325.

Note: `small_leg` has a much worse consistency baseline than est_325
(2.20% vs 1.12% median, 2.47 vs 4.38 agreeing views) — a harder scene with
fewer frames (6 vs 8) and a non-rigid subject.

### T3 — input resolution 518 vs 1022 (est_325, 8 frames)

| metric | 518 crop | 518 pad | **1022** |
|---|---|---|---|
| median rel depth error | 1.12% | 0.85% | **7.72%** |
| p90 rel depth error | 35.00% | 34.19% | 50.90% |
| mean agreeing views | 4.38 | 4.81 | **1.91** |
| pts ≥2 views @5% | 98.2% | 99.5% | **55.1%** |
| pointmap vs depth | 0.95% | 1.04% | **20.16%** |
| inference time | 24.7s | 20.7s | 251.0s |

**Rejected — decisively worse.** Ran without OOM but the geometry collapsed:
the two 3D heads disagree by 20% of scene scale, i.e. they produce different
reconstructions entirely.

Cause: VGGT is trained at 518. DINOv2's positional embeddings are learned for a
37×37 patch grid; 1022 gives 73×73, far out of distribution, and the learned
scale priors do not transfer. **518 is a hard ceiling** — raising input
resolution is not available as a lever.

Implication: effective resolution can only be gained by making the subject fill
more of the 518 frame (tight object-centric crop), not by enlarging the frame.
Grounding DINO + SAM already exist in `pipeline/core/vlm_detect.py` to locate a
crop box. **This is what Stage 0 became** — the object-centric crop suggested here
was built, and `pipeline/detection.py`, the earlier module that held these models,
was deleted once Stage 0 superseded it.

## Stage 2 tests

Metric: **floor planarity RMS**. The floor is real ceramic tile, so its residual
to a fitted plane is VGGT's surface-localisation noise measured against physical
truth — not a self-consistency proxy. All runs from the `*_pad` stage 1 inputs.

Caveat: `cm_per_unit` is hardcoded at 14/0.265, but scene scale varies per run.
Values are comparable **within** a dataset, not across.

| run | conf | mode | points.ply | clean.ply | RMS mm | p95 mm |
|---|---|---|---|---|---|---|
| est_conf30 | 30 | pointmap | 1,502,620 | 72,953 | 3.50 | 7.70 |
| est_conf45 | 45 | pointmap | 1,180,626 | 67,208 | 2.30 | 4.71 |
| est_conf60 | 60 | pointmap | 858,638 | 64,818 | 1.83 | 3.32 |
| est_conf75 | 75 | pointmap | 536,648 | 58,660 | **1.57** | 2.83 |
| est_conf45_depth | 45 | depth | 768,506 | 63,891 | 2.74 | 5.88 |
| est_conf45_noghost | 45 | pointmap | 1,148,917 | — | 2.13 | 4.20 |
| leg_conf45 | 45 | pointmap | 541,492 | 59,876 | **2.78** | 5.01 |
| leg_conf75 | 75 | pointmap | 402,487 | 57,349 | 2.90 | 5.18 |
| leg_conf45_depth | 45 | depth | 885,470 | 60,698 | 3.67 | 8.54 |

### T4 — prediction_mode: pointmap vs depth

pointmap wins on both datasets (2.30 vs 2.74 mm; 2.78 vs 3.67 mm).
**Default confirmed correct.** The depth head is genuinely worse for surface
localisation, not merely unused. Depth mode yields more points on small_leg
(885k vs 541k) while being less accurate — point count is not quality.

### T5 — conf_thres sweep

est_325 falls monotonically 3.50 → 1.57 mm across 30 → 75.
small_leg goes 2.78 → 2.90 mm — slightly **worse** at 75.

So the effect is dataset-dependent and cannot be adopted globally. Also, the
object/floor retention bias grows with threshold:

```
conf 45 -> floor 57.3% kept, object 57.6%   (unbiased)
conf 60 -> floor 42.2%,      object 39.3%
conf 75 -> floor 27.0%,      object 21.3%
conf 90 -> floor 11.5%,      object  5.6%   (subject starved)
```

Lower noise is bought with fewer object points, which forces alpha_shape to a
larger alpha and loses volume. **Needs a stage 3–6 accuracy run to pick.**

### T6 — ghost filter path skips SOR

`export_ply` (legacy) applies statistical outlier removal; `export_ply_dual`
does not. Same threshold gives 2.13 mm (no-ghost) vs 2.30 mm (ghost). Not lost
— Stage 3 applies SOR at `clean.py:57`, inside `_load_and_thin` — but the dense
cloud handed to Stage 3
is dirtier than the legacy path's.

### Context

Floor RMS at the default (1.87–2.30 mm) is the same magnitude as the can's
radial shell noise (2.14 mm). The shell thickness is therefore just VGGT's
baseline surface noise, not something introduced downstream.

## Stage 4 tests

### T7 — box method x obj method (est_325, from reworked stage 3)

| box | obj | box_vol | can cm³ | err vs 325 | box faces |
|---|---|---|---|---|---|
| alpha_shape | alpha_shape | 0.016205 | 339.5 | +4.5% | 12,278 |
| alpha_shape | poisson | 0.016205 | 319.2 | −1.8% | 12,278 |
| box_primitive | alpha_shape | 0.017962 | 306.3 | −5.8% | 12 |
| box_primitive | poisson | 0.017962 | 288.0 | −11.4% | 12 |
| poisson | alpha_shape | 0.015792 | 348.4 | +7.2% | 41,954 |
| poisson | poisson | 0.015792 | 327.6 | +0.8% | 41,948 |

**alpha_shape wins for the object** at every fixed box method (306>288,
339>319, 348>327), consistent with pre-rework results and with the
mesh-vs-hull analysis showing Poisson loses volume by rounding.

### T8 — where each box mesh sits relative to the point cloud

Signed distance from the box cloud (11,833 pts) to each candidate mesh:

| box mesh | volume | mean offset | median | % pts inside |
|---|---|---|---|---|
| box_primitive | 0.017962 | **+0.00480** | +0.00453 | 85.1% |
| alpha_shape | 0.016205 | +0.00107 | **+0.00000** | 71.7% |
| poisson | 0.015792 | −0.00007 | +0.00007 | 61.6% |

`box_primitive` builds a **bounding** prism from min/max extents, so it wraps
the outer envelope of a ~2 mm-noisy shell and sits 4.8 mm outside the points.
On a 265 mm cube that is ~3.6% per dimension ≈ 11% volume — matching the 10.8%
gap to alpha_shape. `alpha_shape` sits at median offset 0, i.e. through the
middle of the shell, which is where the true surface is if the noise is
symmetric.

**Correction to an earlier claim.** I previously reported alpha_shape as
"12.6% below true" for the cube. That reference was the min-area-rectangle
footprint of the same cloud — also a min/max measure, so it carries the same
outward bias. The comparison did not support the conclusion drawn from it.

**Adopted: `alpha_shape` for both box and object.** One method for both is also
easier to defend methodologically than a fitted primitive for the reference.

Remaining caveats:
- Corner-cutting on the cube is visible in renders but unquantified. Median
  offset 0 is dominated by flat faces; corners are few points, large volume.
- None of this validates **scale**. `k = 2744/box_vol` makes the cube read
  14 cm whatever mesh is used. See "no second reference" below.

### T9 — ball_pivot removed

Produced 5.5e7 cm³ and non-watertight output on est_325, 36.2 cm³ on
small_leg, degrading as point density rose. Deleted from `METHODS` and CLI.

## Validation gap — no second reference

With one known object, scale cannot be checked: the cube *defines* it. Shape
can be checked (footprint edges agree to 0.3%; Z/XY = 0.969 reveals the floor
truncation) but never absolute size.

Closing this needs a second object of known dimensions: calibrate on the cube,
**predict** the second object's size, compare to measured truth. That
prediction error is the accuracy figure the project actually needs, and the
pipeline currently cannot produce one.

Also unverified: `REFERENCE_REAL_SIZE_CM = 14.0`. The cube is handmade
(cardboard, printed markers, taped edges). A 2 mm build error is 1.4% linear =
4.3% volume applied to every result.

## Ghost sheets — diagnosis and a failed fix

### The defect

A cross-section through the calf shows **two concentric rings ~2 mm apart** — a
duplicate copy of the surface sitting inside the true one. Visible only once the
cloud is dense enough (obvious at `GHOST_VOXEL_FACTOR` 0.35, invisible at 0.75
purely for lack of points).

Neither existing filter can remove it:

- **voxel dedup** merges points sharing a cell; two sheets 2 mm apart usually
  fall in different cells and both survive.
- **normal-aware filter** rejects points whose normal disagrees with their
  neighbourhood. A ghost sheet is *parallel* to the true surface, so its normals
  agree perfectly. It is structurally blind to this case.

Local surface thickness measured by 30-NN plane fit: std **1.2 mm**, p5–p95
spread ~**2.4 mm**, unimodal and centred — so the sheets overlap rather than
being cleanly separated, except in cross-section where the ring structure shows.

### T10 — multi-view consistency filter: REJECTED

Built `pipeline/multiview.py` — reproject every point into the other cameras,
compare geometric depth against the depth head's prediction, keep points
corroborated by >= N views. Wired into Stage 2 as a per-pixel mask.

| min_views | points | shell std | p5–p95 spread |
|---|---|---|---|
| 0 (off) | 527,769 | 0.72 mm | 2.36 mm |
| 2 | 446,218 | 0.73 mm | 2.42 mm |
| 3 | 311,760 | 0.72 mm | 2.40 mm |

**No effect on shell thickness at any setting**, while discarding up to 41% of
the cloud. At `min_views=3` the thinned cloud also destabilised Stage 3
clustering — the "leg" came out 37 cm tall with the wrong cross-section.

Cause: multi-view consistency assumes per-view errors are **independent**, so a
wrong point fails corroboration. The ghost is the same model making the same
mistake in every view — the views corroborate each other and the ghost passes.
Consistent with the Stage 1 finding that the two 3D heads agree with each other
(1.0% of scene scale) more closely than either agrees with the true surface
(~4%). Shared error, not independent error.

Disabled by default (`MULTIVIEW_MIN_VIEWS = 0`). Kept as a genuine
self-consistency diagnostic; it is simply not a ghost filter.

### Still open

Untried: **MLS / local surface projection** — instead of deleting a sheet,
project all points onto a locally fitted surface so the two sheets collapse into
one. Merges rather than filters, and if both sheets are equally supported the
result lands between them, which is plausibly where the true surface is. More
invasive: it moves points rather than removing them.

## Open

- [ ] Measure the can with calipers (height, diameter) — gates every accuracy claim
- [ ] **Water displacement on a held-out object** — the missing ground truth, and
      the tiebreaker for three deferred decisions. When it exists, re-run:
      E-ghost-steps (does `normal_aware_filter` stay? does `ghost_voxel_downsample`?),
      E-dedup-impl (can `ghost_voxel_downsample` become a library call?), and the Stage 6
      scale question in `stage06_experiments.md`. All three are currently measured
      against the pipeline itself, which cannot settle any of them.
- [ ] `mode="pad"` vs `"crop"` — crop currently discards 44% of each photo
- [ ] `conf_thres` sweep — untested, controls shell thickness
- [ ] length-based scale instead of `k^(1/3)` — more correct, ~3% shift
- [ ] multi-view consistency filter (`pipeline/multiview.py`, written but never
      wired into Stage 2 — its own docstring says where it belongs)
- [ ] single-cluster scene exports nothing (`clean.py:175`, the empty-cluster guard
      in `_clean_cluster` returns empty arrays and nothing downstream notices)

---

## E-marker-colour — how the band's colour is sampled · Aug 2026

**Ground truth used:** main's hardcoded colour-window detector, which the project
lead identified as producing a correct cut. Scored by the angle between the fitted
plane's normal and the limb's own axis, fitted independently from slice centroids
either side of the band (17 slices, straightness 0.049). A band wrapped on a limb
is perpendicular to it, so 0° is correct.

**Root cause found.** `trace_band_colour` reported RGB(37.5, 30.0, 9.0) while the
point cloud's band points hold RGB(69.7, 61.6, 35.7). The trace takes each column's
*most extreme* pixel — the cord's darkest, shadowed core — whereas a 3D point takes
whatever cord pixel its ray landed on, a typical one. Stage 3's contrast axis was
therefore calibrated on a colour the data never contains: true band points scored a
median of 0.298 along an axis thresholded at 0.50, so 85% of the band was discarded
(193 points → 40) and the surviving strip measured 0.21 × 0.04 × 0.03 cm — too thin
to determine a normal.


> **Re-measured 2026-08-23, and the "off perpendicular" column does not
> reproduce.** Re-running all three arms on the current tree gives the *tilt*
> column almost exactly — main's detector at 19.73° against the 19.7° recorded
> here, the fix at 19.04° against 18.4° — but the derived "off perpendicular"
> figures come out 10.49°, 24.52° and 9.37°, not 3.2°, 20.8° and 1.3°. That
> quantity is the 3D angle between the plane normal and a fitted limb axis, and
> it is extremely sensitive to how the axis is fitted: the same three planes
> score 12.5° / 22.6° / 12.2° against one reasonable axis estimate and 10.5° /
> 24.5° / 9.4° against another. **The single-number claim should not be quoted.**
>
> What does reproduce, and is what the fix is actually worth:
>
> | | band points | normal's tilt from vertical |
> |---|---|---|
> | main's hardcoded khaki window | 193 | 19.73° |
> | traced, `dilate=0` — the bug | **45** | **27.06°** |
> | traced, `dilate=±3` — ships | **296** | **19.04°** |
>
> The limb's own axis, fitted on `leg_open.ply` over the upper 60% of its span,
> leans **19.01°**. The shipping detector matches that to 0.03°; the bug is 8°
> out and fits its plane through 45 points instead of 296. Figure:
> [`experiments/cut_plane_band_colour.png`](experiments/cut_plane_band_colour.png).

| variant | marker pts | off perpendicular | verdict |
|---|---|---|---|
| argmax trace (previous default) | 40 | 20.8° | **replaced** |
| main's hardcoded window (ground truth) | 193 | 3.2° | reference |
| **trace dilated ±3 rows** | **302** | **1.3°** | **DEFAULT** |
| threshold tuned to 0.40 | 54 | 1.3° | rejected — not an operating point |
| threshold tuned to 0.35 | 86 | 20.4° | rejected |
| threshold tuned to 0.30 | 175 | 82.2° | rejected |
| SAM mask on the band box | 816 | 8.6° | rejected |
| measured at VGGT's 518 resolution, dilate 0 | 546 | 1.8° | rejected |
| measured at 518, dilate 1 | 14,162 | 87.6° | rejected |

**Why threshold tuning fails.** True band median 0.298 against limb p99 0.300 — the
populations overlap, so no threshold separates them, and at lower thresholds a
larger skin cluster outgrows the band and wins cluster selection.

**Why SAM fails.** Box-prompted SAM returns the dominant object in the box, and
inside a band box that is the limb. Its mask covers ~13% of the box where the cord
is ~1.5%, giving ExG +5.0 against the cord's +13.5. SAM is kept for the leg and cube
masks, which are objects; a 2–3 px cord is not.

**Why measuring at 518 fails**, despite that being the resolution VGGT consumes: at
518 the cord is 2–3 px and every pixel is already blended with skin, so no clean
population remains; one step of dilation swallows the whole limb. At full resolution
the cord is 10–15 px wide, which is what makes dilation work. Confirmed separately
that point colours are **not** averaged across views — `pointcloud.py:61` assigns
each point its own pixel from its own view.

**Corroboration.** Three independent lines agree: the limb axis at that height is
18.0°, main's detector puts the plane at 19.7°, the fixed learned detector at 18.4°.
Volumes agree to 0.33% (1071.46 against 1075.04 cm³) — both under the **parked**
Stage 6. What ships today reports 1081.94 cm³ for the same capture, because Stage 6
is reverted to main's volume-ratio scale; the 0.33% is the agreement between two
*detectors*, and holds whichever Stage 6 consumes them.

**Scope.** `small_leg` only. `est_325` has no band, so the claim that a learned
colour generalises to a red or blue marker is argued, not demonstrated.

---

## E-deferred-cut — measure the reference first, cut only once confirmed · Aug 2026

**Change.** Stage 3 gained `--no-cut` (detect the plane, write no `leg_cut.ply`, so
Stages 4-6 measure only the reference cube) and `--cut-only` (read back
`leg_open.ply` and apply confirmed planes, skipping SOR, RANSAC, DBSCAN, ghost
filtering, MLS and levelling).

**Correctness.** Applying the *detected* plane through the split reproduces the
unsplit run at `1126.1505024938106 cm³` — identical to the last digit.

**Cost.** Confirm pass 23.5 s → 9.1 s (Stage 3 8.7 → 1.2 s; Stage 4 12.2 → 5.2 s
with the cube's mesh reused). Whole measurement 124 s → 107 s. Stage 1 is ~50 s of
that and runs once either way, so this is not primarily a speed change — it stops a
volume being produced from a cut nobody approved.

**Two bugs found doing it.** `leg_no_cut.ply` is the post-floor-close cloud and
carries a fabricated skirt, so cutting it is *not* equivalent to the real path —
hence `leg_open.ply` and a persisted `floor_z`. And Stage 4/5 left stale meshes
behind, so a deferred cut still reported a cut limb until Stage 4 learned to prune
meshes with no object behind them.

---

## E-stage0-reasons — rejection taxonomy and what happens to a rejected frame · Aug 2026

**Audit first.** Four reason strings were written; only three could fire.
`"band not found"` was unreachable, because `band_ok` is `True` whenever the band
is `None` — a missing band never rejected a frame, it silently blocked cropping
instead. `stagerun.py` also re-derived its own copy of the reasons from bounding
boxes, and had drifted: its summary still said `"cube not contained"` after the
stage began distinguishing a cube never seen from one the window cuts.

> **SUPERSEDED 2026-08-23.** The two-outcome taxonomy below was replaced by the
> three-verdict model — `pass` / `warning` / `reject` — in E-stage0-verdicts.
> `_reject_reasons` is now `_frame_verdict`, and the important change is that a
> clipped cube and a missing band are both **warnings** rather than rejections:
> refusing them was costing captures that were perfectly measurable. This entry
> is kept for the reasoning about *severity*, which carried over intact.

**Then**, from a single helper `_reject_reasons(cube_seen, band_seen, cube_ok,
band_ok)` in `prep.py`, used by both the stage and the summary:

| condition | reason | severity | rejected? |
|---|---|---|---|
| band missing, cube seen | `marker missing, not crucial` | not crucial | no |
| cube missing, band seen | `cube missing, crucial` | crucial | yes |
| both missing | `marker and cube missing, very crucial` | very crucial | yes |
| detected, does not fit | `objects out of window` | crucial / very crucial | yes |

Two questions kept apart, because the remedies differ: was the object **seen**, and
did it **survive the window**. A missing band costs the cut but not the scale, and
the cut only needs the band on some frames, so that frame is reported and kept. A
missing or clipped cube costs the scale of every number the run reports, silently.

Which object left the window is deliberately not reported. The window is not
user-adjustable — it is the largest square the photo allows, placed by this stage —
so the remedy is identical either way. Severity still tracks the consequence even
though the message does not, and `cube_ok`/`band_ok` stay in `manifest.json`.

**Rejected frames are now written, uncropped.** Anything not croppable goes to VGGT
untouched: a window that could not hold the cube and the band would cut one of them,
so VGGT's own preprocessing is the better of two bad options. Verified —
`frame_00.png` comes out 2160×3840 raw while the accepted frames are 518×518.
`--continue-on-rejected` now controls only whether the run proceeds, not whether the
frames exist, so a refused capture can still be inspected.

New fields in `framing.json`: `severity`, `cube_seen`, `band_seen`. The web app
shows severity as a colour-coded chip beside the reason.

---

## E-viewer-scale — a schema change that silently destroyed the 3D · Aug 2026

**Symptom.** After Stage 6 was reverted to main's version, the shipped samples
rendered but every freshly measured run showed an empty viewport — no error in the
console, no failed request, the panels and sliders all correct.

**Cause.** Main's Stage 6 writes `ext_x/ext_y/ext_z/size_*_cm`; the viewer read
`obb_a/obb_b/obb_c/height_cm`. Missing columns parsed as 0, so
`linearScale = 14.0 / ((0 + 0) / 2)` returned **Infinity**, which `usePly` then
multiplied into every vertex. The geometry became `NaN`. Nothing errored because
nothing failed — the file arrived and parsed; it was ruined afterwards.

**Why it misled.** The split between working samples and broken live runs pointed
at the network path between the browser and the compute service. That path was
fine: all files 200, CORS clean, cross-origin fetches verified from the page. The
fault was in interpreting the bytes, not moving them.

**Fix.** The viewer now mirrors whichever Stage 6 wrote the file:

| CSV | derivation | cm/unit |
|---|---|---|
| parked Stage 6 | fitted-face horizontals | 59.79 |
| main's Stage 6 | `k = real/mesh`, `k^(1/3)` | 60.86 |
| main's Stage 6 via extents | **rejected** — AABB measures a tilted cube's diagonal | 44.01 |

Mirroring beats choosing the better method: a viewer that scaled differently from
the stage would draw an object at a size contradicting its own printed volume.
`linearScale` returns `null` instead of a non-finite number, and the viewport says
so.

**Two lessons worth keeping.** A missing CSV column produced a *worse* failure than
a missing file, because the missing-file path had error handling and the
missing-column path silently produced Infinity. And the earlier docs recorded this
consequence as "new runs will not display their numbers" — the numbers were the
visible part, but it was the geometry that died. A consequence written down
imprecisely is not much better than one not written down.

---

## E-ghost-steps — which cleaning steps change the answer · Aug 2026

**Question.** `ghost_voxel_downsample` and `normal_aware_filter` both look like decimation.
Does either change the reported volume, or are they ceremony?

**Method.** Stages 3-6 re-run four times from one shared Stage 1-2 source
(`--src verify`), each step disabled in turn. `GHOST_VOXEL_FACTOR = 0` disables
dedup; the normal filter was disabled by raising `max_deviation` past rejection.
Both files restored afterwards and verified identical to the commit.

| variant | points into Stage 4 | limb volume | vs current |
|---|---|---|---|
| A · full chain | 14,722 | 1091.79 cm³ | — |
| B · no `normal_aware_filter` | 15,057 | 1090.97 cm³ | **−0.08%** |
| C · no `ghost_voxel_downsample` | 84,751 | 1121.22 cm³ | **+2.70%** |
| D · neither | 90,321 | 1118.89 cm³ | +2.48% |

**`ghost_voxel_downsample` — KEPT.** 2.7% is larger than the reference cube's own residual,
so it is not noise. The direction matches the mechanism: without it the ghost
survives, the surface stays a thick band, and the alpha shape wraps a fatter
solid. It also keeps Stage 4's input at 14.7k points rather than 84.8k.

Its mechanism is not what it looks like. It does **not** collapse the ghost —
two sheets 2 mm apart fall in different voxels and both survive, and on its own it
leaves the shell at 1.62 mm against a random subsample's 1.39 mm. What it does is
set the point spacing, and `MLS_RADIUS_MULT` is measured in spacings: 0.94 mm
spacing gives MLS a 3.75 mm radius, 2.10 mm spacing gives 8.38 mm, against a ~2 mm
sheet separation. Decimating is what lets MLS's neighbourhood span both sheets.
Running MLS with no decimation at all reaches only 1.12 mm against 0.52 mm.

**`normal_aware_filter` — REMOVABLE, DECISION DEFERRED.** 0.08% on volume and 2.9%
of points, which is indistinguishable from nothing. B even keeps *more* points than
A and lands on the same volume: the strays it would remove get wrapped by the alpha
shape regardless.

Not deleted, because this violates the decision rule at the top of this file. The
evidence is one capture, measured against the pipeline itself rather than a known
volume, and `small_leg` reconstructs cleanly either way. The failure the step
guards against — misoriented points seeding spurious tetrahedra — is a *bad
capture* failure. **Re-run this A/B on the first dataset with independent ground
truth and decide it there.**

Full working, with the cross-section after every function:
[`ghost_removal_chain.md`](experiments/ghost_removal_chain.md).

---

## E-dedup-impl — is `ghost_voxel_downsample` replaceable by `open3d.voxel_down_sample`? · Aug 2026

**Question.** `ghost_voxel_downsample` is ~30 hand-written lines. Open3D ships the same
operation. Same call site, same voxel size, only the function swapped — does the
answer change?

| | box | limb | into Stage 4 | limb volume |
|---|---|---|---|---|
| `ghost_voxel_downsample` (current) | 105,045 → 16,432 | 118,478 → 17,683 | 14,722 | 1091.79 cm³ |
| `o3d.voxel_down_sample` | 105,045 → 16,385 | 118,478 → 17,736 | 14,811 | 1088.59 cm³ |

**0.29% apart.** The only substantive difference is the grid anchor —
`ghost_voxel_downsample` uses `points.min(axis=0)`, Open3D its own origin — so cells fall
elsewhere, a few points cross a boundary and the centroids shift. After MLS the
patch spread is 0.23 mm either way.

**NO DECISION TAKEN.** 0.29% is small but measured on one capture against the
pipeline itself, and it is not clearly below the noise floor of a system whose
reference cube carries a ~2% residual. Deferred to ground truth, with
`normal_aware_filter` (E-ghost-steps).

Three things any swap must preserve, recorded so the work is not redone: colour
averaging (Open3D does it only when the cloud carries colours), the
`voxel_size <= 0` escape hatch that `GHOST_VOXEL_FACTOR = 0` uses to disable the
step, and a deterministic grid origin.

**Method note worth keeping.** These two measured 0.59 mm against 0.75 mm on
cross-section shell thickness — a 27% gap suggesting `ghost_voxel_downsample` was clearly
better — while on volume they differ by 0.29%. Shell RMS in one slice is a noisy
proxy and was over-read earlier in this investigation. Where a proxy and the
reported volume disagree, the volume decides.

Figure: [`dedup_vs_open3d.png`](experiments/dedup_vs_open3d.png), working in
[`ghost_removal_chain.md`](experiments/ghost_removal_chain.md).

---

## E-recon-method — why alpha shape, when Poisson and ball pivoting look better · Aug 2026

> **Partly superseded 2026-08-23.** The comparison of *fidelity* and the argument
> that **χ, not watertightness, is the discriminator** both stand and are the
> reason the χ warning now exists. But the Poisson column was measured against a
> Stage 5 that never called `repair()`; with that fixed Poisson reaches χ = 2 and
> is now the default. Ball pivoting is unaffected — it fails at χ = 256 either
> way. See **E-psr-adopted**.

**Question.** Alpha shape produces the least attractive surface of the three
available reconstructors. Ball pivoting interpolates the points exactly and
Poisson is smooth. Is the choice defensible, or is it inertia?

**Method.** All three run on the **same** Stage 3 clouds from
`work/parity_stagerun` (box 19,573 pts, limb 15,447 pts), with the same normal
estimation, and each is then put through the pipeline's own Stage 5 repair path
(`workers/meshfix_worker.py`: `pymeshfix.fill_holes()`, then Open3D
`fill_holes()`). Volumes are scaled by each arm's own reference cube, exactly as
main's Stage 6 does, so the comparison is like for like.

Figure: [`experiments/recon_method_comparison.png`](experiments/recon_method_comparison.png).

**As produced, before any repair**

| method | faces | components | χ | watertight | p95 point-to-surface |
|---|---|---|---|---|---|
| alpha shape (25×) | 15,880 | **1** | **2** | **yes** | 2.39 mm |
| Poisson, depth 9 | 62,164 | 11 | 0 | no | **1.02 mm** |
| ball pivoting | 28,012 | 133 | −2057 | no | **0.00 mm** |

Ball pivoting passes through **every input point exactly** — it interpolates
rather than approximates — and Poisson fits 2.5× closer than alpha. On fidelity
alone, alpha shape is the worst of the three, and the intuition that the other
two "look better" is correct.

**After the pipeline's own repair**

| method | χ after | limb volume | vs alpha |
|---|---|---|---|
| alpha shape | **2** | 1081.94 cm³ | — |
| Poisson | 22 | 1069.56 cm³ | −1.1% |
| ball pivoting | 256 | 1409.99 cm³ | **+30.3%** |

**Verdict — the discriminator is χ, not watertightness.** All three become
watertight after repair, so "not watertight" is not the argument and never was:
PyMeshFix will close a topological mess into something `is_watertight` calls
true. What it closes ball pivoting's 133 shredded shells into is a bag of small
separate blobs, and the signed volume of that is 30% wrong while the mesh reports
as closed.

χ = 2 is the test that separates *one closed genus-0 solid* from *a closed
something*. It is a yes/no property of the surface rather than a judgement, which
is why Stage 4's ladder selects on it and why the fallback ranking uses
`|χ − 2|`.

Poisson is the closest competitor and worth recording as such: −1.1% with a
2.5× better fit. Its problem is that χ = 22 carries no guarantee — nothing says
how wrong it might be on a different capture, and nothing would flag it.

**An earlier run of this comparison** is in
[`stage06_experiments.md`](stage06_experiments.md), on a different Stage 3 cloud
(the leg mesh with a fabricated base). Its qualitative conclusion is the same —
alpha shape is the only method that closes without repair — but its post-repair
numbers are **−5.5% for Poisson and −25.8% for ball pivoting**, where this run
gives −1.3% and **+30.8%**.

That the sign of ball pivoting's error flips between two captures is not a
contradiction to explain away; it is the strongest form of the argument. A method
whose error is −26% on one cloud and +31% on another, both times reporting
`is_watertight == True` after repair, cannot be corrected for or bounded. Alpha
shape's χ = 2 requirement is what refuses that class of answer up front rather
than discovering it later.

The structural reason, from that earlier entry: alpha shape works from a 3D
Delaunay tetrahedralisation and **never uses normals**. Poisson integrates a
normal field and ball pivoting seeds its ball on normals, and the fabricated base
— a flat cap plus a swept skirt — has no well-defined normal.

**A fourth property, measured later.** Alpha shape is also the most *robust* of
the three. Perturbing the input cloud by removing a few percent of its points
moves alpha's volume by 0.03%, Poisson's by 5.8%, and ball pivoting's by a factor of four.
See E-slice-outliers below.

**Not tested.** Poisson at other depths or trim quantiles; ball pivoting at other
radii. Neither is likely to move χ into range, but neither was swept.

---

## E-preconfirm-scale — can the Review screen get its scale without Stages 4-6? · Aug 2026

**Question.** The chart puts Stages 4-6 after the cut confirmation, but they also
run before it, on the reference cube alone, to produce the cm-per-unit scale the
Review screen needs to draw the cutting plane in real units. Can that scale come
from Stage 3 instead, so 4-6 run exactly once?

**Method.** Take the oriented bounding box of Stage 3's own segmented
`box.ply`, identify the vertical axis from the OBB's rotation, and derive
`14.0 / mean(horizontal edges)`. Compare against Stage 6's measured
`linear_scale`.

| source | scale |
|---|---|
| Stage 3 cube cloud, OBB mean horizontal edge | **40.59 cm/unit** |
| Stage 3 cube cloud, OBB min horizontal edge | 42.48 cm/unit |
| Stage 6 (main's, volume ratio) | **60.87 cm/unit** |
| Stage 6 (parked, fitted faces) | 59.79 cm/unit |

**Verdict — rejected, 33% out.** The OBB's smallest extent is 0.3296 units where
the cube's true edge is `14 / 60.86 = 0.2300`, so the box is 43% too large. The
cause is that `box.ply` is not only the cube: Phase C extends the base to the
floor, adding 1,199 wall points below it, and the OBB spans those too.

The cube has to pass through reconstruction for its size to be known, so the
calibration branch stays. It is now labelled as calibration in the chart and in
`service/app.py`, and a postcondition check (`service/jobs.py:_postcondition`)
fails the job if that pass ever measures the *subject* — the thing the two-pass
split exists to prevent.

**Not tested.** Fitting the cube's faces on the point cloud directly (RANSAC
parallel-plane pairs), which is what `pipeline/core/faces.py` does for a mesh.
That would likely work and would let 4-6 run once, at the cost of a second scale
derivation to keep honest.

---

## E-slice-outliers — per-z-level outlier removal after MLS · Aug 2026

**The idea, and it is a good one.** In the cross-section of
`recon_method_comparison.png` a few points on the right sit well away from the
outline, and alpha shape reaches out and wraps them, producing a visible notch.
The proposal: a limb is a stack of closed cross-sections, so judge each point
against **its own horizontal slice** rather than against the 3D cloud. A point
can sit at a perfectly ordinary 3D density and still be nowhere near its slice's
outline, because its neighbours are above and below it rather than around it. A
global statistical filter cannot see that; a per-slice one can.

Figure: [`experiments/slice_outlier_removal.png`](experiments/slice_outlier_removal.png).

**Method.** Slices of 3 × mean NN spacing, overlapping by half so no point is
judged only by a bin edge. Inside each slice, work in 2D: a point's isolation is
its k-th nearest neighbour distance among the slice's own members, and it is
flagged if that exceeds `mult` × the slice's median. A point is removed only if
flagged in **every** slice containing it. Then all three reconstructors are re-run
on the filtered cloud, both objects, and calibrated as Stage 6 does.

The filter is not in the pipeline. Its implementation is in the appendix below so
it can be revived without rewriting it.

**What it removes**

| threshold | limb removed | cube removed |
|---|---|---|
| `mult = 4.0` | 2 pts (0.01%) | 0 |
| `mult = 3.0` | 453 pts (2.93%) | 111 pts (0.57%) |
| `mult = 2.0` | 627 pts (4.06%) | 837 pts (4.28%) |

**What it changes**

| | alpha raw vol | alpha χ | alpha rung | Poisson χ | ball pivot χ |
|---|---|---|---|---|---|
| no filter | — | 2 | 25× | 22 | 256 |
| `mult = 3.0` | **−0.03%** | 2 | 25× | **2** | 334 |
| `mult = 2.0` | −2.48% | 2 | 25× | **2** | 398 |

**Verdict — sound idea, three findings, and it is not being wired in.**

**1. The thing it was aimed at is not an outlier.** The notch survives every
arm, and the reason is visible in the side view: those points are **256 of them,
spanning 74 mm of height, in the same DBSCAN cluster as the body** at an 8 mm
epsilon. It is a connected protrusion, not scatter. It only looks isolated
because a ±4 mm cross-section cuts a thin sample through something that is sparse
in that plane and dense along its own length. The filter is behaving correctly by
leaving it alone; a filter that removed it would be deleting real geometry.

**2. Alpha shape is barely affected — which is itself a result.** Removing 453
genuinely isolated points changes the limb's raw volume by **0.03%**, far below
the noise floor. At 25 × point spacing alpha's ball is about 62 mm across, so a
handful of stray points distort the *outline* visibly while contributing almost
no *volume*. The defect looks far worse than it costs. Push the threshold to
`mult = 2.0` and the volume moves −2.48%, but by then it is removing 4% of the
cloud and starting to eat real surface — the cube loses 4.3%, which is a flat
dense face that should have no outliers at all.

**3. It helps Poisson substantially, and harms ball pivoting.** PSR's Euler
characteristic goes **22 → 2** on the limb and 30 → 4 on the cube — the filter is
enough to make Poisson topologically valid, which nothing else achieved. That is the
largest single improvement anywhere in this experiment, and it is worth
remembering: if Poisson is ever wanted, this is the preprocessing that moves it
towards viable. Ball pivoting goes the other way — 256 → 334 → 398 — because it
needs dense coverage and every removed point opens another hole for the repair to
close arbitrarily. Its volume swings to 6151 cm³ then 3951.

So the filter does not improve what ships, but it does explain something the
method comparison left open: **alpha shape's advantage is partly robustness.** It
is the only one of the three whose answer barely moves when the input cloud is
perturbed by a few percent.

**Not tested.** Whether the notch is anatomy, a fold, or a reconstruction
artifact — that needs the photographs, not the cloud. Whether a *shape*-aware
per-slice rule (fit the outline, drop points far from it) would do better than a
density rule; it would catch the notch, which is exactly why it might be wrong to
use.

### Appendix — the filter

```python
def slice_outlier_removal(points, colors=None, slice_h=None, k=6,
                          mult=3.0, min_pts=12):
    """Drop points isolated within their own horizontal slice."""
    import numpy as np
    from scipy.spatial import cKDTree
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < min_pts:
        return points, colors, {"removed": 0}
    d, _ = cKDTree(points).query(points, k=2, workers=-1)
    spacing = float(d[:, 1].mean())
    slice_h = slice_h or 3.0 * spacing
    z = points[:, 2]
    lo, hi, step = z.min(), z.max(), slice_h / 2.0
    flagged = np.zeros(n, dtype=np.int32)
    seen = np.zeros(n, dtype=np.int32)
    for i in range(int(np.ceil((hi - lo) / step)) + 1):
        z0 = lo + i * step
        m = np.where((z >= z0) & (z < z0 + slice_h))[0]
        if len(m) < min_pts:
            continue
        xy = points[m][:, :2]
        kk = min(k, len(xy) - 1)
        dd, _ = cKDTree(xy).query(xy, k=kk + 1, workers=-1)
        iso = dd[:, -1]
        med = float(np.median(iso))
        seen[m] += 1
        if med > 0:
            flagged[m] += (iso > mult * med).astype(np.int32)
    keep = ~((seen > 0) & (flagged == seen))
    stats = {"removed": int((~keep).sum()), "slice_h": slice_h, "spacing": spacing}
    if colors is None:
        return points[keep], None, stats
    return points[keep], np.asarray(colors)[keep], stats
```

---

## E-outline-statistic — the cross-section outline, and a number it invented · Aug 2026

**Prompted by looking at the picture.** In `mls_ghost_limb_section.png` the
cross-section outline has sharp corners — a pronounced vertex near the top, and
another on the right. A leg has no corners. That observation turned out to be
about the *measurement*, and it invalidates one number the rework had been
quoting.

Figure: [`experiments/outline_statistic.png`](experiments/outline_statistic.png).

**What the outline was.** Angular wedges around the centroid, and in each wedge
the **maximum** radius. Every vertex is therefore a single extreme point, and the
polygon connecting them is spiky by construction. That is the whole explanation of
the corners: they are the most distant point in each 20° wedge, not features of
the limb.

**What that does to the number.** The consequence is worse than cosmetic. Before
MLS the surface is a doubled shell 1.76 mm thick. A max-radius outline traces its
**outer** face. MLS collapses the shell to 0.79 mm, so the outer face moves inward
by roughly half the reduction — and the metric reports that as a large loss of
cross-sectional area which the limb never had.

Same points, same slice, only the per-wedge statistic changed:

| statistic | area (cm²) | plane vs no MLS | quadratic vs no MLS |
|---|---|---|---|
| max | 65.3 | **−7.57%** | **−6.41%** |
| p90 | 63.7 | −6.22% | −5.13% |
| mean | 60.9 | −1.41% | −0.50% |
| **median** | **60.7** | **−0.59%** | **+0.42%** |

The recorded figures were **−7.64% and −6.67%**. The max row reproduces them to
about a tenth of a percentage point, which is how the cause was identified rather
than guessed. Those two percentages are withdrawn.

**What survives, and is now much better supported.** The reason the quadratic fit
is preferred over the plane:

| statistic | quadratic − plane | positive in |
|---|---|---|
| max | +1.05 pp | 98% of slices |
| p90 | +1.17 pp | 98% |
| mean | +1.01 pp | 98% |
| median | **+1.10 pp** | **100%** |

Median of 40 slices spanning 15-90% of the limb's height, IQR +0.86 to +1.47 pp.
The conclusion was drawn from **one** slice with a metric that inflated it; it now
rests on 40 slices and holds under every statistic tried. A plane fitted to a
curved surface sits inside it, the quadratic does not, and the difference is about
one percentage point of cross-sectional area.

**Verdict.** The absolute area percentages in
[`experiments/rework_main_vs_current.md`](experiments/rework_main_vs_current.md)
were an artifact of the outline statistic and are corrected there. The
plane-versus-quadratic decision was right, and is now measured properly. Use the
**median** radius per wedge for any future cross-section, at 5° rather than 20°
— at 20° the polygon still undershoots a convex outline by a few percent even
with a robust statistic.

**Not tested.** Whether the same statistic was used for the shell-RMS column.
That column is a distance distribution, not an outline, so it is not exposed to
this; but it was not re-derived.

---

## E-ghost-origin — the two sheets are two groups of cameras · Aug 2026

**Prompted by a theory**: if the ghost is a doubled surface, perhaps one of the
two sheets is the true one — specifically the **inner** one — and MLS projecting
onto the midpoint is therefore biasing every surface outward.

Testing that first needed to establish what the two sheets actually are. They
turn out to be something more specific than "a duplicate".

**Method.** Stage 1 produces `world_points` per view, shape `(S,518,518,3)`, so
every point's source camera is known by construction. Rebuild the cloud with a
view label attached, levelled with the same `R_total`, take a slab through the
limb, split each 5° wedge at the largest radial gap, and ask which views
contributed to each side.

**Result — the split is perfectly clean.**

| view | points on the inner sheet | on the outer sheet |
|---|---|---|
| 0 | 0 | 107 |
| 1 | 0 | 45 |
| **2** | **264** | 0 |
| 3 | 0 | 15 |
| **4** | **167** | 0 |
| 5 | 0 | 142 |

Views **2 and 4** place the surface about **2.70 mm inside** where views
**0, 1, 3 and 5** place it, and no wedge mixes them. This is not scatter, and it
is not a duplicate emitted by the model per-pixel: it is a **systematic
disagreement between two groups of cameras**, i.e. a pose or scale inconsistency
in VGGT's own registration.

The same 2.7 mm separation appears on the reference cube, so it is a property of
the reconstruction, not of the subject.

**On the theory itself — unresolved, and the obvious tests are confounded.**

- **Density does not decide it.** In wedges where both sheets appear they are
  evenly populated: limb 51.9% inner against 48.1% outer, cube 48.0% / 52.0%.
  Neither sheet is a faint copy of the other.
- **Vote does not decide it.** Four views say outer, two say inner — but the two
  inner views contribute *more points* in the wedges where both appear.
- **Comparing the groups directly is confounded.** Measuring the cube and the
  limb from each group separately gives a scale-invariant ratio differing by
  9.5%, but the two groups see different faces, so partial coverage contaminates
  that number. It is recorded here as *not evidence*.

**Why it matters, if it can be settled.** The offset is additive, so it does not
cancel under calibration. A surface offset δ inflates the cube's edge by 2δ on a
~230 mm edge and the limb's diameter by 2δ on a ~108 mm diameter — the limb is
affected about twice as hard in relative terms. If the inner sheet is the truth
and MLS lands on the midpoint, the limb is being over-measured by roughly a
percent of diameter more than the cube is, which does not divide out.

**What would settle it.** An independent length in the scene. The ArUco markers
are printed at a known size, so recovering their 3D corners per view group and
comparing against that printed size would say which group has the correct scale —
that is the experiment to run, and it needs `REFERENCE_MARKER_CM` measured with a
ruler first. Failing that, water displacement on a held-out object.

**Not tested.** Whether the split is stable across captures, or whether it tracks
something identifiable about those two camera poses (they may be the two most
oblique views). Both are cheap once a second dataset exists.

---

## E-psr-swap — can Poisson replace alpha shape? · Aug 2026

> **SUPERSEDED 2026-08-23 — the conclusion was wrong.** Everything below was
> measured against a Stage 5 that called only `pymeshfix.fill_holes()` and never
> `repair()`. With the stronger repair in place Poisson reaches χ = 2 on both
> objects in 38 of 46 configurations rather than 0 of 48, and it is now the
> pipeline's default. See **E-psr-adopted**. The reasoning below about *why* χ
> matters still holds; the verdict against Poisson does not.

**Why it was asked.** In E-slice-outliers, adding the per-slice filter pushed
Poisson's Euler characteristic on the **limb** from 22 to 2. If it can be made
topologically valid it is the better surface — it fits the points 2.3× closer
than alpha (p95 1.02 mm against 2.39 mm).

**Method.** Sweep the three knobs that could plausibly fix it — Poisson depth
(8, 9, 10, 11), low-density trim quantile (0, 0.01, 0.03, 0.10) and the per-slice
filter (off, mult 3.0, mult 2.0) — on **both** objects, each put through the
pipeline's own Stage 5 repair. 48 configurations, 96 reconstructions.

Both objects matter: the reference cube sets the scale, so a cube that is not a
single closed solid makes every number derived from it invalid too.

**Result**

| | closes at χ = 2 |
|---|---|
| the limb | **36 of 48** configurations |
| the reference cube | **0 of 48** |

The cube's χ was 4 in 23 configurations, and 6, 8, 10, 12, 22 or 30 in the rest.
No depth, no trim, and no amount of pre-filtering closed it.

**Why, and it is structural.** Poisson solves for a smooth indicator function and
extracts an isosurface. The limb *is* smooth, so it does well. A cube is nothing
but sharp 90° edges and flat faces — exactly what a smooth indicator cannot
represent — so it rounds the edges and leaves handles where the normal field
disagrees with itself along them. Alpha shape has no such trouble because it is a
subcomplex of the points' own Delaunay tetrahedralisation: its topology comes
from the samples, not from a fitted field.

**And the number is not stable.** Across the 48 configurations the reported limb
volume ranges **972 to 1174 cm³ — a 21% spread**. Alpha shape over the same three
filter settings spans 1055 to 1082, a 2.5% spread. A method whose answer moves
20% with a parameter that has no principled setting is not a measurement
instrument.

**Verdict — no. Alpha shape stays.** The proposal is defeated by the reference
object rather than by the subject, which is worth stating plainly: if this
pipeline ever measured only smooth organic shapes, Poisson would be a serious
candidate. It measures a cube and a limb with the same code, and only alpha shape
handles both.

**Not tested.** Screened Poisson with explicit sharp-feature weighting, or
reconstructing the two objects with *different* methods — the pipeline already
supports `--box-recon-method` and `--obj-recon-method` separately, so a
Poisson limb with an alpha-shape cube is a configuration nobody has measured. It
would need the scale question settled first, since the two methods disagree by
about 1% on the same cloud.

---

## E-mixed-recon — alpha shape for the cube, Poisson for the limb · Aug 2026

> **SUPERSEDED 2026-08-23 — the conclusion was wrong.** Everything below was
> measured against a Stage 5 that called only `pymeshfix.fill_holes()` and never
> `repair()`. With the stronger repair in place Poisson reaches χ = 2 on both
> objects in 38 of 46 configurations rather than 0 of 48, and it is now the
> pipeline's default. See **E-psr-adopted**. The reasoning below about *why* χ
> matters still holds; the verdict against Poisson does not.

**The idea.** E-psr-swap found that Poisson closes the limb easily (36 of 48
configurations) and the reference cube **never** (0 of 48), for a structural
reason: a cube is nothing but sharp edges and a smooth indicator function cannot
represent them. So use each method where it works — alpha shape on the cube,
Poisson on the limb. The pipeline already supports this:
`--box-recon-method alpha_shape --obj-recon-method poisson`.

**Method.** Run stages 4-6 both ways from the same cached Stage 3, on **two**
captures. `small_leg` (6 photos, 5 accepted by Stage 0) and `short_leg`
(8 photos). Everything before Stage 4 is identical between the two arms.

> **`short_leg` is a badly framed capture.** Stage 0 rejected essentially every
> frame with `objects out of window`, and the run only proceeded under
> `--continue-on-rejected`, with all frames handed to VGGT uncropped. It is
> included precisely because a method should be judged on the capture that went
> wrong, not only the one that went right.

**Result — it works on one capture and fails on the other.**

| | `small_leg` | `short_leg` |
|---|---|---|
| cube, alpha shape | χ = **2** | χ = **2** |
| limb, alpha shape | χ = **2** | χ = **2** |
| limb, Poisson after Stage 4 | χ = −1, not closed | χ = −10, not closed |
| limb, Poisson **after repair** | χ = **2** ✓ | χ = **−4** ✗ |
| volume vs alpha/alpha | −1.02% | **−16.13%** |

**The number comes from the repair, not the reconstruction.** This is the part
that decides it. Poisson's mesh is not closed when Stage 4 hands it over, so
whatever Stage 5's PyMeshFix does to close it is part of the answer:

| | Stage 4 → Stage 5 volume change |
|---|---|
| alpha shape, both captures | **+0.00%** — already closed, repair is a no-op |
| Poisson, `small_leg` | **−4.22%** |
| Poisson, `short_leg` | **+158.08%** |

A 158% swing is the repair inventing geometry to bridge holes it cannot see the
shape of. Alpha shape never enters that regime because Stage 4 does not hand over
anything open — the ladder searches until it is closed.

**What Poisson genuinely wins.** Fidelity, and by a lot:

| | alpha p95 | Poisson p95 |
|---|---|---|
| `small_leg` | 2.39 mm | **1.30 mm** |
| `short_leg` | 11.84 mm | **1.70 mm** |

Poisson tracks the points about twice as closely on a good capture and **seven
times** as closely on the bad one. That is a real advantage and it is worth
recording, because it says something uncomfortable about alpha shape: on
`short_leg` its surface sits nearly 12 mm from the cloud at p95. Alpha buys its
guarantee with a coarse α (40× spacing there), and coarse means loose.

**Verdict — no, but this was the closest anything has come.** The proposal is
sound in principle, gets both objects to χ = 2 on a good capture, and is defeated
by generality: on the second capture the limb closes at χ = −4 and the number
moves 16%. A method that depends on the repair for between 4% and 158% of its
volume cannot be trusted to report a measurement.

**What would change the answer.** Poisson's problem here is holes, and holes come
from sparse or inconsistent coverage. If the framing gate were enforced rather
than overridden — `short_leg` should never have been measured at all — Poisson's
failure case might simply not occur. That is testable the moment a second
*well-framed* capture exists, and it is the experiment to run before dismissing
this idea for good.

**Not tested.** Screened Poisson with sharp-feature weighting; whether alpha's
11.84 mm p95 on `short_leg` is the capture's fault or the ladder giving up too
coarse; per-slice contour smoothing as a way to close Poisson's holes without
PyMeshFix.

---

## E-stage0-verdicts — three bad band detections, one unreadable format, and a redesign · Aug 2026

**Prompted by looking at an overlay**: on a capture with no marker band at all,
Stage 0 had drawn a band box around the leg.

### Three ways the band detector was wrong

**1. It found a band where there was none.** `_band_bbox` only checked that the
detected box *overlapped* the limb. An open-vocabulary detector always returns
its best candidate for "cord", and with no cord in the scene that candidate is
the leg — which is entirely on the leg, so the overlap test passed trivially.
There was no size check. Band area as a fraction of the limb's mask:

| capture | ratio |
|---|---|
| `inputs/small_leg` — a real band | **0.04 – 0.07** |
| `inputs/est_325` — no band | 1.23 |
| `inputs/short_leg` — no band | 2.19 – 4.02 |

A cord tied round a limb cannot be larger than the limb. `BAND_MAX_LIMB_FRAC`
= 0.35 — five times the largest real band, a third of the smallest false one.

**2. One detection was enough to set the marker colour.** On `short_leg` a single
74 × 60 px false positive on 1 of 8 frames — small enough to pass the size guard —
taught the pipeline that the marker is RGB(217, 207, 198). That is the floor tile.
Measured consequence, running Stage 3's detector both ways on the same cloud:

| marker colour | Stage 3 |
|---|---|
| the spurious floor colour | **198 points, cut at 61% of the limb's height** |
| none at all | **no plane** — the correct answer |

`BAND_MIN_FRAME_FRAC` = 0.6 now requires most of the capture to agree, rounded up:
4 of 6 frames, 5 of 8. A fixed count does not scale — two of six is
corroboration, two of twenty is noise. When the colour is discarded the frames
that thought they saw a band are downgraded with it, so the report cannot say
`pass` on a frame whose band the capture just threw away.

**3. A missing band disabled cropping entirely.** `can_crop` read
`band is not None and _fits(window, band)`, making a band a **precondition for
cropping at all**. On any capture without a marker — `est_325` has none by design
— `can_crop` was always False and the stage silently fell back to VGGT's own
centre crop, which is the exact failure Stage 0 exists to prevent.

### It could not read HEIC

`cv2.imread` returns None for HEIC and only `vggt/utils/load_fn.py` registers the
opener — that is Stage 1's loader. Every frame of a HEIC capture reached Stage 0
as None and was recorded `file unreadable`. On `est_325` that was **all 8 of 8**:
the gate had never once looked at that dataset. Stage 0 now falls back to PIL with
`pillow_heif`, and prints the decode error instead of returning None in silence.

### The redesign: three verdicts

Two outcomes could not express the situation. A capture with no marker band is
perfectly measurable — it simply cannot be cut automatically — and refusing it
outright was costing good captures.

| condition | verdict |
|---|---|
| everything found and framed | **pass** |
| band missing, band clipped, **or cube clipped** | **warning** — used, with a caveat |
| cube not detected, nothing detected, file unreadable | **reject** — not used |

A clipped cube is a warning and not a rejection because the cube *was* found, so
the frame is a real viewpoint; what fails is only this stage's crop, and VGGT's
centre crop takes over. Only a genuine absence is unrecoverable.

Four consumers were out of step with the new model and were fixed with it: the
per-frame `summary.txt` line re-derived its own verdict and printed `REJECTED`
for frames the pipeline was using; `framing.json` carried no per-frame `verdict`
and no `rejected` list; the end-of-stage summary only printed when something was
rejected, so a warning-only run said nothing at all; and `service/app.py` gated
`/run` on `all_passed`, which a warning-only capture can never satisfy — it would
have refused every band-free upload.

### What it is worth

| capture | before | after |
|---|---|---|
| `inputs/small_leg` | 5 of 6 usable, colour RGB [44,37,16] | **unchanged** |
| `inputs/short_leg` | 1 of 8, run refused | **8 of 8**, colour correctly discarded |
| `inputs/est_325` | 0 of 8, all unreadable | **8 of 8**, colour correctly discarded |

**Not tested.** Whether `BAND_MAX_LIMB_FRAC` = 0.35 holds for a wide bandage or a
band seen edge-on; both would read larger than a cord. Whether 0.6 is right when a
band is genuinely occluded on half the orbit. Both want a second banded capture.

**Consequence for earlier work.** `short_leg` was described in E-mixed-recon as
"a badly framed capture", and that was wrong — it was well framed and rejected for
a hallucinated band. Every frame now reaches VGGT cropped rather than raw, so the
`short_leg` baseline and the Poisson comparison built on it both need re-running.

---

## E-psr-adopted — Poisson becomes the default, after Stage 5 was fixed · Aug 2026

> **This entry supersedes E-psr-swap and E-mixed-recon.** Both concluded against
> Poisson, and both were measuring a Stage 5 that was not doing its job.

**The mistake.** `workers/meshfix_worker.py` called `pymeshfix.fill_holes()` and
nothing else, on the reasoning that it never drops faces and so preserves shape.
PyMeshFix also has `repair()`, which additionally removes self-intersections and
non-manifold edges. The pipeline never called it. `fill_holes` closes the
boundary but leaves exactly the defects that make a mesh watertight-but-invalid,
so every Poisson mesh arrived at Stage 6 closed and topologically wrong — and I
attributed that to Poisson.

**What changed when Stage 5 was fixed**

| | `fill_holes` | `repair()` |
|---|---|---|
| `small_leg` box | χ = 30 | **χ = 2** |
| `small_leg` limb | χ = 22 | **χ = 2** |
| `short_leg` box | χ = 10 | **χ = 2** |
| `short_leg` limb | χ = 14 | χ = −16 |

And it is free: the reported volume moves **0.005%** (1069.56 → 1069.61 cm³).
The shape is preserved after all; it is the tunnels that go. `joincomp` and
`remove_smallest_components` were both tried and changed nothing, so both stay
off. **Alpha-shape meshes are byte-identical either way**, because they arrive
already closed and Stage 5 is a no-op on them.

**The parameter sweep, re-run.** 48 configurations — depth 8–11, trim 0–0.10,
with and without the per-slice filter:

| | χ = 2 on **both** objects |
|---|---|
| with `fill_holes` | **0 of 48** |
| with `repair()` | **38 of 46** (two killed as redundant; depth 10 and 11 agreed everywhere) |

In the shipped configuration — no pre-filter — all twelve depth/trim combinations
with `trim ≥ 0.01` give χ = 2 on both objects, and the limb volume spans
**1069.44 to 1072.89 cm³, a 0.32% spread**. The earlier figure of 21% was the
spread across configurations that were mostly invalid.

**`trim > 0` is a hard requirement.** Every `trim = 0.00` row gives box χ = 4:
Poisson's low-density extrapolated skin is what breaks the topology, and
discarding the bottom 1% is enough. The pipeline already trims at 2–5%.

**Head-to-head, three arms, two captures**

| `small_leg` | limb cm³ | box χ | limb χ | Stage 5 moved | p95 to cloud |
|---|---|---|---|---|---|
| alpha / alpha | 1081.94 | 2 | 2 | +0.00% | 2.39 mm |
| alpha cube + PSR limb | 1070.85 | 2 | 2 | −4.22% | **1.30 mm** |
| **PSR / PSR** | **1074.32** | **2** | **2** | −4.22% | **1.30 mm** |

| `short_leg` | limb cm³ | box χ | limb χ | Stage 5 moved | p95 to cloud |
|---|---|---|---|---|---|
| alpha / alpha | 2763.24 | 2 | **2** | +0.00% | 21.80 mm |
| PSR / PSR | 2146.39 | 2 | **−18** | +2.25% | **2.19 mm** |

**Decision: Poisson is now the default for both objects.** It fits the points
1.8× closer on `small_leg` and 10× closer on `short_leg`, reaches χ = 2 on both
objects there, and is insensitive to its own parameters at 0.32%.

**What it costs, stated plainly.** Alpha shape's ladder *guarantees* χ = 2 because
it selects on it; Poisson has no such guarantee and on `short_leg` closes at
χ = −18 — a surface with about ten handles, whose signed volume is not the volume
of a solid, and which reads 22% below the alpha answer.

That capture is unusual for a specific reason: with no marker band there is no
cut, so the object is the **entire leg including the foot** — toes, arch, the gap
under the instep. Genuine topological complexity, not a reconstruction defect.

**The mitigation, which matters more than the default.** Stage 5 now computes the
Euler characteristic of every final mesh and prints a loud warning when it is not
2, naming the fallback:

```
leg_cut: 47,653 verts, 95,342 faces, watertight, chi=-18 ** NOT A SIMPLE SOLID **
  WARNING: leg_cut is closed but has chi=-18 (~10 handles or extra pieces). Its
  signed volume is NOT the volume of a solid, and Stage 6 will report it anyway.
  Re-run this object with --obj-recon-method alpha_shape, whose selection
  guarantees chi=2.
```

Before this change nothing in the live pipeline checked χ at all — the only such
check was in the **parked** Stage 6 code. A closed-but-holed mesh returns True
from `is_watertight`, so Stage 6 integrated it and reported a number with no sign
that anything was wrong. That gap existed under alpha shape too; it simply could
not fire, because Stage 4's ladder never emitted such a mesh.

**Not tested.** Whether Poisson holds up on a *cut* limb from a second capture —
both successes here are `small_leg` and both failures are the same uncut foot.
That is the experiment that would settle whether the default is right.
