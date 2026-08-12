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
Grounding DINO + SAM already exist in `pipeline/detection.py` to locate a crop
box.

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
— Stage 3 applies SOR at `clean.py:111` — but the dense cloud handed to Stage 3
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
- [ ] `mode="pad"` vs `"crop"` — crop currently discards 44% of each photo
- [ ] `conf_thres` sweep — untested, controls shell thickness
- [ ] length-based scale instead of `k^(1/3)` — more correct, ~3% shift
- [ ] multi-view consistency filter (`debug_ghost.py:165`, written but unwired)
- [ ] single-cluster scene exports nothing (`clean.py:234-252`)
