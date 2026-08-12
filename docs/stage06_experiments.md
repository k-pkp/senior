# Stage 6 — Volume: method history and what blocks a defensible result

Scope: everything about **how a real-world volume is produced** — the volume
estimator, the oriented bounding box, and above all the **scale calibration** —
from the state on `origin/main` (`bab2bbc`) to now.

Companion documents: [`experiments.md`](experiments.md) for pipeline-wide
experiments, [`update.md`](update.md) for the full change set,
[`pipeline_flowchart.md`](pipeline_flowchart.md) figure 6 for the dataflow.

> `experiments.md` still describes this as "Stage 7" and states
> `k = 2744 / box_vol`. Both are stale — the stage was renumbered to 6 and the
> calibration was replaced. This file supersedes it on those points.

---

## What Stage 6 does today

```
ref_edge     = mean of the reference cube's two HORIZONTAL OBB edges
linear_scale = REFERENCE_REAL_SIZE_CM / ref_edge          # cm per mesh unit
k            = linear_scale ** 3                          # cm³ per unit³
V_real       = V_mesh * k
size_cm      = OBB extents * linear_scale
```

Volume estimator priority per mesh:

1. **watertight** — exact signed volume, `V = (1/6) Σ (v₀ × v₁)·v₂`
2. warp + flood fill — GPU BVH surface mark, CPU flood fill
3. trimesh voxel
4. convex hull — labelled `UNRELIABLE`

Plus an independent voxel occupancy cross-check alongside the exact value.

---

## Method history

### M1 — calibrate by volume ratio · `origin/main` · **REPLACED**

```python
k = real_ref_vol_cm3 / ref_mesh_vol      # 2744 / V_box
linear_scale = k ** (1/3)
```

**Two defects.**

*Mathematical.* This uses `V_mesh^(1/3)` as the reference edge length, which
only holds for a **perfect cube**, and the cube root compounds any deviation
three times. Measured: at 2.2% off cubic it under-read the edge by 3.1% and
**inflated every volume by ~10%**.

*Epistemic, and worse.* It **forces the reference's error to zero by
construction**. The cube always reports exactly 2744 cm³ and exactly 14 cm, so
the system can never report its own accuracy. `experiments.md` flagged this at
the time: *"the cube cannot validate scale on its own."*

**Verdict:** replaced. A calibration that guarantees its own success is not a
calibration.

---

### M2 — calibrate by length, mean of **three** OBB edges · **REPLACED (tautology)**

```python
linear_scale = 14.0 / mean(obb_a, obb_b, obb_c)
```

Fixes the cube-root problem and lets the reference volume disagree with nominal.
But it introduced a subtler tautology, found by asking *"how do you get the cm
number? what's the reference?"*:

Since scale is `14 / mean(3 edges)`, the **mean of the three reported
dimensions is always exactly 14.0000**. Reporting "the cube measures 13.9 ×
14.0 × 14.1 cm" looked like validation and was arithmetic.

Only the **spread** between edges and the **volume** carried information.

**Verdict:** replaced — but note the tautology was *narrowed*, not removed. See
[Circularity](#circularity--what-is-and-is-not-evidence).

---

### M3 — calibrate by length, **two horizontal** edges only · **CURRENT**

```python
_vi  = argmax_i |R[:,i] · ẑ|          # axis most aligned with world up
ref_edge = mean(the two non-vertical edges)
linear_scale = 14.0 / ref_edge
```

The vertical edge is excluded because the cube's underside rests on the floor
and never reconstructs — including it drags the estimate small and inflates
everything measured against it.

Measured on `small_leg`:

```
horizontal  0.2264, 0.2325 units — disagree by 2.68%
vertical    0.2192 units — 4.47% short of the horizontal mean
ref edge    0.229427  ->  linear_scale = 61.02 cm/unit
```

**Verdict:** current. Vertical is now genuinely independent evidence (it does
not enter its own calibration). The two horizontals are still circular in their
mean.

---

### M4 — forced box primitive · **REJECTED (tautology)**

An attempt to make the reference clean by fitting a prism and forcing
`side = mean(fx, fy)`, so the box came out square by construction.

This is circular in the worst way: it manufactures the answer the reference is
supposed to *test*. Rejected as soon as it was stated plainly. The
`box_primitive` method still exists in the worker but is **not** the default and
must not be used for the reference.

**Verdict:** rejected. Do not reinstate.

---

### M5 — AABB → oriented bounding box · **KEPT**

`main` used axis-aligned bounds. An AABB around a yaw-rotated object reports its
**diagonal**, not its size.

| object | AABB | OBB |
|---|---|---|
| est_325 can, width | 6.09 cm | 5.75 cm |
| ArUco cube, edge | 14.95 cm | 14.0 cm |

**Verdict:** kept. A 6.8% error on the reference edge is a 21% error in volume.

---

### M6 — OBB axes ordered by **orientation**, not magnitude · **KEPT**

Sorting the three extents by size picks the wrong edge whenever two are short in
the same direction. Since Stage 3 levels the scene, the axis to exclude can be
identified **geometrically**:

```
i_vertical = argmax_i | R[:,i] · ẑ |
```

CSV columns renamed accordingly: `size_a/b/c_cm` → `height_cm` / `width_cm` /
`depth_cm`.

**Verdict:** kept. Identifying the truncated axis by geometry beats guessing
from which edges happen to agree.

---

### M7 — Euler-number integrity check · **KEPT, high value**

`is_watertight` is **not sufficient**. A surface riddled with tunnels is still
closed, and the signed-volume integral faithfully subtracts those tunnels — the
mesh looks correct from outside while reporting far too little volume.

`χ = V − E + F`, and `χ = 2 − 2g` for genus `g`. Measured on the reference cube
during α selection:

| α | watertight | χ | volume |
|---|---|---|---|
| 30× | yes | **−1** | 1898 cm³ |
| 40× | yes | **2** | 2467 cm³ |

Selecting on watertightness alone under-read by **31%**.

**Verdict:** kept. Stage 4 now selects α on watertight **and** χ = 2; Stage 6
warns loudly if χ ≠ 2 reaches it anyway.

---

### M8 — voxel occupancy as an independent cross-check · **KEPT**

Computed *alongside* the exact value, never instead of it. Boundary voxels are
counted whole so it over-reads by a few percent and converges downward onto
exact as resolution rises.

Expected band **+1% to +8%**. A result *below* exact indicates a
self-intersecting or inverted surface.

```
box.ply      exact 0.010323  voxel 0.010725  +3.89%   ✅ in band
leg_cut.ply  exact 0.003746  voxel 0.004078  +8.84%   marginally over
```

**Verdict:** kept. Two independent estimators bracket the answer — see
[Recommendation](#recommendation--report-a-bracket).

---

### M9 — honest failure labelling · **KEPT**

- non-watertight → explicit warning that flood fill **leaks** through holes and
  can under-read by an order of magnitude while returning a plausible number
- convex-hull fallback → labelled `convex_hull (UNRELIABLE)`, because it ignores
  the surface entirely and a broken mesh scores the same as a good one
- χ ≠ 2 → explicit warning

**Verdict:** kept. Every one of these fired during development and each caught a
number that would otherwise have been reported as a measurement.

---

## Current measurement

`inputs/small_leg`, 6 photos, alpha shape, both meshes watertight with χ = 2:

```
       name     method  euler   volume    obb_a    obb_b    obb_c
    box.ply watertight      2 0.010323 0.219166 0.226351 0.232503
leg_cut.ply watertight      2 0.003746 0.508577 0.123537 0.258523

       name  height_cm  width_cm  depth_cm  real_vol_cm3
    box.ply      13.37     13.81     14.19       2345.55
leg_cut.ply      31.03      7.54     15.78        851.28
```

### Decomposition of the reference's −14.5%

```
perfect cube of the measured edge    0.012076 units³
actual OBB box volume                0.011534   →   −4.49%   vertical short
actual mesh volume                   0.010323   →  −10.50%   mesh under-fills its box
                                                   −14.52%
```

Two independent defects requiring different treatment:

**−4.49% — vertical truncation.** The cube's underside rests on the floor and
never reconstructs. Partly self-inflicted: the floor-removal band is
`±2 × 0.00497 = 0.0099` units = **0.61 cm**, and the vertical deficit is
**0.63 cm**. Widening the band to eliminate the floor slab also shaves the base
off anything resting on the floor. The floor extension does fire
(`[box] extended to floor: gap 0.0113`) and recovers roughly half.

**−10.50% — the mesh under-fills its own bounding box.** Alpha shape rounds
edges and corners. This is `fill ratio = V/(abc) = 0.8950` where a perfect cube
is 1.0000.

---

## Circularity — what is and is not evidence

Because `linear_scale = 14.0 / mean(two horizontal edges)`, **the mean of the
two reported horizontal dimensions is exactly 14.0000 by construction**:

```
13.81 and 14.19  →  mean = 14.0000
```

That is the definition rearranged, not a result. **"The cube measures ~14 cm"
must never be quoted as validation.**

### Genuinely non-circular evidence available today

| quantity | value | why it is real |
|---|---|---|
| horizontal edge spread | 2.68% | a ratio; scale cancels |
| vertical vs horizontal | −4.47% | vertical is excluded from its own calibration |
| fill ratio `V/(abc)` | 0.8950 | dimensionless; 1.0 for a perfect cube |
| voxel vs exact | +3.89% | dimensionless, independent estimator |
| euler | 2 | topological invariant |

Five real numbers. **"2346 vs 2744" is not one of them in isolation** — it is
only meaningful once `REFERENCE_REAL_SIZE_CM = 14.0` is itself verified.

---

## Why NOT to go back to volume calibration

The tempting fix is `k = 2744 / V_ref_mesh` — it forces the reference to zero
error and cancels any shared multiplicative bias.

**Do not.** The rounding loss is **shape-dependent**:

| | fill ratio | sharp features |
|---|---|---|
| cube | 0.8950 | 12 edges, 8 corners — essentially all of its surface |
| limb | 0.2307 | none |

A cube loses 10.5% to rounding *because it is nothing but edges and corners*. A
smooth limb loses far less proportionally. Applying the cube's correction factor
to the limb would **over-correct** it — and you would never detect the mistake,
because the reference would read 2744 cm³ by definition.

Length calibration is correct. The reference's error must stay visible.

---

## Recommendation — report a bracket

The two independent estimators already computed bound the answer from opposite
sides:

```
leg   exact  851.3 cm³     interpolating surface, rounds inward   → lower bound
      voxel  926.5 cm³     boundary voxels counted whole          → upper bound

      →  851 – 927 cm³
```

Costs nothing — both numbers are already in `volumes.csv`. A single point
estimate implies a precision the pipeline does not have.

---

## What blocks a defensible result

Ranked. Items 1–3 are **experiment design, not code**.

### 1. `REFERENCE_REAL_SIZE_CM = 14.0` is unverified — **blocks everything**

The cube is handmade: cardboard, printed markers, taped edges. A **2 mm build
error is 1.4% linear = 4.3% volume**, applied to every result the project will
ever produce.

Every accuracy claim is conditional on a number nobody has measured.

**Action:** caliper all three edges, several times each. Two reasons it matters
more than it sounds:

- If the real cube is 13.7 cm, part of the −14.5% is your denominator.
- If the real cube is not square, part of the 2.68% spread is **real object**,
  not pipeline error.

**Cost:** minutes. **Value:** unblocks every other claim.

### 2. No second reference — scale cannot be checked at all

With one known object, scale cannot be validated: the cube *defines* it. Shape
can be checked; absolute size cannot. One equation, two unknowns.

**Action:** add a second object of known dimensions with a **low edge-to-volume
ratio** (ball, cylinder). Calibrate on the cube, **predict** the second object's
size, compare to measured truth.

- scale error scales both objects equally
- rounding hits the cube hard and the sphere barely

Two shapes, two equations → a **bias model** instead of one mystery percentage,
and a stated expected error for the limb rather than a guess from the cube.

**That prediction error is the accuracy figure this project actually needs, and
the pipeline currently cannot produce one.**

### 3. No ground truth for any measured object

| object | what we have | what we need |
|---|---|---|
| est_325 can | "325 ml" — **fill** volume | **external displacement**, unmeasured; likely 340–355 cm³ |
| legs | nothing | water displacement |

Fill volume and external displacement are different quantities. No error
percentage against 325 ml is defensible.

**Action:** water displacement. Archimedes, 1 mL = 1 g on a kitchen scale. This
is the clinical gold standard for limb volumetry and the only thing that can
confirm 851 cm³.

### 4. The measured error is inside the noise floor

From [`experiments.md`](experiments.md): point-shell radial noise on the can is
**std 0.214 cm on a 2.62 cm radius**. Volume goes as `r²`, so that is
**±16% volume**.

```
radius p25 → 260.2 cm³
radius med → 286.4 cm³
radius p75 → 333.1 cm³
```

**The reference's −14.5% is smaller than the ±16% noise floor.** On a single
dataset it cannot be distinguished from noise, and any change moving the result
by less than that cannot be judged.

**Action:** repeat runs across multiple capture sessions and report a
distribution, not one number. Until then, single-run deltas below ~16% mean
nothing.

### 5. MLS shrinkage is unresolved

`MLS_RADIUS_MULT = 4.0` removes **4.6% of hull volume** by design. Whether that
is noise removal or real surface depends on whether the true surface is the
inner or the outer ghost sheet — which we cannot currently determine.

That 4.6% sits inside the −14.5% and is not separable from it.

### 6. The floor-band trade-off

`PLANE_REMOVAL_BAND_MULT = 2.0` shaves **0.61 cm** off anything resting on the
floor. It was necessary — at 1.0 the floor slab survived and corrupted
clustering entirely — but it costs the reference roughly its whole vertical
deficit.

**Possible fix:** extend the box down to the floor plane *after* removal, since
we know a priori that it rests there. Not attempted.

---

## Summary

| | status |
|---|---|
| volume estimator | solid — exact signed volume, χ = 2 verified, independent cross-check |
| calibration method | correct in form — length, not volume; horizontals only |
| calibration **value** | **unverified** — `14.0` never measured |
| reported accuracy | **not yet possible** — no second reference, no ground truth |
| single-run sensitivity | **±16%** — larger than the effect being measured |

The pipeline is in good shape. The **measurement protocol** is not, and no
amount of further code will fix that. The next three actions are a caliper, a
second reference object, and a bucket of water.
