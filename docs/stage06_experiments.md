# Stage 6 — Volume: method history and what blocks a defensible result

Scope: everything about **how a real-world volume is produced** — the volume
estimator, the oriented bounding box, and above all the **scale calibration** —
from the state on `origin/main` (`bab2bbc`) to now.

Companion documents: [`experiments.md`](experiments.md) for pipeline-wide
experiments, [`progress.md`](progress.md) for the full change set,
[`pipeline_flowchart.md`](pipeline_flowchart.md) figure 6 for the dataflow.

> `experiments.md` still describes this as "Stage 7" and states
> `k = 2744 / box_vol`. Both are stale — the stage was renumbered to 6 and the
> calibration was replaced. This file supersedes it on those points.

---

> ## STATUS — reverted to M1, awaiting review
>
> **`pipeline/stages/volume.py` now runs main's version, i.e. M1.** Everything
> from M2 to M11 is parked below main's code in that file as a commented block
> under a `PARKED` header, kept for review rather than deleted. `stagerun.py` and
> `pipeline/orchestrator.py` carry matching `PARKED` markers where the
> `inference_dir` argument used to be threaded through for M11's marker check.
>
> This was a deliberate hand-off, not a rollback on evidence — the stage's author
> should decide what to adopt. But **read the M1 entry and
> [Why NOT to go back to volume calibration](#why-not-to-go-back-to-volume-calibration)
> before deciding**, because both were written against exactly the state that is
> now running again. In particular, with M1 active the reference cube reports
> exactly 2744.00 cm³ on every run, by construction. That is an identity, not a
> measurement, and the stage currently emits no error signal about itself.
>
> Two consequences to be aware of:
>
> - Dimensions come from an axis-aligned box again (M5 is parked), so the 14 cm
>   cube reports **19.18 × 19.47 × 14.09 cm** — an AABB around a tilted cube
>   measures its diagonal.
> - The CSV columns changed back (`ext_x/ext_y/ext_z/size_*_cm`) from the parked
>   method's `obb_a/obb_b/obb_c/height_cm`. The web viewer now reads both and
>   mirrors whichever derivation wrote the file, so runs display either way — but
>   the scales differ (59.79 cm/unit from fitted faces against 60.86 from the
>   volume ratio), and main's axis-aligned extents must **not** be used for scale:
>   on a tilted cube they measure the diagonal and give 44.01, a 26% error. See
>   `progress.md` §3.5.
>
> Stages 0–5 are unaffected — verified by re-running them after the revert and
> byte-comparing the meshes, which came out identical.

## What Stage 6 does today

**Active (M1, main's version):**

```
k            = real_ref_vol_cm3 / ref_mesh_vol            # 2744 / V_box
linear_scale = k ** (1/3)                                 # cm per mesh unit
V_real       = V_mesh * k
size_cm      = AABB extents * linear_scale
```

**Parked (M10/M11, in the commented block):**

```
ref_edge     = the cube's own fitted FACE separations, horizontals only
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

### M1 — calibrate by volume ratio · `origin/main` · **ACTIVE AGAIN (reverted to; see STATUS)**

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

**Verdict:** replaced at the time, on the reasoning above — and then reverted to,
in Aug 2026, as a hand-off rather than a reversal of that reasoning. Both defects
still hold and are reproducible on demand: a cold run on `inputs/small_leg` reports
the cube at exactly 2744.00 cm³ and 19.18 × 19.47 × 14.09 cm. A calibration that
guarantees its own success is not a calibration; that judgement has not changed,
only who gets to make it.

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

### M3 — calibrate by length, **two horizontal OBB** edges · **REPLACED by M10**

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

**Verdict:** superseded by M10. The rule was right; the *ruler* was not. The
oriented bounding box overstated the edge by 2.2%, which under-read every volume
by ~6%.

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

### M10 — measure the edge from the cube's own FACES · **PARKED**

The single largest accuracy gain in Stage 6, and it changed nothing about the
*rule* — only about how the edge is measured.

An oriented bounding box has to guess the orientation. On this reference it
guessed about **1.3 degrees wrong**, which is enough to enclose 6.8% more volume
than the convex hull of the very same points — and an OBB must *contain* the
hull, so that excess is pure fitting error:

```
face-to-face box       2505.7 cm3     the cube's real size
convex hull of points  2479.2 cm3     -1.1%
oriented bounding box  2647.8 cm3     +5.7%    <- must contain the hull, yet exceeds it
```

Both Open3D and trimesh fail here. Open3D's `get_oriented_bounding_box` returned
boxes with **2.1x the AABB volume** on this cloud; trimesh's was +5.7%. Neither
is trustworthy for a measurement, and this *is* a measurement.

A cube does not need a bounding box. **Its face normals are its axes.**
`pipeline/core/faces.py` clusters the mesh's triangles by normal (area-weighted,
so the many tiny corner triangles cannot outvote a real face), pairs opposite
faces, and measures their separation.

Two details that were not obvious:

- **Area weighting is what makes it work.** Rounded corner triangles are
  numerous but carry almost no area.
- **Opposite faces are never exactly anti-parallel.** On a reconstructed cube
  their normals splay by about a degree, so differencing the two plane offsets
  measures the gap *extrapolated away from the object* — it read 0.265 units
  against a true 0.252. The separation is therefore measured **through the mesh
  centroid**: the sum of the centroid's distances to the two planes, which is
  well defined however the normals splay. Verified on a synthetic cube rotated
  30 deg / 20 deg: exactly 2.0000 on all three axes.

Falls back to the OBB with an explicit warning when fewer than two face pairs
are found. Applied to the reference only — meaningless for a limb.

**Measured effect, both datasets:**

| | M3 (OBB edge) | M10 (fitted faces) |
|---|---|---|
| cube, leg scene | 2460.1 cm3 (-10.3%) | **2694.2 cm3 (-1.8%)** |
| cube, can scene | 2449.9 cm3 (-10.7%) | **2643.6 cm3 (-3.7%)** |
| limb | 981.6 cm3 | 1075.0 cm3 |
| can | 291.4 cm3 | 314.4 cm3 |
| horizontal edges disagree by | 2.68% / 1.46% | **0.79% / 1.14%** |

**Verdict:** current. Reference error more than halved, and the two horizontal
edges — independent measurements of the same physical 14 cm — now agree to
about 1%.

---

### M11 — squareness as a quality gate · **PARKED**

From an idea of the project owner's: normalise the three edges so they sum to
100%; each should be 33.33% if the cube is cubic. Deviation is **scale-free** —
it needs no ground truth.

Tested as a *selection* rule (pick the axis closest to 33.33%) it gained little:
0.8 pp on OBB edges against 6.0 pp for fixing the measurement. And it cannot
work as a correction, because forcing the shares to sum to 100% makes any
**common-mode** error mathematically invisible — inflate all three edges equally
and the shares do not move. That is exactly the defect M10 fixed, which is why
the measurement had to change rather than the weighting.

Kept as what it is genuinely good for — an **error bar and a warning gate**:

```
squareness = shares 33.04 / 33.29 / 33.67%   (33.33 each if perfectly cubic)
             deviation -0.29 / -0.05 / +0.33 pp
```

Warns when a horizontal deviates by more than 2 pp. Before this, nothing caught
a badly-reconstructed reference before its edge silently set the scale.

**Verdict:** kept as a gate, rejected as a correction.

---

## Current measurement

Full pipeline, both datasets, fresh from Stage 1. `run.py` and `stagerun.py`
agree to 0.0000 cm3 and reruns are bit-identical.

```
       name     method  euler   volume    obb_a    obb_b    obb_c
    box.ply watertight      2 0.011439 0.226249 0.231968 0.235381
leg_cut.ply watertight      2 0.004565 0.531543 0.131647 0.267971

       name  height_cm  width_cm  depth_cm  real_vol_cm3
    box.ply      13.97     14.33     14.54       2694.24
leg_cut.ply      32.83      8.13     16.55       1075.04

can scene:  box 2643.6 cm3 (-3.7%)   can 314.4 cm3
```

Reference error is now **-1.8% / -3.7%** across two independent scenes, against
-10.3% / -10.7% before M10.

### Decomposition of the reference's remaining error

The earlier decomposition in this file attributed the gap to "4.49% vertical
truncation + 10.50% mesh under-filling its box". **That was wrong**, and the
error is instructive: *fill ratio* divides mesh volume by the **OBB** volume,
and the OBB was itself inflated by 2.2% per axis. A metric artifact was being
read as a physical defect.

Measured against the cube's real face-to-face size instead:

```
convex hull of the points   98.9% of the face-to-face box   -> corners barely rounded
mesh volume                 97.8% of the face-to-face box   -> reconstruction loses ~2%
implied corner radius       0.74 cm on a 13.7 cm cube       -> near-physical for taped cardboard
```

So the mesh is a good mesh. What remains after M10 is **-1.8% / -3.7%**, and the
vertical edge still reads 1.3–2.5% short of the horizontals. Two candidates, not
separable from here:

- the cube genuinely is not square. It is handmade cardboard; 13.4 x 13.7 x 13.7
  is entirely plausible for a taped box.
- a small vertical bias specific to the cube's top face.

Against this, the **can shows no vertical anomaly** and its bottom face sits
exactly on the detected floor plane, so the floor extension is not leaving a gap.
That argues for the cube simply not being cubic — which only a caliper settles.

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

## Reconstruction method comparison (2026-08-12)

Run on the same Stage 3 cloud, no pipeline changes. Files in
`temp_output_compare_recon/`.

```
alpha_shape                  15,696 tri  wt=True   chi=  2   1 comp   1075.0 cm3
poisson_d9                   90,319 tri  wt=False  chi=260  207 comp        —
  + pymeshfix               101,238 tri  wt=True   chi=394  207 comp   1016.1 cm3   -5.5%
ball_pivot                   34,087 tri  wt=False  chi=-576  40 comp        —
  + pymeshfix                60,874 tri  wt=True   chi= 66   40 comp    797.7 cm3  -25.8%
```

> **Re-run 2026-08-23 on the current Stage 3 cloud** —
> [`experiments/recon_method_comparison.png`](experiments/recon_method_comparison.png)
> and `experiments.md` under E-recon-method. Same conclusion, but note that ball
> pivoting's post-repair error came out **+30.3%** there against **−25.8%** here.
> The two runs use different clouds; the instability of the sign is itself the
> argument against the method, since both report watertight after repair.

**Alpha shape is the only method that closes without repair**, and the reason is
structural: it works from a 3D Delaunay tetrahedralisation and **never uses
normals**. Poisson integrates a normal field and ball pivoting seeds the ball on
normals, and the fabricated base — a flat cap plus a 4-level swept skirt — has no
well-defined normal. Normal coherence there is 7x worse than on the real surface
(median deviation 0.0021 vs 0.0003).

Ball pivot is the prettiest, and that is measurable: it keeps **every input point
as a vertex, 1:1**, so its surface roughness is *identical* to the cloud's, 0.300
mm to three decimals. But 0.300 mm is the shell-noise floor, and VGGT at 518x518
over a 35 cm limb resolves ~0.7 mm at best — so that texture is noise rendered
faithfully, not skin detail.

Ball pivot also carried **39 shells buried inside the body** (8.20 cm3). Blender's
"Select Interior Faces" does not find them: it selects faces whose edges have
more than 2 face users, and these are fully detached loose parts whose edges have
exactly 2. Applying the pipeline's own largest-component filter plus
`orient_triangles()` removes them and fixes the inconsistent winding — but the
volume only moves 797.7 -> 796.3. The -25.8% was never the buried geometry; it is
PyMeshFix fanning a flat surface across the sole where ball pivot left a hole.

**Verdict:** alpha shape stays the default for both objects. Ball pivot is a
better *picture* and a worse *measurement*, and the thing that makes it
mis-measure is the fabricated base, not the algorithm.

### Blender repair operations, tested and rejected

| operation | result |
|---|---|
| Recalculate Outside | normals already 98.7% correct by a radial test; forcing them changed euler by +-30 and closed nothing. Does fix ball pivot's genuinely inconsistent winding. |
| Merge by Distance | at 0.25x spacing ball pivot went 40 -> **41** components. The pieces are not nearly-touching; the holes are missing surface, not split seams. |
| Select Interior Faces | 0 interior faces on every mesh including the merged scenes. |

All three are repairs for *modelling* damage — flipped winding after mirroring,
doubled verts after joining, buried geometry after booleans. Surface
reconstruction cannot produce any of them.

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

### 4. ~~The measured error is inside the noise floor~~ — WITHDRAWN

This entry claimed the reference's error could not be distinguished from noise,
citing ±16% volume from [`experiments.md`](experiments.md). **That was wrong.**

The ±16% came from a pre-rework measurement on the *can*, before MLS existed,
using radial spread about a fitted cylinder — a metric that folds shape error in
with noise. Re-measured on the current pipeline as local surface thickness:

```
audit_leg   shell 0.29 mm on r = 3.94 cm  ->  volume floor ~1.5%
audit_can   shell 0.15 mm on r = 2.66 cm  ->  volume floor ~1.1%
```

The real floor is **~1-2%**. Errors of a few percent are measurable, and the
reference's error was a real signal worth chasing — which is exactly what M10
found and more than halved. Repeat runs across capture sessions are still worth
having, but not because single-run deltas are unreadable.

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
| volume estimator | solid — exact signed volume, chi = 2 verified, independent cross-check |
| calibration method | correct in form — length, not volume; horizontals only |
| edge measurement | **fixed (M10)** — fitted faces, not a bounding box |
| reference error | **-1.8% / -3.7%** across two scenes, was -10.3% / -10.7% |
| calibration **value** | **unverified** — `14.0` never measured |
| reported accuracy | **not yet possible** — no second reference, no ground truth |
| single-run sensitivity | **~1-2%** (corrected from a stale ±16%) |

The pipeline is in good shape and the remaining reference error is now smaller
than the cube's own build tolerance — a 2 mm error on a handmade cardboard cube
is 4.3% in volume, larger than the -1.8% we are chasing.

That is the headline: **further code cannot improve the number until the cube is
measured.** The next three actions are a caliper, a second reference object of a
different shape, and a bucket of water.
