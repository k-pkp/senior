# Engineering Update — every change against `origin/main`

Baseline: commit `bab2bbc` ("Merge pull request #7 from senior-ict/champ-branch"),
the current tip of `main` on GitHub.

This document explains **what changed, why, and the mathematics of each part**.
It is written to be read without the diff open. Where a constant has a value,
the measurement that produced it is given — no number here was chosen by feel.

The previous engineering update is preserved at
[`update-legacy-2026-08-07.md`](update-legacy-2026-08-07.md). It describes an
architecture that no longer exists (7 stages, Poisson default, fitted box
primitive) and should be read only as history.

Related: [`../pipeline.md`](../pipeline.md) for current behaviour,
[`experiments.md`](experiments.md) for the experiment log,
[`pipeline_flowchart.md`](pipeline_flowchart.md) for the dataflow diagrams.

---

## 0. Change inventory

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

## 1. Stage 1 — VGGT inference

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

## 2. Stage 2 — point cloud export

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

## 3. Stage 3 — segment, detect, cut, close

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
# ... in the flip branch:
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

## 4. Stage 4 — reconstruction

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

## 5. Stage 5 — watertight repair

PyMeshFix repair, with one behavioural change: it now reports **honestly**
when repair was skipped because the mesh was already closed.

```
box: already watertight — repair skipped (5,645 verts, 11,286 faces)
```

That line is a useful signal in the other direction too: if repair *fires*,
Stage 4 struggled, and the result deserves suspicion.

---

## 6. Stage 6 — real-world volume

Renumbered from Stage 7.

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

## 7. Configuration — every new constant

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

## 8. Tooling added

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

## 9. End-to-end result

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

## 10. Known open issues

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
