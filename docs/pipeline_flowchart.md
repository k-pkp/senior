# Pipeline Flowcharts

Eight figures, smallest scope first. Each is self-contained and readable on its
own — the detail lives in the tables beneath each one rather than inside the
node labels.

| # | figure | scope |
|---|---|---|
| 1 | [Pipeline spine](#figure-1--pipeline-spine) | 6 stages, files between them |
| 2 | [Stage 3 · Phase A](#figure-2--stage-3-phase-a--clustering) | clustering |
| 3 | [Marker detection](#figure-3--marker-detection) | colour rule and plane fit |
| 4 | [Stage 3 · Phase B+C](#figure-4--stage-3-phase-b--c--level-cut-close) | levelling, cut, close |
| 5 | [Stage 4](#figure-5--stage-4-alpha-selection) | α selection |
| 6 | [Stage 6](#figure-6--stage-6-volume-and-scale) | volume and scale |
| 7 | [Full system](#figure-7--full-system-with-website) | pipeline + website |
| 8 | [Review screen](#figure-8--review-screen-coordinate-flow) | coordinate handling |

Then a [per-stage I/O reference](#per-stage-io-reference).

Everything runs **locally** — VGGT on one GPU, the web app on `localhost:3111`.
Nothing is sent to a cloud service.

Conventions: blue = a file on disk · amber dashed = **not built yet**.

---

## Figure 1 — pipeline spine

```mermaid
flowchart LR
    IN[/"photos<br/>6-9 frames"/]:::file
    S1["STAGE 1<br/>VGGT inference"]
    F1[/"predictions.npz"/]:::file
    S2["STAGE 2<br/>point cloud"]
    F2[/"points.ply<br/>~541k pts"/]:::file
    S3["STAGE 3<br/>segment · cut · close"]
    F3[/"leg_cut · leg_no_cut<br/>box · cutting_line"/]:::file
    S4["STAGE 4<br/>alpha shape"]
    F4[/"*_recon.ply"/]:::file
    S5["STAGE 5<br/>watertight"]
    F5[/"*.ply + *.stl"/]:::file
    S6["STAGE 6<br/>volume"]
    F6[/"volumes.csv"/]:::file

    IN --> S1 --> F1 --> S2 --> F2 --> S3 --> F3 --> S4 --> F4 --> S5 --> F5 --> S6 --> F6

    classDef file fill:#eef4fb,stroke:#4a72a8,color:#16324f
```

| stage | time | in | out |
|---|---|---|---|
| 1 inference | ~22 s | image folder | `predictions.npz` |
| 2 point cloud | ~2 s | `predictions.npz` | `points.ply` |
| 3 clean | ~14–20 s | `points.ply` | 4 PLY + 2 JSON |
| 4 reconstruct | ~5–20 s | `objects/*.ply` | `*_recon.ply` |
| 5 watertight | ~1 s | `*_recon.ply` | `*.ply` + `.stl` |
| 6 volume | ~1 s | watertight meshes | `volumes.csv` |

---

## Figure 2 — Stage 3 Phase A · clustering

Runs in **original VGGT space, before levelling**. `main` levelled first, which
made clustering depend on a RANSAC fit that could fail.

```mermaid
flowchart TD
    IN[/"points.ply"/]:::file
    A1["Statistical outlier removal"]
    A2["Remove dominant plane<br/>as a BAND"]
    A3["DBSCAN<br/>adaptive epsilon"]
    A4["Score clusters<br/>median-normalised"]
    A5["Identify BOX<br/>by cubeness"]
    A6["Marker detection<br/>see figure 3"]
    A7["Ghost filter<br/>per cluster"]
    A8["MLS projection"]
    OUT[/"BOX cloud + OBJ cloud<br/>+ marker planes"/]:::file

    IN --> A1 --> A2 --> A3 --> A4 --> A5 --> A6 --> A7 --> A8 --> OUT

    classDef file fill:#eef4fb,stroke:#4a72a8,color:#16324f
```

| step | what it does | the detail that matters |
|---|---|---|
| A2 **band removal** | drop `\|n·p + d\| ≤ 2·τ` | **not** just the RANSAC inliers. VGGT ghosts the floor, so it is a sandwich ~2 thresholds thick; removing the middle leaves both skins — 41,749 points of floor slab that DBSCAN then welds to the limb |
| A3 DBSCAN | ε from mean NN distance | objects rest on the floor, so without A2 every object links through the ground |
| A4 scoring | `0.4·N/med(N) + 0.3·ρ/med(ρ) + 0.3·(1 − D/max D)` | scale-free. `main` used magic constants that encoded an assumed scale |
| A5 box ID | cubeness = `min extent / max extent` | plus an ArUco-likeness score |
| A7 ghost filter | voxel dedup, then reject `δ = 1 − \|n·n̄\| > 0.3` | **per cluster** — box and limb have different densities and must not share a voxel size |
| A8 MLS | project onto a local degree-2 height field | collapses both ghost sheets onto one surface. Radius must exceed ghost separation |

---

## Figure 3 — marker detection

The marker band defines where the measurement stops.

```mermaid
flowchart TD
    IN[/"dense limb cloud<br/>with RGB"/]:::file
    M1["RGB to HSV"]
    M2["Brightness floor<br/>V > 15%"]
    M3["Hue window<br/>70-180 deg, S > 25%"]
    M4["Excess green<br/>2G-R-B > 10"]
    M5["Union of masks"]
    M6["DBSCAN, eps 0.03"]
    M7["Reject clusters<br/>under 40 points"]
    M8["Fit plane by SVD"]
    OUT[/"marker planes<br/>centroid + normal + npts"/]:::file

    IN --> M1 --> M2
    M2 --> M3 --> M5
    M2 --> M4 --> M5
    M5 --> M6 --> M7 --> M8 --> OUT

    classDef file fill:#eef4fb,stroke:#4a72a8,color:#16324f
```

**Why the brightness floor is first.** Hue is a ratio to the channel maximum, so
as `V → 0` it becomes numerically arbitrary. `main` had no floor and an
upper-open window (`h > 60`, accepting 83% of the hue wheel), which classified
this as markers:

| what | RGB | ExG | hue | V | old rule |
|---|---|---|---|---|---|
| shadow | (8, 6, 8) | −4 | 300° | **3.1%** | passes |
| skin | (139, 87, 89) | −54 | **358°** | 54.5% | passes |
| **the band** | (60, 52, 30) | **+14** | 44° | 23.5% | passes via ExG |

The shadow cluster reached 3,349 points against 197 for the real band and won
the cut.

**M4 is what actually finds the marker, not M3.** The band is khaki, hue 44° —
it fails every green hue test. Excess green separates it from skin (+14 vs −54).
Raising the ExG threshold to 15 silently lost the only real marker in the
dataset.

A height gate (reject below 20% of object height) also applies, but **not here** —
it runs in Phase C, because detection happens before levelling and "height" has
no meaning yet. See figure 4.

---

## Figure 4 — Stage 3 Phase B + C · level, cut, close

```mermaid
flowchart TD
    IN[/"clean clouds<br/>+ marker planes"/]:::file
    B1["RANSAC floor plane"]
    B2["Rodrigues rotation<br/>floor normal to +Z"]
    B3{"Upside down?"}
    B4["R_total = flip · R"]
    B5["Rotate clouds AND planes"]
    C1["Floor cut"]
    C2["Reject planes below<br/>20% of height"]
    C3["Build review cloud<br/>always floor-closed"]
    NC[/"leg_no_cut.ply"/]:::file
    C4{"marker count"}
    C5["no cut"]
    C6["keep BELOW"]
    C7["keep BETWEEN"]
    C8["Cap cut faces"]
    C9["Close to floor"]
    OUT[/"leg_cut.ply"/]:::file

    IN --> B1 --> B2 --> B3
    B3 -->|yes| B4 --> B5
    B3 -->|no| B5
    B5 --> C1 --> C2 --> C3 --> NC
    C3 --> C4
    C4 -->|0| C5 --> C9
    C4 -->|1| C6 --> C8
    C4 -->|2| C7 --> C8
    C8 --> C9 --> OUT

    classDef file fill:#eef4fb,stroke:#4a72a8,color:#16324f
```

> **Note on the diagram:** for 2 markers, `C9` is a pass-through — the bottom is
> already the lower cut face, so nothing is added. See the table.

**`R_total` — a bug fixed.** An upside-down scene triggers a 180° flip *after*
the first rotation. `main` applied the flip to the cloud but kept the original
`R` for the marker planes, so the cut plane landed in the wrong frame and cut
99 mm off the wrong end.

**Why C2 lives here and not in detection.** Detection runs in original VGGT
space, where the vertical axis is whatever the camera gave — on `small_leg` the
limb's long axis is **Y**, not Z. A Z-based height test there measures sideways.
Only after `R_total` does "height" mean height.

**The cut rule.** Each normal is first flipped to point along world up, so the
detected normal's sign cannot change the outcome. Then:

| markers | keep | bottom of the kept region | who closes it |
|---|---|---|---|
| 0 | everything | the floor | floor extend + bottom cap |
| 1 | `d ≤ 0` (below) | the floor | floor extend + bottom cap |
| 2 | `(d₁ ≤ 0) XOR (d₂ ≤ 0)` | the **lower cut face** | `cap_points_on_plane` |

The XOR needs no ordering of the planes: a point between them is below the upper
and above the lower, so exactly one test is true.

`main` used a **centroid-side** rule instead — keep the side the cloud's centroid
is on. That inverts the whole selection when a plane is dragged past the
centroid, and only means "keep between" when the centroid already sits between
the two planes.

`main` also **filled before cutting**, which fabricated a floor skirt inside
2-marker segments. `leg_no_cut.ply` is built from its own copy so it stays
floor-closed for review, while the cut runs on the unfilled cloud.

---

## Figure 5 — Stage 4 α selection

```mermaid
flowchart TD
    IN[/"objects/*.ply"/]:::file
    R1["Downsample only above<br/>90,000 points"]
    R2["Delaunay tetrahedralisation<br/>ONCE"]
    R3["Next alpha<br/>8x .. 90x"]
    R4["Keep largest component"]
    R5{"watertight<br/>AND euler = 2 ?"}
    R6["Select this alpha"]
    R7["Cleanup<br/>revert if it reopens"]
    OUT[/"*_recon.ply"/]:::file

    IN --> R1 --> R2 --> R3 --> R4 --> R5
    R5 -->|no| R3
    R5 -->|yes| R6 --> R7 --> OUT

    classDef file fill:#eef4fb,stroke:#4a72a8,color:#16324f
```

**Why Euler and not watertightness.** A surface riddled with tunnels is still
closed, and the signed-volume integral faithfully subtracts those tunnels — the
mesh looks like a perfect cube from outside while reporting far too little
volume. `χ = V − E + F`, and `χ = 2 − 2g` for genus `g`, so each tunnel costs 2.
Measured on the reference cube:

| α | watertight | χ | volume |
|---|---|---|---|
| 30× | yes | **−1** | 1898 cm³ |
| 40× | yes | **2** | 2467 cm³ |

Selecting on watertightness alone would have under-read by 31%.

**R4 before R5.** Stray fragments each contribute their own `χ = 2`, so debris
can push a good mesh to `χ = 6` and get it rejected.

**R2 once.** Alpha shape is computed *from* a Delaunay tetrahedralisation, and
3-D Delaunay on surface-sampled points is pathological — a 42× slowdown was
measured. Building it once and reusing it across all 12 α is 4.3× faster.

**R7 reverts itself.** Cleanup calls `remove_non_manifold_edges()`, which can
tear a closed surface open — undoing the very property α was selected for. It
snapshots first and restores if that happens.

---

## Figure 6 — Stage 6 volume and scale

```mermaid
flowchart TD
    IN[/"watertight meshes"/]:::file
    V1["Oriented bounding box"]
    V2["Order axes by orientation<br/>index 0 = most aligned with up"]
    V3["Exact signed volume<br/>divergence theorem"]
    V4["Voxel occupancy<br/>independent cross-check"]
    V5["Reference edges from FITTED FACES<br/>not a bounding box"]
    V6["linear_scale = 14.0 / edge<br/>k = scale cubed"]
    V7["Report reference error<br/>unforced"]
    OUT[/"volumes.csv"/]:::file

    IN --> V1 --> V2 --> V3 --> V4 --> V5 --> V6 --> V7 --> OUT

    classDef file fill:#eef4fb,stroke:#4a72a8,color:#16324f
```

**The one thing to understand about this pipeline:** scale comes from a measured
**length**, not a volume ratio — and that length is measured from the cube's own
fitted faces, not from a bounding box. An OBB has to guess the orientation, and
on this reference it guessed 1.3 degrees wrong, inflating every edge by 2.2%.

```
ref edge     = mean of the cube's two HORIZONTAL FITTED-FACE edges
linear_scale = 14.0 cm / ref edge
k            = linear_scale ** 3
V_real       = V_mesh * k
```

`main` used `k = V_real / V_mesh`, `ℓ = k^(1/3)`. That treats `V_mesh^(1/3)` as
the edge length, which only holds for a perfect cube, and the cube root
compounds any deviation three times — at 2.2% off cubic it under-read the edge
by 3.1% and inflated every volume ~10%. It also **forces the reference's own
error to zero**, so the system could never report its own accuracy.

The vertical edge is excluded because the cube's underside rests on the floor
and never reconstructs.

**V1/V2.** An AABB around a tilted object reports its *diagonal* — the can
measured 6.09 cm wide by AABB against 5.75 cm by OBB. Axes are ordered by
orientation (`argmax |R[:,i] · ẑ|`), not magnitude, so index 0 is reliably the
axis the floor truncates.

**V4.** Voxel occupancy over-reads by a few percent (boundary voxels counted
whole) and converges downward onto exact. Expect **+1% to +8%**; a result *below*
exact means a self-intersecting or inverted surface.

Current run:

```
horizontal edges  0.2264, 0.2325 units — disagree by 2.68%
linear_scale      14.0 / 0.229427 = 61.02 cm/unit
reference reads   2694 cm3 vs 2744 nominal  ->  -1.8%   (leg scene)
                  2644 cm3 vs 2744 nominal  ->  -3.7%   (can scene)
```

That -1.8% / -3.7% is the honest error bar on every other number — and it is now
smaller than the cube's own build tolerance, since a 2 mm error on a handmade
cardboard cube is 4.3% in volume.

---

## Figure 7 — full system with website

The pipeline is unchanged. **The amber dashed path is not built** — the browser
cannot currently trigger a run.

```mermaid
flowchart TD
    USER["User with photos"]
    UP["Upload screen"]
    API["FastAPI service"]:::todo
    PROC["Processing screen"]:::todo
    CLI["stagerun.py"]
    PIPE["6-stage pipeline<br/>figure 1"]
    FIX[/"web/public/samples/<br/>meshes · volumes.csv · cutting_line"/]:::file
    SAM["Samples screen"]
    REV["Review screen<br/>figure 8"]
    RES["Result screen"]

    USER --> UP
    UP -.->|"photos"| API
    API -.-> PIPE
    API -.->|"stage events"| PROC
    PROC -.-> REV

    CLI --> PIPE --> FIX
    FIX --> SAM --> RES
    FIX --> REV --> RES

    classDef file fill:#eef4fb,stroke:#4a72a8,color:#16324f
    classDef todo fill:#fdf3e3,stroke:#c98a2e,color:#5a3a06,stroke-dasharray: 5 4
```

Today the Upload screen probes `NEXT_PUBLIC_API_URL`, finds nothing, disables
**Start processing** and says *"The compute service is not reachable right now."*
Fixtures are produced by `stagerun.py` and copied into `web/public/samples/` by
hand.

The missing piece is small: accept a photo folder, shell out to
`stagerun.py 1-6`, stream stage progress, serve the result directory. The
front end already has the Processing screen built.

---

## Figure 8 — Review screen coordinate flow

The subtle part of the web layer. The pipeline is **Z-up**; three.js is
**Y-up**; the loader also recentres the cloud. Anything positioned in the same
space — the cut planes especially — must receive the identical transform.

```mermaid
flowchart TD
    P1[/"leg_no_cut.ply"/]:::file
    P2[/"cutting_line_levelled.json"/]:::file
    T1["rotateX -90<br/>Z-up to Y-up"]
    T2["scale to cm"]
    T3["recentre<br/>store sceneOffset"]
    T4["pointToScene / dirToScene<br/>same transform for planes"]
    Y["Yellow plane<br/>detected marker, FIXED"]
    G["Green disc<br/>proposed cut, EDITABLE"]
    SL["Sliders<br/>height · tilt · direction"]
    SP["splitByPlanes<br/>mirrors apply_marker_cut"]
    OUT["Blue kept · grey discarded"]

    P1 --> T1 --> T2 --> T3 --> SP
    T3 --> T4
    P2 --> T4
    T4 --> Y
    T4 --> G --> SL --> SP --> OUT

    classDef file fill:#eef4fb,stroke:#4a72a8,color:#16324f
```

**Why `cutting_line_levelled.json` and not `cutting_line.json`.** The two do not
share a frame — `cutting_line.json` is written pre-levelling, `leg_no_cut.ply`
post-levelling. Drawing them together needs the levelled file.

**Slider angles fold to the upper hemisphere first.** A plane is unchanged by
negating its *whole* normal, but **not** by flipping `z` alone. Reading tilt as
`arccos|n_z|` while keeping the original `x, y` did exactly that, and the plane
visibly reversed the moment any slider moved. Round trip now gives
`|n · n₀| = 1.0000` against 0.7963 before.

**`splitByPlanes` mirrors `apply_marker_cut` exactly**, so the browser preview
and the pipeline agree on which side is kept.

> **Caveat.** The kept/discarded *counts* are not comparable. The browser cuts
> `leg_no_cut.ply` (floor-closed); the pipeline cuts the raw cloud and closes
> afterwards. Both are correct for what they do.

---

## Per-stage I/O reference

### Stage 1 — inference · `pipeline/stages/inference.py`

**In:** `inputs/<name>/*.jpg|png|heic`, 6–9 frames orbiting the subject, cube
visible in every shot.
**Out:** `01_inference/predictions.npz`.

| sub-process | in | out | notes |
|---|---|---|---|
| frame load | image folder | frame list | capped by `--max_frames` |
| preprocess | frames | `(S,3,518,518)` | `crop` discards ~44% of a 9:16 photo; `pad` keeps it |
| checkpoint | HF hub | state dict | prefers the commercial repo (gated, needs `HF_TOKEN`); fallback is loud |
| forward pass | tensor | pointmap, depth, conf, camera | DINOv2 ViT-L/14, alternating frame/global attention |

`predictions.npz`: `world_points (S,H,W,3)`, `world_points_conf (S,H,W)`,
`depth`, `depth_conf`, `images`, `extrinsic`, `intrinsic`.

---

### Stage 2 — point cloud · `pipeline/stages/pointcloud.py`

**In:** `predictions.npz` · **Out:** `02_pointcloud/points.ply`

| sub-process | maths | why |
|---|---|---|
| adaptive threshold | `τ = percentile(conf, p)`, `p` = `--conf_thres` (45) | VGGT's confidence scale shifts with texture and lighting |
| mask | `conf ≥ τ AND conf > 1e-5` | second term drops exact-zero padding |
| colour attach | reshape `images` → `(S,H,W,3)` | handles both channel orders |
| multi-view consistency | reproject, require `min_views` agreements | **disabled** — removed 41% of points with zero change in shell thickness, because the ghost is the same error in every view |

---

### Stage 3 — clean · `pipeline/stages/clean.py`

**In:** `points.ply` · **Out:** 4 PLY + 2 JSON

| file | what |
|---|---|
| `objects/leg_cut.ply` | the measured cloud, cut and closed |
| `objects/leg_no_cut.ply` | complete uncut cloud — the review artifact |
| `objects/box.ply` | reference cube |
| `objects/merged.ply` | both, for inspection |
| `debug/cutting_line.json` | marker planes, **original VGGT space** |
| `debug/cutting_line_levelled.json` | same planes after `R_total` — what the web reads |

Phases: A → figure 2, marker sub-flow → figure 3, B and C → figure 4.

---

### Stage 4 — reconstruction · `workers/recons_methods_worker.py`

**In:** `03_clean/objects/*.ply` · **Out:** `04_recon/mesh/*_recon.ply`

Methods: `alpha_shape` (default), `poisson`, `poisson_omp1`, `box_primitive`.
`ball_pivot` was removed — it never produced a usable mesh.

α multipliers: `[8, 10, 12, 14, 16, 20, 25, 30, 40, 55, 70, 90]` × mean NN
distance. Detail → figure 5.

---

### Stage 5 — watertight · `pipeline/stages/watertight.py`

**In:** `*_recon.ply` · **Out:** `05_watertight/mesh/*.ply` + `.stl`, plus
`scene_colour.ply`

Skips repair when the mesh is already closed **and says so**. Repair firing is a
signal that Stage 4 struggled and the result deserves suspicion.

---

### Stage 6 — volume · `pipeline/stages/volume.py`

**In:** watertight meshes · **Out:** `06_volume/volumes.csv` + `summary.txt`

| sub-process | maths |
|---|---|
| oriented bbox | OBB, not AABB |
| axis ordering | index 0 = `argmax_i \|R[:,i] · ẑ\|` |
| exact volume | `V = (1/6) Σ (v₀ × v₁) · v₂` |
| voxel cross-check | independent occupancy; expect +1% to +8% |
| scale | `linear_scale = 14.0 / mean(two horizontal ref edges)` |
| conversion | `k = linear_scale³`, `V_real = V_mesh · k` |
| integrity | warn if χ ≠ 2, not watertight, or convex-hull fallback fires |

`volumes.csv` columns: `name`, `is_ref`, `volume`, `method`, `obb_a`, `obb_b`,
`obb_c`, `voxel`, `euler`, `real_vol_cm3`, `real_vol_L`, `height_cm`,
`width_cm`, `depth_cm`.
