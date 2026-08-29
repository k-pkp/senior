# VGGT Volume Measurement

Measure an object's real-world volume from a handful of phone photos, using an
ArUco-marked cube of known size as the scale reference.

Seven stages: **framing gate → VGGT inference → point cloud → segment & detect →
surface reconstruction → watertight check → real-world volume**, with the marker
cut applied once a person has confirmed where it falls. Stages 0 and 1 run neural
networks; the rest is geometry. A cold run is ~80 s on an RTX 4060.

Two stages stop for a decision. **Stage 0** refuses a capture it cannot frame — a
photo that clips the reference cube corrupts the scale of every number the run
reports, with no visible sign. **Stage 3** detects the cutting plane but does not
apply it, so no volume is ever produced from a cut nobody approved.

**It reads 1.7% against water displacement** on four limb captures — see
[Accuracy](#accuracy--measured-2026-08-27). Two of the five have residuals under
2%, one is unresolved at +19.4%, and a sixth capture is excluded because its
reconstruction failed.

There is also a browser front end: `./serve.sh` starts the viewer and the compute
service together.

| | |
|---|---|
| Full technical detail | [`pipeline.md`](pipeline.md) |
| Running the web app | [`docs/running_the_web_app.md`](docs/running_the_web_app.md) |
| **Full progress log — findings, derivations, maths** | [`docs/progress.md`](docs/progress.md) |
| Experiment log and verdicts | [`docs/experiments.md`](docs/experiments.md) |
| Stage 6 — **resolved: keep M1** | [`docs/stage06_experiments.md`](docs/stage06_experiments.md) |
| Contract and silent-failure sweep | [`docs/repo_review.md`](docs/repo_review.md) |
| **What changed against `main`, and why it is better** | [`docs/updates.md`](docs/updates.md) |
| Every stage and sub-process in one chart | [`docs/full_flowchart.md`](docs/full_flowchart.md) |
| Moving Least Squares, derived in full | [`docs/mls_explained.md`](docs/mls_explained.md) |

---

## Install

```bash
conda create -n senior python=3.11 -y
conda activate senior
pip install -r requirements.txt
```

Backends are auto-detected: **CUDA → MPS → CPU**. Developed on an RTX 4060
(8 GB); a full run is ~80 s there, of which Stage 1 inference is ~19 s and the
detectors in Stage 0 most of the rest.

### Model weights

The default checkpoint is gated and requires accepting Meta's licence:

```bash
hf auth login          # after accepting terms at
                       # huggingface.co/facebook/VGGT-1B-Commercial
```

Without a token the pipeline falls back **loudly** to `facebook/VGGT-1B`, which
is **CC BY-NC-SA 4.0 — not licensed for commercial use.** Set
`VGGT_USE_COMMERCIAL = False` in `pipeline/config.py` to select it deliberately.

> The commercial checkpoint's Acceptable Use Policy forbids unlicensed medical
> or health-professional practice and inferring health data without consent.
> Relevant if you are measuring limbs.

---

## Run

```bash
python run.py -i inputs/est_325 --no-segment-leg   # rigid object, no marker band
python run.py -i inputs/small_leg                  # limb with a marker band
python run.py -i inputs/est_325 --skip_mesh        # point cloud only
```

`--no-segment-leg` is required for anything **without** a coloured marker band,
otherwise marker detection has nothing to find.

### The reference cube — two of them

`REFERENCE_REAL_SIZE_CM` defaults to **10.0**, the 3D-printed cube used from
August 2026. The original fixtures (`small_leg`, `short_leg`, `est_325`) were
shot with a 14 cm handmade cardboard cube and need the override, because
measuring one against the other rescales every volume by `(14/10)³ = 2.74`
with no visible sign:

```bash
REFERENCE_REAL_SIZE_CM=14 python run.py -i inputs/small_leg
```

What you should see at the end:

```
       name  size_x_cm  size_y_cm  size_z_cm  real_vol_cm3  real_vol_L     method
leg_cut.ply      22.54      19.06      22.81       1081.94        1.08 watertight
    box.ply      19.18      19.55      14.07       2744.00        2.74 watertight
```

Both meshes should report `watertight` with `euler 2`. If either says
`warp+floodfill` or `convex_hull (UNRELIABLE)`, the reconstruction failed and
the number is not a measurement.

> That example is `inputs/small_leg`, which uses the **14 cm** cube — run it with
> `REFERENCE_REAL_SIZE_CM=14` or every number is 2.74× too large. **The limb
> figure is also stale:** the current tree reads **1067.57 cm³**, reproducible to
> seven significant figures across cold runs. The difference is upstream of the
> cut — Stage 0 now accepts 6 of 6 frames where the run that produced this table
> accepted 5, which is the verdict redesign recorded in `experiments.md` under
> E-stage0-verdicts. Every `1081.94` in these documents is from the older tree.

Two things in that table are artefacts of **Stage 6 currently being reverted to
`main`'s version** (see [`docs/stage06_experiments.md`](docs/stage06_experiments.md)):
the reference prints exactly `2744.00` because that method derives scale from the
cube's own volume, and the 14 cm cube reads 19.18 × 19.47 × 14.09 because the
dimensions are an axis-aligned box around a tilted cube, which measures its
diagonal. Neither is a measurement error; both are the method reporting itself.
The parked method reads 13.97 × 14.33 × 14.54 and 2694.24 cm³ on the same capture.

---

## Web app

First run only — install the front-end dependencies:

```bash
cd web && npm install
```

Start the site and the compute service together (`serve.sh` uses the `senior`
interpreter directly, so no `conda activate` is needed):

```bash
./serve.sh
```

Open `http://localhost:3111` for the site; it drives the compute service on
port 8000. Ctrl-C stops both. See
[`docs/running_the_web_app.md`](docs/running_the_web_app.md) for letting a phone
on the same wifi reach it.

---

## Output

```
output/
  leg_mesh.ply / .stl        the measured object
  box_mesh.ply / .stl        the ArUco reference
  scene_mesh.ply / .stl      both merged (PLY keeps vertex colours)
  for_debug/
    01_inference/   predictions.npz, raw/
    02_pointcloud/  points.ply
    03_clean/       objects/  debug/
    04_recon/       mesh/
    05_watertight/  mesh/
    06_volume/      volumes.csv
```

Three meshes at the top; everything intermediate under `for_debug/`.

---

## Stage-by-stage runner

`stagerun.py` runs stages individually and caches Stage 1, so parameter sweeps
cost no inference time:

```bash
python3 stagerun.py 1 -i inputs/est_325 --name est_test    # ~35 s, cached after
python3 stagerun.py 2-6 --name est_test                    # seconds
python3 stagerun.py 3 --name est_test                      # re-run one stage
python3 stagerun.py 4-6 --name variant --src est_test \
        --obj-recon-method poisson                         # branch a variant
```

`--src` reads the previous stage from a **different** run, so you can fork an
experiment without recomputing everything before it.

Each stage writes a `summary.txt` with point counts, extents, watertightness and
volumes. Stage 1 also writes `raw/` — every model output as PNG, PLY and JSON,
so you can see what VGGT actually produced rather than infer it.

---

## Key flags

| flag | default | note |
|---|---|---|
| `--conf_thres` | 45.0 | confidence percentile; 45 filters floor and object evenly |
| `--prediction_mode` | `pointmap` | `depth` measured worse on both datasets |
| `--preprocess-mode` | `crop` | `pad` keeps the full frame; `crop` discards ~44% of a 9:16 photo |
| `--recon-method` | `poisson` | also `alpha_shape`, `poisson_omp1`, `box_primitive`. **Does not affect the reference cube** — use `--box-recon-method` for that |
| `--box-recon-method` / `--obj-recon-method` | — | per-object override |
| `--no-segment-leg` | off | **required** for objects with no marker band |
| `--cut-mode` | `auto` | what the bands bound. `upper` measures everything **below** the highest band (one-band capture); `span` measures the segment **between** an upper and a lower band; `auto` cuts a span when two marker planes survive Stage 3's gates. **Pass it explicitly when the run has to match a ground truth measured a particular way** — the table above was measured foot-to-upper-band |
| `--no-fill` | off | skip bottom cap and floor extend |
| `--no-watertight` | off | publish the Stage 4 recon as the final meshes |
| `--voxel-res` | 150 | voxel cross-check resolution |
| `--seed` | 42 | seeds random, NumPy, PyTorch, Open3D |
| `--no-prep-crop` | off | hand VGGT the raw frames. **Fixed 2026-08-27** — `run.py` parsed this flag and ignored it; only `stagerun.py` honoured it |

Tunable constants live in [`pipeline/config.py`](pipeline/config.py), each with
the measurement that set it.

---

## Viewing

```bash
python viewer.py output/leg_mesh.ply
python viewer.py output/scene_mesh.ply --info
```

---

## How to photograph an object

- **6–9 photos**, orbiting the object. More is not better — Stage 1 caps frames.
- **The reference cube must be visible in every shot**, resting on the same
  surface as the object.
- **A coloured band** on the limb where the measurement should stop. Use
  **saturated green tape**, and treat this as the single most load-bearing item
  in this list. The detector separates band from limb by their distance in
  chromaticity, and below `MARKER_MIN_AXIS = 0.05` it stops discriminating:
  every Aug 2026 capture measured 0.021–0.043 with an olive cord on tan skin,
  so the learned-colour detector was refused on all of them and the pipeline
  fell back to its hardcoded colour window. `small_leg`'s khaki measures 0.094.
  A saturated band roughly triples the separation.
- **Keep the reference cube close and unobstructed.** It fills 15–18% of the
  frame width on the Aug 2026 captures against 42% on `small_leg`, which is
  2 100–3 900 surviving points instead of 19 600. On `inputs/blue shirt` the
  cube sits behind the foot in most frames and the reconstruction failed
  outright — see `inputs/blue shirt/UNUSABLE.md`.
- **Even lighting.** Avoid hard shadows under the object — dark pixels have
  numerically meaningless hue and used to be detected as markers.
- **A matte floor if possible.** Gloss produces reflections that VGGT
  reconstructs as real geometry.

---

## Accuracy — measured, 2026-08-27

**The pipeline reads 1.7% mean absolute error against water displacement**, on
four limb captures with a 10 cm 3D-printed reference. This is the project's
first accuracy figure; everything before it was the system checking itself.

| capture | measured | displacement | error |
|---|---|---|---|
| `orange shirt` | 4094 cm³ | 4090 | **+0.1%** |
| `keng` | 2249 cm³ | 2210 | **+1.8%** |
| `black shirt` | 3648 cm³ | 3510 | **+3.9%** |
| `sunshine` | 3093 cm³ | 3130 | **−1.2%** |
| `champ` | 3354 cm³ | 2810 | **+19.4%** — unresolved, below |
| `blue shirt` | — | 3420 | capture unusable, `inputs/blue shirt/UNUSABLE.md` |

**These numbers were measured foot-to-upper-band, so reproducing them now needs
`--cut-mode upper`.** The default became `auto` on 2026-08-29, and auto cuts a
span wherever two marker planes survive Stage 3's gates — which `champ` does, so
under the default it now reports the segment between its two bands (~2377 cm³)
rather than the 3354 cm³ above. That is a different quantity, not a correction.

Verified on a full cold run — Stage 0's detectors, a fresh VGGT pass, every
stage after it — which reproduces the cached-stage-1 numbers to seven
significant figures. The pipeline is deterministic end to end under a fixed
seed, which is what makes an error bar worth quoting.

**Read these before quoting the number.**

- **n = 4, one subject class, one floor, one cord colour.** This is
  repeatability, not generality.
- **±1.7% sits on the ~1–2% surface-noise floor** measured independently as
  local shell thickness. The two agreeing is corroboration, not coincidence,
  but it also means the pipeline cannot currently resolve better than its own
  noise.
- **`REFERENCE_REAL_SIZE_CM = 10.0` is a design dimension, not a caliper
  reading.** Printed parts shrink 0.3–0.8%; at 10 cm a 2 mm error is 2.0%
  linear and **6.1% of volume**, which would dominate the residual above.
  Measuring the cube once is still the cheapest accuracy work available.
- **`champ` is unresolved at +19.4%.** Its cut is the best-aligned of the five
  (2.4° off the limb's own axis) and its cube the second most cubic, and
  2.81 L matches neither segment its two bands bound — below the upper band is
  3379 cm³, between the two is 2377 cm³. Its limb reconstructs about 9% wider
  in circumference than 2.81 L over that length implies, which is the whole
  discrepancy. A tape measure round the calf at a marked height separates
  "reconstructs wide" from "truth is wrong" without more water.
- **The cube's own dimensions are still partly circular.** Stage 6 derives
  scale from the reference's volume, so the cube reports exactly 1000 cm³ every
  run. Only its *edge lengths* carry information; they read 9.96–10.92 cm.
- **The 325 ml can is not ground truth.** That is its *fill* volume; the
  pipeline measures external displacement.

---

## Layout

```
run.py            full pipeline, stages 0-6
stagerun.py       per-stage runner with caching and metrics
viewer.py         PLY/STL viewer
volume.py         standalone volume CLI
serve.sh          starts the web app and the compute service together

pipeline/
  orchestrator.py   stage sequencing, final output publishing
  cli.py            argparse
  config.py         every tunable constant, with its justification
  ghost.py          the whole ghost chain: voxel dedup, normal-aware
                    filter, and MLS surface projection
  multiview.py      multi-view consistency (written, not wired into Stage 2)
  core/             cluster, faces, fill, filters, markers3d, mesh, plane,
                    segmentation, vlm_detect
  stages/           prep, inference, pointcloud, clean, reconstruct,
                    watertight, volume
  utils/            seeding, runlog

workers/          recons_methods_worker.py, meshfix_worker.py
service/          HTTP front end: upload, run, serve artifacts
web/              viewer and review UI (Next.js)
tools/            com_vol.py
inputs/           sample image sets
work/             stagerun and web-job outputs (gitignored)
```
