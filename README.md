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

There is also a browser front end: `./serve.sh` starts the viewer and the compute
service together.

| | |
|---|---|
| Full technical detail | [`pipeline.md`](pipeline.md) |
| Running the web app | [`docs/running_the_web_app.md`](docs/running_the_web_app.md) |
| **Full progress log — findings, derivations, maths** | [`docs/progress.md`](docs/progress.md) |
| Experiment log and verdicts | [`docs/experiments.md`](docs/experiments.md) |
| Stage 6 — under review | [`docs/stage06_experiments.md`](docs/stage06_experiments.md) |
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

What you should see at the end:

```
       name  size_x_cm  size_y_cm  size_z_cm  real_vol_cm3  real_vol_L     method
leg_cut.ply      22.54      19.06      22.81       1081.94        1.08 watertight
    box.ply      19.18      19.55      14.07       2744.00        2.74 watertight
```

Both meshes should report `watertight` with `euler 2`. If either says
`warp+floodfill` or `convex_hull (UNRELIABLE)`, the reconstruction failed and
the number is not a measurement.

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
| `--no-fill` | off | skip bottom cap and floor extend |
| `--no-watertight` | off | publish the Stage 4 recon as the final meshes |
| `--voxel-res` | 150 | voxel cross-check resolution |
| `--seed` | 42 | seeds random, NumPy, PyTorch, Open3D |

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
  **saturated green tape**. The detector separates the band from skin by excess
  green (`2G − R − B`); a beige or khaki band gives a margin of ~24 units, which
  works but is not robust.
- **Even lighting.** Avoid hard shadows under the object — dark pixels have
  numerically meaningless hue and used to be detected as markers.
- **A matte floor if possible.** Gloss produces reflections that VGGT
  reconstructs as real geometry.

---

## Accuracy — read before quoting numbers

- **The reported cube dimensions are partly circular.** Scale is derived as
  `14.0 / mean(two horizontal cube edges)`, so the **mean of the two reported
  horizontal dimensions is exactly 14.00 by construction**. Only their *spread*,
  the *vertical* edge, and the *volume* carry information.
- **`REFERENCE_REAL_SIZE_CM = 14.0` has never been measured.** The cube is
  handmade cardboard; a 2 mm build error is 1.4% linear = **4.3% volume** on
  every result — which is now *larger than the error the pipeline still has*.
- **Scale cannot be validated with one reference.** Under the *parked* Stage 6
  the cube's volume is free to disagree with nominal — **2694 vs 2744 cm³, −1.8%**
  (leg scene) and **2644 vs 2744, −3.7%** (can scene). Under the Stage 6 that
  currently ships it is not free at all: it is the denominator, so it prints
  2744.00 by construction and carries no information. Either way, turning this
  into an accuracy figure needs a **second object of known size**.
- **The noise floor is ~1–2% volume**, measured as local surface thickness on
  the current pipeline (0.29 mm shell on a 3.94 cm radius limb). An earlier
  ±16% figure quoted here came from a pre-rework measurement using a metric that
  mixed shape error into the noise; it was wrong and is withdrawn.
- **No ground truth exists for any measured object.** The can's "325 ml" is its
  *fill* volume, not external displacement — different quantities.

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
