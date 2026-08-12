# Cubit — volume from photographs

Measure an object's real-world volume from a handful of phone photos, using an
ArUco-marked cube of known size as the scale reference.

Six stages: **VGGT inference → point cloud → segment & cut → surface
reconstruction → watertight check → real-world volume.** Only the first runs a
neural network; the rest is geometry.

Everything runs **locally** — one GPU for the pipeline, a Next.js app on
`localhost` for the viewer. Nothing is sent to a cloud service.

| document | what |
|---|---|
| [`pipeline.md`](pipeline.md) | full technical detail, stage by stage |
| [`docs/pipeline_flowchart.md`](docs/pipeline_flowchart.md) | eight dataflow diagrams, pipeline and web |
| [`docs/update.md`](docs/update.md) | every change vs the original, with the maths |
| [`docs/experiments.md`](docs/experiments.md) | experiment log, results and verdicts |
| [`docs/stage06_experiments.md`](docs/stage06_experiments.md) | volume + calibration method history |

---

## Contents

- [Install](#install) · [Model weights](#model-weights)
- [Quick start](#quick-start)
- [Running the pipeline](#running-the-pipeline)
- [Running the website](#running-the-website)
- [Stage-by-stage runner](#stage-by-stage-runner)
- [Output layout](#output-layout)
- [Key flags](#key-flags)
- [How to photograph an object](#how-to-photograph-an-object)
- [Accuracy — read before quoting numbers](#accuracy--read-before-quoting-numbers)
- [Repository layout](#repository-layout)

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Backends are auto-detected: **CUDA → MPS → CPU**. Developed on an RTX 4060
(8 GB); a full run is ~40 s there.

For the website you also need **Node 18+**:

```bash
cd web && npm install
```

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

## Quick start

```bash
# 1. run the pipeline on a bundled sample
python run.py -i inputs/small_leg

# 2. look at the result
python viewer.py output/leg_mesh.ply

# 3. or browse it in the web app
cd web && npm run build && npm start      # http://localhost:3000
```

---

## Running the pipeline

```bash
python run.py -i inputs/est_325 --no-segment-leg   # rigid object, no marker band
python run.py -i inputs/small_leg                  # limb with a marker band
python run.py -i inputs/est_325 --skip_mesh        # point cloud only
```

`--no-segment-leg` is required for anything **without** a coloured marker band,
otherwise marker detection has nothing to find.

What you should see at the end:

```
       name  height_cm  width_cm  depth_cm  real_vol_cm3     method
    box.ply      13.37     13.81     14.19       2345.55 watertight
leg_cut.ply      31.03      7.54     15.78        851.28 watertight
```

Both meshes should report `watertight` with `euler 2`. If either says
`warp+floodfill` or `convex_hull (UNRELIABLE)`, the reconstruction failed and
the number is not a measurement — see
[`docs/stage06_experiments.md`](docs/stage06_experiments.md).

---

## Running the website

The web app shows the reconstruction in 3D, lets you adjust the cutting plane
interactively, and reports the volume with its reference check.

```bash
cd web
npm install        # first time only
npm run build
npm start          # http://localhost:3000
```

For development with hot reload:

```bash
cd web && npm run dev
```

To serve on a different port (useful under WSL):

```bash
npm start -- -p 3111
```

### Screens

| screen | needs a server? | what it does |
|---|---|---|
| **Samples** | no | precomputed runs shipped in `web/public/samples/` |
| **Upload** | **yes** | disabled until a compute service is reachable |
| **Review** | no | interactive cut adjustment on the point cloud |
| **Result** | no | volume, oriented dimensions, reference check |
| **How it works** | no | method explanation |

Direct links:

```
http://localhost:3000/?screen=review&dataset=small_leg
http://localhost:3000/?screen=result&dataset=small_leg
```

### Publishing your own run to the website

There is **no compute backend yet** — the Upload screen probes
`NEXT_PUBLIC_API_URL`, finds nothing, and says so honestly. To view your own run
in the browser, copy its outputs into a sample folder:

```bash
S=work/<run-name>
D=web/public/samples/<dataset-id>
mkdir -p $D
cp $S/05_watertight/mesh/leg_cut.ply        $D/leg_mesh.ply
cp $S/05_watertight/mesh/leg_cut.stl        $D/leg_mesh.stl
cp $S/05_watertight/mesh/box.ply            $D/box_mesh.ply
cp $S/05_watertight/mesh/scene_colour.ply   $D/scene_mesh.ply
cp $S/03_clean/objects/leg_no_cut.ply       $D/leg_no_cut.ply
cp $S/06_volume/volumes.csv                 $D/volumes.csv
cp $S/03_clean/debug/cutting_line_levelled.json $D/cutting_line.json
```

Then register it in `web/src/lib/data.ts` under `SAMPLES`, and rebuild.

> Use `cutting_line_levelled.json`, **not** `cutting_line.json`. The two do not
> share a coordinate frame — one is written before levelling, the other after.

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

## Output layout

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

`stagerun.py` writes to `work/<name>/` with the same per-stage structure.

---

## Key flags

| flag | default | note |
|---|---|---|
| `--conf_thres` | 45.0 | confidence percentile; 45 filters floor and object evenly |
| `--prediction_mode` | `pointmap` | `depth` measured worse on both datasets |
| `--preprocess-mode` | `crop` | `pad` keeps the full frame; `crop` discards ~44% of a 9:16 photo |
| `--recon-method` | `alpha_shape` | also `poisson`, `poisson_omp1`, `box_primitive` |
| `--box-recon-method` / `--obj-recon-method` | — | per-object override |
| `--no-segment-leg` | off | **required** for objects with no marker band |
| `--no-fill` | off | skip bottom cap and floor extend |
| `--no-watertight` | off | publish the Stage 4 recon as the final meshes |
| `--voxel-res` | 150 | voxel cross-check resolution |
| `--seed` | 42 | seeds random, NumPy, PyTorch, Open3D |

Tunable constants live in [`pipeline/config.py`](pipeline/config.py), each with
the measurement that set it.

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

Read [`docs/stage06_experiments.md`](docs/stage06_experiments.md) in full before
reporting any figure. In short:

- **The reported cube dimensions are partly circular.** Scale is derived as
  `14.0 / mean(two horizontal cube edges)`, so the **mean of the two reported
  horizontal dimensions is exactly 14.00 by construction**. Only their *spread*,
  the *vertical* edge, and the *volume* carry information.
- **`REFERENCE_REAL_SIZE_CM = 14.0` has never been measured.** The cube is
  handmade cardboard; a 2 mm build error is 1.4% linear = **4.3% volume** on
  every result.
- **Scale cannot be validated with one reference.** The cube's *volume* is now
  free to disagree with nominal — currently **2346 vs 2744 cm³, −14.5%** — but
  turning that into an accuracy figure needs a **second object of known size**.
- **The noise floor is ±16% volume.** Surface localisation error is ~2 mm, and
  volume goes as r². Most parameter tuning lives below that threshold, so
  single-run deltas smaller than ~16% cannot be judged.
- **No ground truth exists for any measured object.** The can's "325 ml" is its
  *fill* volume, not external displacement — different quantities.

---

## Repository layout

```
run.py            full pipeline, stages 1-6
stagerun.py       per-stage runner with caching and metrics
viewer.py         PLY/STL viewer
volume.py         standalone volume CLI

pipeline/
  orchestrator.py   stage sequencing, final output publishing
  cli.py            argparse
  config.py         every tunable constant, with its justification
  ghost.py          voxel dedup + normal-aware ghost filter
  mls.py            moving-least-squares surface projection
  multiview.py      multi-view consistency (disabled — documented failure)
  detection.py      Grounding DINO + SAM seed detection
  core/             plane, cluster, fill, filters, mesh, segmentation
  stages/           inference, pointcloud, clean, reconstruct, watertight, volume

workers/          recons_methods_worker.py, meshfix_worker.py
tools/            com_vol.py
web/              Next.js 15 / React 19 / three.js front end
docs/             flowcharts, change log, experiment logs
inputs/           sample image sets
work/             stagerun outputs (gitignored)
```
