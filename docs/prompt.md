# Website / web app design brief

A brief for building the web interface to this project. Written to be handed to
a designer, a developer, or an AI coding assistant. Everything factual is drawn
from `pipeline.md` and `experiments.md` — **do not invent numbers.**

This is **not** a static showcase. The user uploads photos, and partway through
processing they interactively adjust where the limb is cut before the volume is
computed. That single requirement drives most of the architecture below.

---

## 1. What the project does

Measures the real-world volume of a physical object from a handful of ordinary
phone photos.

You photograph the object next to an ArUco-marked cube of known size (14 cm). A
vision transformer (VGGT-1B) reconstructs the scene in 3D from those images
alone — no depth sensor, no turntable, no calibration rig. The pipeline
separates the object from the reference cube, closes the surface into a solid
mesh, and uses the cube to convert model units into centimetres.

Target application is limb volume measurement — tracking swelling, e.g.
lymphedema monitoring or prosthetic fitting. A coloured band is worn on the limb
to mark where the measurement should stop.

**One-liner:** *photos in, a solid 3D mesh and a volume in cm³ out.*

---

## 2. The pipeline in detail

Six stages. Only Stage 1 runs a neural network; everything after is geometry.
Timings are for 8 photos on an RTX 4060.

| # | stage | input | output | time | user interaction |
|---|---|---|---|---|---|
| 0 | upload | folder of photos (6–9) | image set | — | **YES — upload** |
| 1 | VGGT inference | images | `predictions.npz` (9 arrays) | ~20–25 s | none |
| 2 | point cloud export | predictions | `points.ply` ~1.2 M pts | ~2 s | none |
| 3a | segment & detect | `points.ply` | `leg_no_cut.ply`, `box.ply`, `cutting_line.json` | ~5 s | none |
| — | **review the cut** | leg cloud + planes | adjusted planes | — | **YES — the core interaction** |
| 3b | cut & close | leg + adjusted planes | `objects/leg_cut.ply` | ~2 s | none |
| 4 | surface reconstruction | object clouds | `mesh/*_recon.ply` | ~10 s | optional method choice |
| 5 | watertight check | recon meshes | `mesh/*.ply` | <1 s | none |
| 6 | volume | closed meshes | `volumes.csv` | ~2 s | none |

Total ≈ 45 s of compute, plus however long the user spends on the review step.

### What each stage does

**Stage 1 — VGGT inference.** One forward pass produces a 3D point and a
confidence value for every pixel of every image, plus camera poses. This is the
only GPU-heavy step and the only non-deterministic-feeling one (it is seeded and
reproducible, but it is where all the uncertainty originates).

**Stage 2 — point cloud export.** Filters points by confidence (keeps ~55%) and
removes statistical outliers. Output is a single coloured cloud of the whole
scene — floor, limb, cube, background.

**Stage 3a — segment and detect.** Removes the dominant plane (the floor —
without this, everything is connected through the ground and cannot be
separated), clusters what remains into objects, and identifies which cluster is
the reference cube (by how cube-like and how black-and-white it is) versus the
limb. Then detects the coloured marker band on the limb using HSV + Excess Green
colour thresholds, clusters those marker points, and fits a plane through each
cluster by SVD. Finally levels the scene so Z is up.

**→ Review step.** See section 3.

**Stage 3b — cut and close.** Cuts the limb at the marker plane(s), caps the
exposed cross-section with a flat disc lying in the cut plane, extends the base
down to the detected floor (VGGT never reconstructs the shadowed contact
region), and caps the bottom.

**Stage 4 — surface reconstruction.** Turns the point cloud into a triangle
mesh using alpha shapes, choosing the tightest alpha that still yields a closed
surface.

**Stage 5 — watertight check.** Verifies the mesh is closed; repairs it if not.
With the default reconstructor it is always already closed, so this is
insurance rather than a processing step.

**Stage 6 — volume.** Computes exact volume by integrating the closed surface,
then converts to cm³ using the reference cube's measured edge length.

---

## 3. The review step — the core interaction

**This is the feature that makes it an app rather than a demo.**

After Stage 3a the user sees the limb point cloud (`leg_no_cut.ply`) in 3D with
the detected cutting plane(s) drawn on it, and can adjust before committing.

### What the user must be able to do

1. **See the uncut limb** as an interactive 3D point cloud — orbit, zoom, pan.
2. **See each detected cutting plane** rendered as a semi-transparent disc or
   grid intersecting the limb, with the discarded side visually distinguished
   (greyed, hidden, or tinted).
3. **Adjust the plane's height** — slide it up and down the limb's vertical
   axis. This is the primary control and should be the most obvious one.
4. **Adjust the plane's rotation** — tilt it off horizontal. Limbs are rarely
   cut perpendicular to the floor; the detected marker band is usually slanted.
   Two rotational degrees of freedom (tilt angle and the direction of tilt).
5. **Add a plane** the automatic detection missed, or when no marker was worn.
6. **Remove a plane** that was detected incorrectly.
7. **See the effect live** — the kept/discarded split should update as the plane
   moves, ideally with a running point count.
8. **Confirm** to run Stages 3b–6.

### Cut semantics — important for the UI to convey

The rule is *keep the side the limb's centroid is on*:

- **One plane** → keeps everything on the centroid's side. Cuts the limb once.
- **Two planes** → keeps the region *between* them. This extracts a segment
  (e.g. calf only), which is a genuinely useful clinical measurement.
- **Three or more** → keeps the region between the outermost planes.

So "add a marker" is not merely additive — going from one plane to two changes
the operation from *truncate* to *extract a slice*. The UI should make that
legible, because a user adding a second plane expecting another truncation will
be surprised.

### Parameterisation

Internally each plane is a point and a normal:

```json
{
  "centroid": [-0.0298, -0.2460, 0.9778],
  "normal":   [-0.2269,  0.9300,  0.2893],
  "npts": 191
}
```

Map the UI controls onto this as:

- **height slider** → moves `centroid` along the limb's vertical axis
- **tilt controls** → rotate `normal` away from vertical
- **add** → append a new entry, defaulting to horizontal at the clicked height
- `npts` is provenance only (how many marker points supported this plane);
  user-created planes can carry `npts: 0` or a `"source": "user"` field

### A required pipeline change

`cutting_line.json` is currently written in **original VGGT space**, before
levelling, while `leg_no_cut.ply` is written **after** levelling. A UI that
shows the cloud and the planes together needs them in the same frame, and "Z
height" is only meaningful once the scene is levelled.

**The pipeline must export the levelled marker planes** (transformed by
`R_total`) alongside the levelled cloud, and accept levelled planes back. This
is a small change in `pipeline/stages/clean.py` Phase C, but it must be done
before the UI can work.

Stage 3 must also be splittable at that point — currently `clean_and_extract()`
runs detection and cutting in one call.

---

## 4. Architecture implications

**A backend is required.** The pipeline needs a GPU and ~45 s per run; it cannot
run in the browser. This is a job-queue web app, not a static site.

```
browser ── upload photos ──> API ──> job queue ──> GPU worker
   ^                                                    |
   |<──── stage 3a done: leg cloud + planes ────────────|
   |                                                    |
   |───── adjusted planes ─────────────────────────────>|
   |                                                    |
   |<──── final meshes + volume ────────────────────────|
```

Requirements this creates:

- **Job state** — a run pauses mid-pipeline awaiting user input, so jobs need
  persistent state and IDs, not a single request/response.
- **Resumability** — the user may take minutes at the review step, or leave and
  come back. Stage 1's output is expensive; it must be cached and resumable.
  `stagerun.py` already implements exactly this caching model and is the natural
  basis for the worker.
- **Progress feedback** — 45 s is too long for a spinner with no detail. Show
  which stage is running; the stage boundaries are natural progress ticks.
- **Concurrency limit** — one GPU, ~8 GB. Serialise inference or queue it.
- **Upload validation** — 6–9 images, consistent camera, the cube visible in
  frame. Bad input fails late and confusingly otherwise.

**3D in the browser.** See §4b — three.js, and the assets are small enough to
ship as-is.

---

## 4b. Implementation notes — stack and 3D specifics

**Stack**: Next.js + React + three.js, via
[`@react-three/fiber`](https://r3f.docs.pmnd.rs) (three.js as React components)
and `@react-three/drei` (helpers — `OrbitControls`, `Grid`, `Bounds`,
`Center`). Writing raw three.js imperatively inside React is possible but fights
the framework.

**PLY loads natively** — `PLYLoader` from `three/examples/jsm/loaders/PLYLoader`.
No format conversion needed. All meshes carry vertex colours, so use
`meshStandardMaterial` with `vertexColors` enabled, and call
`computeVertexNormals()` after load (the exported PLYs do not always carry
normals).

**Payload sizes are small — ship the PLYs directly:**

| asset | verts | faces | size |
|---|---|---|---|
| `leg_mesh.ply` | 1,941 | 3,878 | 146 KB |
| `box_mesh.ply` | 6,945 | 13,886 | 522 KB |
| `scene_mesh.ply` | 8,886 | 17,764 | 469 KB |
| `leg_no_cut.ply` (review step) | 2,269 pts | — | ~60 KB |

No decimation or GLB conversion required. **Do not ship
`for_debug/02_pointcloud/points.ply`** — 1.2 M points, ~19 MB.

### Coordinate convention — this will bite you

**The pipeline outputs Z-up; three.js is Y-up.** Stage 3 levels the scene so the
floor is horizontal and Z is vertical — `leg_mesh` measures X=0.108, Y=0.108,
Z=0.251, i.e. the limb runs along Z.

Loading a PLY without correcting this lays the object on its side. Rotate the
loaded geometry −90° about X:

```js
mesh.rotation.x = -Math.PI / 2   // Z-up (pipeline) -> Y-up (three.js)
```

Do this once at load, consistently, or the grid will not line up with the
object's base.

### Grid

Use drei's `<Grid>` on the XZ plane (after the rotation above, that is the real
floor). Because Stage 3 levels the scene against the detected ground plane, the
object's base genuinely sits on y=0 — the grid is a physically meaningful floor,
not decoration.

Size grid cells in **real units**: multiply mesh coordinates by `linear_scale`
from `volumes.csv` (~53.6 cm per mesh unit), or scale the whole scene into
centimetres at load so a 1-unit grid cell is 1 cm. The latter is simpler and
makes dimension labels trivial.

### Review-step plane widget

The cutting plane needs a custom object — no drei helper covers it. Build it as
a `<planeGeometry>` with `transparent`, `opacity ~0.35`, `side: DoubleSide`,
positioned at the plane centroid and oriented by its normal
(`quaternion.setFromUnitVectors(new Vector3(0,0,1), normal)`).

For the kept/discarded split, the cheapest approach is two point-cloud objects
computed client-side — the signed-distance test is a dot product per point, and
at 2,269 points that runs comfortably every frame while dragging.

---

## 4c. Deployment — local compute, hosted frontend

**Decision: the pipeline runs locally, not in the cloud.**

```
Next.js frontend ──> Vercel / static host (free, always up)
                        |
                        v
FastAPI + pipeline ──> the project machine (RTX 4060),
                        exposed via Cloudflare Tunnel
```

### Why local

**Health data.** Limb photos are health data. The commercial VGGT licence
explicitly forbids inferring health information "without rights and consents
required by applicable laws", and uploading patient scans to a rented GPU
creates real compliance exposure (PDPA, GDPR). Processing locally means images
never leave the machine — a methodological strength worth stating explicitly,
not just an implementation convenience.

**Cold starts would hurt the demo.** Measured on the project machine:

```
model load     14.3 s      <- paid again on every cold container start
inference      20–25 s
full pipeline  ~45–60 s
```

On serverless GPU that 14.3 s recurs per cold start, plus container init — 60–90
seconds before anything appears. Locally the model stays warm and timing is
predictable, which matters in front of a committee.

**Cost.** The GPU already exists. Cloud GPU is billed idle-or-not (~$0.50–1/hr)
or per-second on serverless; neither is justifiable for bursty student-project
usage.

**The job model already fits.** `stagerun.py` caches Stage 1 and resumes from
any stage — exactly what the paused review step needs, and it behaves the same
locally.

### Required fallback — precomputed sample results

The obvious failure mode is the demo machine being off or the network dropping.
**Ship precomputed outputs for the sample datasets** (`est_325`, `small_leg`,
`short_leg`, `baam` — all already produced). The site serves those with no
backend at all, so:

- the "look what it does" path always works, even with the GPU offline
- only *new uploads* require the tunnel to be live
- a visitor who does not want to upload anything still sees the full result

Build this path first. It is also the entire static-site version of the product,
and it de-risks the demo completely.

### Upgrade path

If this later needs multi-user or 24/7 availability, serverless GPU (Modal,
RunPod) is the natural move — per-second billing, with the model kept warm on a
schedule to avoid the cold-start cost. The local-first design does not block
that; the pipeline is already a queue-friendly, resumable job.

---

## 5. Page structure

### Landing
What it does in one sentence. One strong visual: input photos beside the
resulting 3D mesh. A clear call to action to upload.

### Sample results — build this first
Precomputed runs for the bundled datasets, browsable with no backend. This is
the whole product minus the upload, it always works, and it is what a visitor
who does not want to upload anything actually wants. See §4c.

### Upload
Drag a folder of photos. State the requirements plainly: 6–9 photos, orbit
around the object, keep the reference cube visible in every shot, marker band on
the limb where you want the measurement to stop.

Requires the compute backend to be reachable; degrade gracefully to the sample
results when it is not, rather than showing a dead form.

### Processing
Stage-by-stage progress. Worth showing intermediate artifacts as they appear —
the point cloud arriving after Stage 2 is genuinely impressive and fills the
wait.

### Review the cut
The interactive step. Should feel like the centre of the product, not a
technical detour. See section 3.

### Result
The centrepiece is **`leg_mesh` rendered in interactive 3D beside the volume
figure.** Requirements:

- Orbit / zoom / pan, with the mesh auto-framed on load
- **A ground grid** under the object, sized in real units — the pipeline levels
  the scene so the grid is a true floor plane, not decoration. It gives the
  object scale and makes the measurement feel physical
- The volume in cm³ and in mL, large and legible
- Real-world dimensions (height × width × depth), from the OBB not the AABB
- Toggle between `leg_mesh` and `scene_mesh` so the reference cube is visible
  next to the object — that comparison is what makes the scale believable
- Downloads: PLY and STL
- The reference cube's measured volume shown as an honesty indicator, see §7

### How it works
The six stages, explained for someone who wants to understand rather than
operate. Can be a separate page.

---

## 6. Visual assets available

All exist already or are one command away:

- **Input photos** — `inputs/*/`, ordinary phone shots
- **Point cloud** — `for_debug/02_pointcloud/points.ply`, ~1.2 M coloured points
- **Segmented objects** — `for_debug/03_clean/objects/`
- **Before/after the cut** — `leg_no_cut.ply` vs `leg_cut.ply`, a clear pairing
- **Final meshes** — `output/{leg_mesh,box_mesh,scene_mesh}.ply`
- **Depth and confidence maps** — `for_debug/01_inference/raw/`, already rendered
  as turbo-colormapped PNGs, visually striking
- **Marker planes** — `debug/cutting_line.json`

Renders can be produced headless via Open3D's `OffscreenRenderer` (EGL), so
consistent camera angles across all assets are easy to generate.

---

## 7. Results, stated carefully

| what | value |
|---|---|
| test object | 325 ml drink can |
| measured volume | ~309 cm³ |
| reference cube reads | ~2500 cm³ vs 2744 nominal |
| surface noise floor | ~2 mm |

**These must not be presented as an accuracy claim.** State plainly why:

- 325 ml is the can's *fill* volume; the pipeline measures *external*
  displacement, which is larger. Different quantities.
- With one known object, scale **cannot be validated** — the cube defines it, so
  it can never disagree with itself.
- The 14 cm reference size is itself unverified; the cube is handmade.

Showing the reference cube's measured volume in the result UI is a strength: it
is the closest thing to a live self-check the system has. Under the old scale
derivation it always read exactly 2744 cm³ because it was the denominator —
meaningless. It can now disagree, and that disagreement is informative.

---

## 8. Tone

Technical, plain, honest. This project's most interesting property is that it is
**rigorously self-critical** — several promising results turned out to be two
errors cancelling, and the work is stronger for having caught them.

Avoid: "revolutionary", "AI-powered", "state-of-the-art", percentages without
their caveat.

---

## 9. What NOT to do

- **Do not claim clinical accuracy or medical utility.** The commercial VGGT
  licence explicitly forbids unlicensed medical practice and inferring health
  data without consent. Limb scans are health data.
- **Do not present a single headline accuracy percentage.** It changes entirely
  depending on which ground truth is assumed.
- **Do not describe it as real-time or instant.** ~45 s of compute plus user
  review time.
- **Do not hide the review step** behind an "advanced" toggle. It is where the
  user's judgement enters the measurement, and it is the product.

---

## 10. Open question

The strongest possible version of this includes a **validation result** —
calibrate on the cube, predict a second known object's dimensions, compare
against a caliper measurement. That number does not exist yet. If it does by the
time this is built, it should be the headline.
