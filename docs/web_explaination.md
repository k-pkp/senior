# The web app, explained from scratch

Written for someone who has not worked with web development before, and who may
need to explain this to a supervisor. It covers what the app is for, what each
tool does and why it was chosen, how the code is organised, and where the real
engineering is.

---

## 1. What it is, and what it is not

**It is a viewer and a review tool.** It shows the results the Python pipeline
produced: the reconstructed 3D meshes, the measured volume, the photos that were
accepted or rejected, and — the important one — it lets a person inspect and
correct the plane where the limb was cut.

**It does not do the measurement.** No reconstruction, no VGGT, no volume
integration happens in the browser. All of that is the Python pipeline, on the
GPU. The web app reads what the pipeline wrote.

That division matters when explaining it: the science is in the pipeline, and the
web app is how a human checks and steers it. If a supervisor asks "does the
browser compute the volume?", the answer is no — with one deliberate exception
described in section 6, where it recomputes a *preview* so that dragging a control
feels immediate.

---

## 2. The tools, and why each one

If you have not built a website before, the vocabulary is the hardest part. Each
tool below solves one problem.

### HTML, CSS, JavaScript — the browser's three languages

Every web page is these three. **HTML** is the content and structure ("this is a
heading, this is a button"). **CSS** is the appearance ("this button is blue").
**JavaScript** is the behaviour ("when this button is clicked, do something").
A browser understands only these three. Everything below eventually becomes them.

### React — building a page out of reusable pieces

Writing raw HTML for a page that changes constantly is painful, because you have
to describe *how* to change it: find this element, update that text, hide this
panel. React lets you instead describe *what the page should look like for a given
state*, and it works out the changes for you.

The unit is a **component** — a function that returns a piece of the page. In this
project, `Result.tsx` is a component, `Viewport.tsx` is a component. They can be
reused and nested, which is why the file list looks like a set of building blocks.

The other React idea used constantly here is **state**: a value that, when it
changes, causes the page to redraw. For example in `page.tsx`:

```tsx
const [screen, setScreenRaw] = useState<Screen>("samples");
```

`screen` remembers which page the user is on. Calling `setScreen("result")`
changes it and React redraws with the Result screen showing. There is no manual
"hide this div, show that div" anywhere.

### Next.js — the framework around React

React alone is just the component library. **Next.js** supplies everything else a
real site needs: a development server that reloads when you save a file, a build
step that bundles and optimises the code, routing, and static file serving.

Here it is used in a deliberately simple way. There is **one page**
(`src/app/page.tsx`) and the "screens" are swapped inside it, rather than being
separate URLs. Next.js also serves `public/` directly — so a file at
`public/samples/small_leg/leg_mesh.ply` is fetchable at
`/samples/small_leg/leg_mesh.ply`. That is how the meshes reach the browser.

`next build` produces a fully static site: plain files that can be hosted
anywhere, with no server running behind them.

### TypeScript — JavaScript that checks itself

**TypeScript** is JavaScript plus type declarations. You state what shape data
has, and mistakes are caught before the code runs rather than as a blank screen
later.

This is load-bearing here because the browser is reading files the Python pipeline
wrote, and the two sides have to agree on their contents. `src/lib/types.ts` is
that written-down agreement:

```ts
/** A row of for_debug/06_volume/volumes.csv */
export interface VolumeRow {
  name: string;
  is_ref: boolean;
  volume: number;        // mesh units³
  obb_a: number;         // the VERTICAL axis, not the largest
  real_vol_cm3: number;
  ...
}
```

If Stage 6 ever renames a column, the browser code stops compiling instead of
silently displaying a wrong number. The comments in that file are part of the
point — `obb_a` being the vertical axis rather than the longest is exactly the
kind of detail that causes a subtle bug months later.

### three.js — 3D graphics in a browser

Browsers can draw 3D through a low-level interface called WebGL, but programming
it directly is extremely tedious. **three.js** is the standard library that wraps
it in usable concepts: a scene, a camera, meshes, lights, materials.

It also supplies the **PLYLoader**, which reads the `.ply` mesh files the pipeline
exports. That is why the pipeline's output format needs no conversion for the web.

### React Three Fiber and drei — three.js expressed as React components

**@react-three/fiber** lets you write three.js using React components instead of
imperative setup code, so the 3D scene lives in the same style as the rest of the
app. **@react-three/drei** is a collection of ready-made helpers on top of it —
this project uses `OrbitControls` (drag to rotate, scroll to zoom), `Grid` (the
floor), and `Bounds` (frame the camera on an object).

Those two are why `Viewport.tsx` is short. Without them it would be several
hundred lines of camera and renderer boilerplate.

---

## 3. How the code is organised

```
web/
  package.json              the tool list and the run commands
  src/app/
    layout.tsx              the HTML shell wrapped around every page
    page.tsx                the single page: header, tabs, and which screen shows
    globals.css             colours, fonts, the light/dark theme variables
  src/components/
    screens/                one file per screen the user sees
      Samples.tsx           pick a precomputed dataset
      Upload.tsx            choose your own photos
      Framing.tsx           Stage 0's verdict: which photos passed and why
      Processing.tsx        stage-by-stage progress
      Review.tsx            inspect and correct the cut plane      <- the core screen
      Result.tsx            the measured volume and the 3D result
      How.tsx               plain-language explanation of the method
    three/                  everything 3D
      Viewport.tsx          the shared 3D stage: camera, lights, grid, controls
      MeshView.tsx          draw one mesh
      CutReview.tsx         the cut plane and the live split
      usePly.ts             load a .ply and put it in the right place  <- important
    ui/primitives.tsx       small shared pieces: Button, Panel, Label, Stat, Slider
  src/lib/
    types.ts                the shape of every file read from the pipeline
    data.ts                 which samples exist, CSV parsing, scale derivation
    api.ts                  talking to the compute service            <- the live half
    theme.ts                light/dark handling
  public/samples/           the precomputed pipeline outputs the site ships with

service/                    the compute service (Python, FastAPI)
  app.py                    the HTTP endpoints and the artifact map
  jobs.py                   job state and the single-GPU worker queue
```

The convention worth knowing: **a file is a component, and its name is what it
renders.** `Review.tsx` exports `Review`. `.tsx` means "TypeScript containing
page markup".

---

## 4. What the user actually does

```
Samples ─────────────────────────────► Result
   │        (open a precomputed run)
   │
   └─► Upload ─► Framing ─► Processing ─► Review ─────► Result
       photos    accepted/   stages 1-6   check the     volume
                 rejected                 cut, drag it

                            re-cut, stages 3-6
                            └───────◄──────────┘
                            (Review sends the planes back, and the
                             new volume returns to Result)
```

**Samples** — two precomputed runs ship with the site (`small_leg`, `est_325`), so
it demonstrates fully with no GPU and no server. This is what you would show in a
presentation.

**Upload** — choose 6–12 photos. The files are POSTed to the compute service in
`service/`, which stores them under `work/<job>/input/` and starts Stage 0. This
path needs that service running; `page.tsx` probes it once at startup so the
screen can say so honestly rather than appearing to work and then hanging.

**Framing** — Stage 0's verdict, per photo. Each is shown with the boxes the
pipeline actually measured drawn on it: yellow for the crop it chose, magenta for
the reference cube, green for the marker band, and a badge saying ACCEPTED or
REJECTED with the reason. If a photo is rejected the user needs to know *which*
one and *why*, and a filename alone does not communicate that.

The reason comes with a **severity chip**, because rejection is not one thing:

| reason | severity | rejected? |
|---|---|---|
| `marker missing, not crucial` | not crucial | **no** |
| `cube missing, crucial` | crucial | yes |
| `marker and cube missing, very crucial` | very crucial | yes |
| `objects out of window` | crucial / very crucial | yes |

A missing marker band costs the *cut* but not the *scale*, and the cut only needs
the band on some photos — so that photo is flagged and still used. A missing or
clipped cube costs the scale of every number the run reports, which is what the
gate exists to prevent. The out-of-window reason does not say which object was
clipped, because the window is not adjustable — it is the largest square the photo
allows — so "step back and re-take" is the answer either way.

This screen is a **gate**, not a step: it decides whether the photos can be
measured at all. When every photo passes there is one button. When some do not,
there are two — re-take, or measure anyway with what passed. Measuring anyway
still runs the pipeline; it just hands the rejected frames to VGGT uncropped, so
whatever VGGT's own centre crop removes is lost.

**Processing** — the six stages, ticked off as the service finishes each one.
The ticks are observed, not timed: the service launches each stage as its own
subprocess, so it knows which one is running. The seconds beside each row are
expectations only. If a stage crashes, this screen shows which one and the tail
of its output, because a failure that looks like a successful run producing
nothing is the worst thing a progress bar can do.

**Review** — described in section 6.

**Result** — the volume, the dimensions, and the 3D mesh to rotate. Where a
nominal ground truth exists (the 325 ml can) it shows the comparison; where none
exists (a limb) it deliberately shows none rather than inventing one.

---

## 5. Where the data comes from

There is **no database**. Both halves work in files: the pipeline writes them and
the site reads them over plain HTTP. The difference between the two paths is only
where those files sit.

- A **sample** is a directory under `web/public/samples/`, served as static
  content. No Python involved.
- A **live job** is a directory under `work/<job_id>/`, served by `service/`.

`src/lib/api.ts` exploits that symmetry: `jobDataset(id)` returns the same
`SampleDataset` shape with the URLs pointed at the service, so `Result` and
`Review` never learn that jobs exist.

`src/lib/data.ts` lists the samples:

```ts
export const SAMPLES: SampleDataset[] = [
  {
    id: "small_leg",
    label: "Lower leg",
    nominalMl: null,          // no ground truth for a limb — do not invent one
    frames: 6,
    meshes: {
      leg: "/samples/small_leg/leg_mesh.ply",
      box: "/samples/small_leg/box_mesh.ply",
      ...
    },
    volumesCsv: "/samples/small_leg/volumes.csv",
    cuttingLine: "/samples/small_leg/cutting_line.json",
  },
  ...
];
```

Those paths are files under `public/`, copied from a pipeline run. To add a
dataset you copy the outputs in and add an entry.

---

## 6. The interesting engineering

Three problems here are worth explaining to a supervisor, because they are where
the browser has to agree precisely with the Python.

### 6.1 Two different ideas of "up"

The pipeline levels the scene so that **Z is vertical** — Stage 3 fits the ground
plane and rotates everything so the floor is flat and Z points up. three.js
instead assumes **Y is vertical**.

Load a mesh without accounting for that and the limb lies on its side, and the
grid cuts through it instead of meeting its base. `usePly.ts` fixes it on load:

```ts
geom.rotateX(-Math.PI / 2);   // Z-up -> Y-up
```

### 6.2 Two different ideas of "one unit"

The pipeline's meshes are in arbitrary reconstruction units, not centimetres. The
conversion is the `linear_scale` Stage 6 derives from the reference cube, and
`data.ts` reproduces that derivation exactly:

```ts
// Mirror Stage 6 exactly: scale comes from the two HORIZONTAL edges only.
// obb_a is the vertical axis, which the floor truncates — averaging it in
// drags the estimate small and inflates everything measured against it.
const meanEdge = (ref.obb_b + ref.obb_c) / 2;
return REFERENCE_CM / meanEdge;
```

That is applied once at load, so **one grid square is one centimetre** and every
displayed dimension needs no further conversion. The mesh is then centred and
dropped so its lowest point sits at y = 0 — standing on the floor rather than
floating.

A subtlety that caused a real bug: anything positioned in the same space —
notably the cut planes — must receive the *identical* transform, or it drifts away
from the mesh. So the loader records what it did:

```ts
geom.userData.sceneOffset = offset;
geom.userData.sceneScale = scale;
```

and `pointToScene` / `dirToScene` replay it. Note that directions rotate but must
never be translated or scaled — a normal vector has no position and no length in
centimetres. Getting that wrong tilts the plane subtly and is hard to spot by eye.

### 6.3 The cut rule, mirrored in the browser

This is the heart of the app.

The pipeline cuts the limb at a plane found from the coloured marker band. That
detection can be wrong, and the volume depends on it directly — so a human should
be able to check it and adjust it, with the result updating as they drag rather
than after a round trip to the GPU.

That means the browser has to apply **the same cut rule** the pipeline does. From
`CutReview.tsx`:

```
Mirrors core/segmentation.py:apply_marker_cut exactly:
  0 planes -> no cut
  1 plane  -> keep what is BELOW it
  2 planes -> keep what is BETWEEN them
```

The test itself is one dot product per point per plane: for each point, is it on
the keep side of this plane? At a few thousand points that is far below one
frame's budget, so it can run on every drag with no delay and no debouncing.

Two details that are easy to get wrong and are handled explicitly:

- **Normal direction.** A plane's normal can point either way, and the detected
  sign is arbitrary. Every normal is flipped to point upward first, so "below"
  means the same thing regardless of which way detection happened to orient it.
- **Vertical planes.** A plane standing exactly vertical has no above or below —
  the rule is undefined. It is skipped rather than guessed at.

The user manipulates the plane through three controls — height, tilt, and tilt
direction — and `planeFromControls` in `Review.tsx` rebuilds the plane's normal
from them, inverting the load-time transform so the height slider reads in real
centimetres.

The screen also keeps the plane's **original detected position** when the user
moves it (`origin` in the `CutPlane` type), so the display can show what was
measured next to what the user is proposing. Without that, dragging erases all
evidence of what the pipeline actually found.

Confirming sends the planes back. `POST /jobs/{id}/recut` writes them to
`work/<job>/planes.json` and re-runs **Stages 3-6** — `clean.py` takes them as
`override_planes` in place of what it detected, and the cut, the cross-section
caps and the volume all follow the person instead of the colour threshold. Stage
1 is not re-run: its `predictions.npz` is already on disk, and it is by far the
expensive part, so an edit costs about thirty seconds rather than a full run.

So the browser split and the Python cut do the same arithmetic for two different
purposes. The browser's is instant and approximate — points, no surface. The
service's is authoritative: it rebuilds a watertight mesh and integrates it. The
displayed result always comes from the second.

---

## 7. Running it

Both halves at once, which is what a demo needs:

```bash
./serve.sh       # web on :3111, compute service on :8000
```

The site by itself, for styling or layout work:

```bash
cd web
npm install      # download the tools listed in package.json (once)
npm run dev      # development server, reloads on save -> http://localhost:3000
npm run build    # produce the site
npm start        # serve the built site
```

`npm` is Node.js's package manager — the equivalent of `pip` for Python.
`node_modules/` is the downloaded dependencies, equivalent to a virtualenv, and is
not committed.

The samples path needs none of the Python pipeline running. Only Upload does.
`docs/running_the_web_app.md` covers the rest: the API, what happens during an
upload, and the Windows step WSL2 needs before a phone on the same wifi can
reach it.

---

## 8. Honest limitations

Worth stating plainly, because a supervisor will ask.

- **The upload path needs the compute service running.** It lives in `service/`
  now (see `docs/running_the_web_app.md`), but it is a separate process holding a
  GPU, not something the page can do by itself. Without it, only the precomputed
  samples work. The app detects this and says so rather than pretending.
- **The service is unauthenticated.** CORS is limited to localhost and private
  address ranges, but anyone who can reach the port can queue a job. It is built
  for a demo on one wifi, not for the internet.
- **Edits in Review are previewed in the browser, then confirmed in Python.**
  Moving the plane re-splits the point cloud immediately so the effect is visible
  while dragging, but that split is an estimate. Confirming sends the planes back
  and re-runs Stages 3-6, which produces the watertight mesh and the exact volume.
  The two numbers will not agree to the last decimal, and only the second one
  should be quoted.
- **Stage 6 is mid-discussion.** The volume stage has been reverted to the version
  on `main` pending review by the person who designed it. It writes different CSV
  columns from the parked version; the app reads both and scales the scene the way
  whichever stage wrote the file did, so runs display either way. What is not
  settled is the measurement: under main's method the reference cube reports
  exactly 2744.00 cm³ by construction, so the stage currently gives no error signal
  about itself. See `docs/stage06_experiments.md`.
- **Two datasets ship with it**, both from the same session. It demonstrates the
  method; it does not demonstrate accuracy across subjects.
- **No accuracy claim is displayed for the limb**, because none is justified —
  there is no independent ground truth for it. `nominalMl: null` in `data.ts` is
  deliberate, with the comment "do not invent one".

---

## 9. If you are asked one question, expect this one

*"Why build a web interface at all — isn't the pipeline enough?"*

Because one step in the pipeline is a judgement, not a calculation. Where the limb
is cut determines the volume, and that plane comes from detecting a coloured band,
which can fail — it has failed, on real data, in ways described in
`docs/changes_newVSold.md`. A number produced from a wrong cut looks exactly like a
number produced from a right one.

The web app makes that step visible and correctable: it shows where the cut landed,
lets a person move it, and shows the consequence immediately. It also shows which
photographs were rejected and why, so a bad capture can be fixed at the point where
fixing it is cheap.

The rest of the interface — the meshes, the dimensions, the stage list — is there
so that judgement is made with context.
