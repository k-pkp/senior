# Repository review — 2026-08-22

Prompted by a bug that nothing caught: Stage 6 was reverted, its CSV column names
changed, and the web app read the old ones. `14.0 / 0` produced `Infinity`, every
vertex was multiplied by it, and the 3D silently vanished — no error, because
nothing had failed. The file arrived and parsed; it was ruined afterwards.

That failure has a shape: **a contract that nobody checks, and a failure path that
says nothing.** This is a sweep for the rest of that shape, done mechanically
rather than by reading.

**How it was checked** — repeatable:

| check | method |
|---|---|
| dead functions | AST walk, name referenced ≤ once across the tree |
| unread config | every `^[A-Z_]+ =` in `config.py` grepped against all other sources |
| silent failures | AST walk of every `ExceptHandler`, flagged when the body neither prints, logs, raises nor exits |
| artefact contracts | keys written into each JSON by a real run vs keys read by any Python or TypeScript consumer, both directions |
| dependencies | AST import walk vs `requirements.txt` |
| stale references | grep for names of removed flags, files and stages |

---

## High — would break for someone else, or change a number without saying

### 1. `fastapi` is missing from `requirements.txt`
A fresh clone that runs `pip install -r requirements.txt` cannot start `serve.sh`;
`service/app.py` fails on import. `fastapi`, `uvicorn[standard]` and
`python-multipart` were installed ad hoc when the service was written and never
recorded. `pydantic` arrives with fastapi, so one line fixes two of the three.

**Verified.** The imports are real and the names are absent from the file.

### 2. `.gitignore` blankets `*.ply` and `*.stl`
Lines 110 and 115. This already nearly shipped a broken demo: the web app's
sample meshes live in `web/public/samples/` and were only committed because they
were force-added. Nothing prevents the next person adding sample data and silently
losing it. A path-scoped exception (`!web/public/samples/**`) would.

**Verified.** `git check-ignore` confirms the sample meshes match the rule.

### 3. `_is_closed` falls back to a *different* definition of watertight
`workers/recons_methods_worker.py:94` and `pipeline/core/mesh.py:45`. Both build a
trimesh to test closure and, on any exception, silently fall back to Open3D's
`is_watertight()` — which the docstring itself says is not the definition the rest
of the pipeline uses.

This is the criterion the alpha ladder selects on. If the trimesh construction ever
throws, α selection quietly changes its meaning and the chosen mesh may not be
closed by the definition Stage 6 then integrates.

**Latent, not observed.** No run in this session hit the fallback. The risk is that
if it ever fires, nothing says so.

### 4. Statistical outlier removal can silently do nothing
`pipeline/core/filters.py:50`. On any exception the function returns the points,
colours and confidences unchanged — no message. A failure here means the cloud
carries its outliers into clustering and reconstruction, and the only symptom is a
different number.

**Latent, not observed.**

### 5. A detector failure silently shrinks the cube's bounding box, inside the gate
`pipeline/stages/prep.py:220`. `_cube_bbox` unions the ArUco face quads with a
GroundingDINO box. If `vlm.detect` raises, `det = None` and the box falls back to
the faces alone — which is *smaller*. A smaller cube box is easier to fit inside
the crop window, so a frame that should be rejected could pass.

That inverts the gate's purpose. It is the one place in the pipeline where a
swallowed exception makes the check more permissive rather than less.

**Latent, not observed.**

---

## Medium — dead weight and duplicate truth

### 6. 17 MB of YOLO weights that nothing loads
`yolo11n.pt`, `yolo11n-seg.pt`, `yolo11n-pose.pt` at the repository root. No
`ultralytics` import exists anywhere; `SEG_MODEL = "yolo11n-seg.pt"` in
`prep.py:66` is never read, and the `seg_model=` parameter on `prepare_frames` is
declared and never used in the body. Stage 0 moved to GroundingDINO + SAM and this
was left behind.

### 7. A second `compute_volumes` at the root
`volume.py` defines its own `compute_volumes` at line 269 with a different
signature from `pipeline/stages/volume.py`, and calls it at 358 with arguments the
pipeline version does not accept. Nothing imports it. It is a parallel
implementation of the project's most contested stage, unmaintained and out of step
with the one that runs.

`viewer.py` is likewise orphaned — nothing imports it.

### 8. `thresholds_from_colour()` — dead, and harmful if revived
`pipeline/core/segmentation.py:97`. Written to re-aim the colour detector at a
learned marker colour, never wired up. Tested during this session: it selects
**102,988 points** — the entire limb — and puts the cut 89° off, because its hue
window admits skin. It should be deleted rather than left as an apparently
available option.

### 9. `_merge_duplicates` in `workers/meshfix_worker.py:52` is unreferenced.

### 10. 26 MB of leftover experiment directories
`temp_output_compare_recon/` (20 MB), `compare_ghost/` (4.7 MB),
`compare_recon/` (1.1 MB), plus `Volume measurement app brief.zip` at the root.
All predate this session; none are referenced by any code or document.

---

## Low

- `pipeline/stages/volume.py` prints `STAGE 7` in its banner. Cosmetic, and it is
  main's code, so it predates the rework.
- `requirements.txt` calls warp "optional: GPU voxelization for Stage 7".

---

## Checked and clean

Worth recording so it is not re-done.

- **JSON artefact contracts.** `framing.json`, `manifest.json`, `levelling.json`,
  `cutting_line_levelled.json` — every key written by a real run is read by some
  consumer. No dangling writes.
- **`volumes.csv` contract, both directions.** The web reads 18 columns; every one
  exists in one of the two Stage 6 schemas, and the reader handles both. This was
  the bug that prompted the review and it is now closed.
- **Config constants.** Every `^[A-Z_]+` in `config.py` is read somewhere.
- **`--ghost-filter`** is *not* a stale flag. Remaining mentions are prose
  describing the operation, not a removed option.
- **`pipeline/detection.py`** is live, behind `--use-detection`, and imports
  cleanly.

---

## Not checked

Stated so the review is not mistaken for more than it is.

- **`vggt/`** — 42 files of vendored upstream model code, not reviewed.
- **`web/`** — only the data contract was checked, not the React or three.js.
- **Numerical correctness** of any geometry beyond what the session already
  measured. This sweep looked for *unchecked contracts and silent failures*, not
  for wrong maths.
- **Whether the latent failures in 3, 4 and 5 can actually be triggered.** They are
  reachable by inspection; none was observed firing.
