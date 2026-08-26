# Repository review — 2026-08-22

> **Status: every finding in this file is now closed.** Items 3, 4, 5 and 11 —
> the three latent silent failures and the ignored flag — were fixed on
> 2026-08-27; each entry below records what it became. The High items and all
> the dead code were already fixed. What each finding
> turned into is recorded inline below. Two things changed after this review was
> first written and are worth reading before the list: a much more serious bug
> was found afterwards (Stage 0's output being discarded by `stagerun.py 0-6`,
> see `experiments/stage0_bypass_bug.png`), and three items listed here as
> "orphaned" were wrong — `viewer.py`, `volume.py` and `pipeline/detection.py`
> were all reachable. "Nothing imports it" is not a test for a CLI tool.

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

### 1. `fastapi` is missing from `requirements.txt` · **FIXED**
A fresh clone that runs `pip install -r requirements.txt` cannot start `serve.sh`;
`service/app.py` fails on import. `fastapi`, `uvicorn[standard]` and
`python-multipart` were installed ad hoc when the service was written and never
recorded. `pydantic` arrives with fastapi, so one line fixes two of the three.

**Verified.** The imports are real and the names are absent from the file.

### 2. `.gitignore` blankets `*.ply` and `*.stl` · **FIXED**
Lines 110 and 115. This already nearly shipped a broken demo: the web app's
sample meshes live in `web/public/samples/` and were only committed because they
were force-added. Nothing prevents the next person adding sample data and silently
losing it. A path-scoped exception (`!web/public/samples/**`) would.

**Verified.** `git check-ignore` confirms the sample meshes match the rule.

### 3. `_is_closed` falls back to a *different* definition of watertight · **FIXED 2026-08-27**
`workers/recons_methods_worker.py:94` and `pipeline/core/mesh.py:45`. Both build a
trimesh to test closure and, on any exception, silently fall back to Open3D's
`is_watertight()` — which the docstring itself says is not the definition the rest
of the pipeline uses.

This is the criterion the alpha ladder selects on. If the trimesh construction ever
throws, α selection quietly changes its meaning and the chosen mesh may not be
closed by the definition Stage 6 then integrates.

**Latent, not observed.** No run in this session hit the fallback.

**Fixed:** both sites now print a warning naming the exception and stating that
the criterion the alpha ladder selects on has changed definition.

### 4. Statistical outlier removal can silently do nothing · **FIXED 2026-08-27**
`pipeline/core/filters.py:50`. On any exception the function returns the points,
colours and confidences unchanged — no message. A failure here means the cloud
carries its outliers into clustering and reconstruction, and the only symptom is a
different number.

**Fixed:** the handler names the exception and the point count it kept.

### 5. A detector failure silently shrinks the cube's bounding box, inside the gate · **FIXED 2026-08-27**
`pipeline/stages/prep.py:220`. `_cube_bbox` unions the ArUco face quads with a
GroundingDINO box. If `vlm.detect` raises, `det = None` and the box falls back to
the faces alone — which is *smaller*. A smaller cube box is easier to fit inside
the crop window, so a frame that should be rejected could pass.

That inverts the gate's purpose. It is the one place in the pipeline where a
swallowed exception makes the check more permissive rather than less.

**Fixed:** the handler says so explicitly — that the box has fallen back to the
faces alone, that this under-covers the cube, and that the frame's framing check
is therefore more permissive than it should be.

### 11. `--no-prep-crop` is silently ignored by `run.py` · **FIXED 2026-08-27**
`pipeline/cli.py:29` parses the flag into `args.prep_crop`, but the
`prepare_frames` call at `pipeline/orchestrator.py:202` omits both `crop=` and
`output_size=`. The value is never read, so Stage 0 crops whatever the user asked
for. `stagerun.py:325` passes both, so the same flag works there and not here.

Same shape as the rest of this list: a contract nobody checks, failing without a
word. It is the second thing found in `orchestrator.py`'s Stage 0 call — the
first was `stagerun.py` discarding the stage's output entirely.

**Verified**, not inferred. Stubbing `prepare_frames` and running
`run.py --no-prep-crop` showed the call arriving with exactly
`band_heights, centre_on_subject, min_frames, pad, strict`.

**Fixed:** `orchestrator.py` now threads `crop=` and `output_size=` through, via
`getattr` so the two entry points cannot drift apart again if a flag is added to
only one parser. Re-verified by the same stubbing method: the call now arrives
with `crop=False` when the flag is passed.

---

## Medium — dead weight and duplicate truth

### 6. 17 MB of YOLO weights that nothing loads · **DELETED**
`yolo11n.pt`, `yolo11n-seg.pt`, `yolo11n-pose.pt` at the repository root. No
`ultralytics` import exists anywhere; `SEG_MODEL = "yolo11n-seg.pt"` in
`prep.py` is never read, and the `seg_model=` parameter on `prepare_frames` is
declared and never used in the body. Stage 0 moved to GroundingDINO + SAM and this
was left behind.

### 7. A second `compute_volumes` at the root · **KEPT — the claim was wrong**
`volume.py` defines its own `compute_volumes` at line 269 with a different
signature from `pipeline/stages/volume.py`, and calls it at 358 with arguments the
pipeline version does not accept. Nothing imports it. It is a parallel
implementation of the project's most contested stage, unmaintained and out of step
with the one that runs.

**Correction.** Both were checked by running them, and both work: `volume.py`
prints a full scale report, `viewer.py` prints usage and is advertised by
`orchestrator.py:138`. They are standalone tools, not stale duplicates, and
"nothing imports it" was the wrong test. Both kept. `volume.py` does duplicate
the scale derivation, which is worth resolving when Stage 6 settles.

### 8. `thresholds_from_colour()` — dead, and harmful if revived · **DELETED**
Formerly in `pipeline/core/segmentation.py`. Written to re-aim the colour detector at a
learned marker colour, never wired up. Tested during this session: it selects
**102,988 points** — the entire limb — and puts the cut 89° off, because its hue
window admits skin. It should be deleted rather than left as an apparently
available option.

### 9. `_merge_duplicates` in `workers/meshfix_worker.py` · **DELETED**

### 10. 26 MB of leftover experiment directories · **DELETED**
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
- **`pipeline/detection.py`** was live behind `--use-detection` — but the block
  calling it discarded both of its results (`_, _ = seeds_to_xyz_labels(...)`,
  and a write-only `args._detection_seeds`), so the flag cost two model loads per
  run and changed nothing. Block, flag and module removed together. If detection
  was meant to feed Stage 3's clustering and simply never got wired up, that is a
  feature to build: `git show 594fbb1^:pipeline/detection.py` brings it back.

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
