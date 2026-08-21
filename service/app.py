"""HTTP front end for the measurement pipeline.

The browser cannot do this work: stage 1 is VGGT-1B and stage 0 loads
GroundingDINO and SAM, so "end to end in the web app" means the page drives a
GPU on the local network. This service is that link, and nothing more — it
validates an upload, runs `stagerun.py`, and serves what the stages wrote.

Run it from the repo root, which is where `work/` lives:

    uvicorn service.app:app --host 0.0.0.0 --port 8000

The flow it exposes mirrors the pipeline's own shape rather than flattening it.
Stage 0 is a gate: it decides whether a set of photos can be measured at all,
and a frame that clips the reference cube corrupts the scale for every number
downstream while looking perfectly normal. So an upload runs stage 0 alone and
stops, the user is shown what was rejected and why, and only then do stages
1-6 run.

The cut is treated the same way. Stages 1-6 run with `--no-cut`, so the limb is
reconstructed and measured whole while the marker planes are only detected and
published. The user approves or moves them, and stages 3-6 re-run to apply the
cut they chose. Cutting on the first pass instead would put a volume on screen
that was derived from a cut nobody had agreed to, and then ask them to confirm
it -- the confirmation would be theatre.
"""

import os
import re
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline.config import IMAGE_EXTENSIONS
from service.jobs import PROJECT_ROOT, REGISTRY, Job

# A capture needs enough views to triangulate and few enough to fit one GPU
# pass. Six is stage 0's own minimum (pipeline/config.py MIN_FRAMES); twelve is
# where VGGT's memory stops being comfortable at 518.
MIN_FILES = 6
MAX_FILES = 12

# A 12-megapixel HEIC is around 3 MB and a JPEG of the same shot around 8 MB.
# 25 leaves room for a large JPEG without letting an accidental video through.
MAX_FILE_BYTES = 25 * 1024 * 1024

# Localhost plus the three private ranges, any port. Wider than one fixed
# origin because the phone reaches this over the LAN and the host's address is
# assigned by the router, but not `*` — this is an unauthenticated service
# holding a GPU.
LAN_ORIGIN = re.compile(
    r"^http://("
    r"localhost|127\.0\.0\.1|\[::1\]|"
    r"192\.168\.\d{1,3}\.\d{1,3}|"
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r")(:\d+)?$"
).pattern

app = FastAPI(title="cubit measurement service")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=os.environ.get("CUBIT_ALLOWED_ORIGINS", LAN_ORIGIN),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── artifact map ──
#
# The names on the left are what the web app asks for; the paths on the right
# are where the stages actually write. Keeping the mapping here means the UI
# does not encode the pipeline's directory layout, and a stage that moves its
# output breaks one line rather than several components.
#
# Each entry is a list of candidates, best first. Stage 5 makes a mesh
# watertight and stage 4 only reconstructs it, so a run that stopped early
# still serves something viewable.
ARTIFACTS = {
    "leg_mesh.ply":     ["05_watertight/mesh/leg_cut.ply",
                         "04_recon/mesh/leg_cut_recon.ply"],
    "box_mesh.ply":     ["05_watertight/mesh/box.ply",
                         "04_recon/mesh/box_recon.ply"],
    "scene_mesh.ply":   ["05_watertight/mesh/scene_colour.ply",
                         "05_watertight/mesh/scene.ply",
                         "04_recon/mesh/scene_recon.ply"],
    "leg_no_cut.ply":   ["03_clean/objects/leg_no_cut.ply"],
    "volumes.csv":      ["06_volume/volumes.csv"],
    # The levelled file, never the other one: cutting_line.json is written
    # before levelling while every mesh is written after, so the two do not
    # share a frame. The review copy wins when it exists, so a reload after an
    # edit shows the planes the volume was actually computed from.
    "cutting_line.json": ["03_clean/debug/cutting_line_review.json",
                          "03_clean/debug/cutting_line_levelled.json"],
    "prep/framing.json": ["00_prep/framing.json"],
}


class RunRequest(BaseModel):
    # True refuses to measure a set stage 0 rejected. False is the user
    # overruling that after seeing the overlays — the pipeline still runs, but
    # VGGT does its own centre crop on the frames we could not frame.
    strict: bool = True


class Plane(BaseModel):
    centroid: tuple[float, float, float]
    normal: tuple[float, float, float]
    npts: int = 0


class RecutRequest(BaseModel):
    planes: list[Plane] = []


@app.get("/health")
def health():
    return {"ok": True, "queue": REGISTRY.depth()}


@app.post("/jobs")
async def create_job(files: list[UploadFile] = File(...)):
    """Accept a photo set and run stage 0 on it."""
    if not (MIN_FILES <= len(files) <= MAX_FILES):
        raise HTTPException(
            400,
            f"{len(files)} photos submitted; {MIN_FILES}-{MAX_FILES} are needed. "
            "Orbit the subject, keeping the reference cube in every shot.")

    job = Job(id=uuid.uuid4().hex[:12], frames=len(files))
    inputs = os.path.join(job.dir, "input")
    os.makedirs(inputs, exist_ok=True)

    for i, upload in enumerate(files):
        name = os.path.basename(upload.filename or f"img_{i:02d}.jpg")
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            raise HTTPException(400, f"{name}: {ext or 'no extension'} is not a "
                                     f"supported image type")
        data = await upload.read()
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(
                400, f"{name} is {len(data) / 1e6:.0f} MB; the limit is "
                     f"{MAX_FILE_BYTES // (1024 * 1024)} MB per photo")
        if not data:
            raise HTTPException(400, f"{name} is empty")
        # Numbered so the order the user picked survives — stage 0 reports
        # rejections by position, and "re-take photo 3" has to mean the third
        # one they submitted.
        with open(os.path.join(inputs, f"{i:02d}_{name}"), "wb") as f:
            f.write(data)

    REGISTRY.put(job)
    # Stage 0 runs lenient here so it always finishes and always writes its
    # report. The gate is not being skipped — it moves to /run, where the user
    # has seen the overlays and can answer for the decision.
    REGISTRY.submit(job, "0",
                    ["-i", os.path.join(job.dir, "input"),
                     "--continue-on-rejected"],
                    running_state="prep", done_state="awaiting-framing")
    return {"job_id": job.id, "frames": len(files)}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = _lookup(job_id)
    return {
        "job_id": job.id,
        "state": job.state,
        "stage": job.stage,
        "frames": job.frames,
        "measured": job.measured,
        "error": job.error,
        "log": job.log,
        "queue": REGISTRY.depth(),
        "framing": job.framing(),
    }


@app.post("/jobs/{job_id}/run")
def run_job(job_id: str, req: RunRequest):
    """Measure a job whose framing the user has seen."""
    job = _lookup(job_id)
    if job.state not in ("awaiting-framing", "awaiting-cut", "done", "failed"):
        raise HTTPException(409, f"job is {job.state}")

    report = job.framing()
    if report is None:
        raise HTTPException(409, "stage 0 has not produced a framing report")
    if req.strict and not report.get("all_passed"):
        raise HTTPException(
            409,
            f"{report.get('accepted')} of {report.get('submitted')} frames "
            f"passed framing. Re-take the rejected photos, or continue anyway "
            f"— VGGT will centre-crop the ones we could not frame.")

    # --no-cut is the whole point of this pass. Stage 3 detects the cutting
    # planes and publishes them, but does not apply them: the uncut limb is
    # what gets measured. Applying the detected cut here would produce a volume
    # from a cut nobody has approved, and the review that follows would then be
    # asking the user to confirm a number already on screen.
    REGISTRY.submit(job, "1-6",
                    ["-i", os.path.join(job.dir, "00_prep", "images"), "--no-cut"],
                    running_state="running", done_state="awaiting-cut")
    job.measured = True
    return {"job_id": job.id, "state": "queued"}


@app.post("/jobs/{job_id}/recut")
def recut(job_id: str, req: RecutRequest):
    """Re-measure with cutting planes the user placed.

    Stages 3-6 only. Stage 1's predictions.npz is already on disk and is by far
    the expensive part, so an edit costs seconds rather than a whole run.
    """
    job = _lookup(job_id)
    if not job.measured or job.state not in ("awaiting-cut", "done", "failed"):
        raise HTTPException(409, "the job has not been measured yet")
    if len(req.planes) > 2:
        raise HTTPException(
            400, "at most two planes — one keeps what is below it, two keep "
                 "what lies between them, and a third can only contradict one "
                 "of those")

    path = os.path.join(job.dir, "planes.json")
    with open(path, "w") as f:
        import json
        json.dump({"markers": [p.model_dump() for p in req.planes],
                   "space": "levelled"}, f, indent=2)

    # --cut-only reuses everything the measuring pass already computed: the
    # levelled, filtered limb is on disk exactly as the cut operates on it, and
    # nothing before the cut depends on where the planes go.
    REGISTRY.submit(job, "3-6", ["--cut-only", "--planes", path],
                    running_state="running", done_state="done")
    return {"job_id": job.id, "state": "queued", "planes": len(req.planes)}


@app.get("/jobs/{job_id}/files/{path:path}")
def get_file(job_id: str, path: str):
    job = _lookup(job_id)
    root = os.path.realpath(job.dir)

    candidates = ARTIFACTS.get(path)
    if candidates is None and path.startswith("prep/"):
        # The two kinds of image the framing report names: the annotated
        # overlay ("overlay") and the clean crop handed to VGGT ("cropped").
        # They live in different directories, so try both rather than making
        # the report carry paths.
        name = os.path.basename(path)
        candidates = ["00_prep/for_debug/" + name, "00_prep/images/" + name]
    if candidates is None:
        raise HTTPException(404, f"unknown artifact: {path}")

    for rel in candidates:
        full = os.path.realpath(os.path.join(root, rel))
        if not full.startswith(root + os.sep):
            raise HTTPException(400, "path escapes the job directory")
        if os.path.isfile(full):
            # These files are rewritten in place by a recut, so a cached copy
            # would show the previous cut's mesh next to the new volume.
            return FileResponse(full, headers={"Cache-Control": "no-store"})

    raise HTTPException(404, f"{path} has not been produced yet")


def _lookup(job_id: str) -> Job:
    job = REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job


# Running from anywhere else would put `work/` in the wrong place and leave
# stagerun.py unable to find the pipeline package.
os.chdir(PROJECT_ROOT)
