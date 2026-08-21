"""Job state and the single-GPU worker.

A job is nothing more than a `work/<job_id>/` directory — the same layout
`stagerun.py` already uses for every manual run. That is deliberate: it makes
the existing CLI the backend rather than a second implementation of it, so a
job can be inspected, re-run or debugged from a terminal with the exact
commands used everywhere else in this project.

Stages run as subprocesses. Two reasons, both learned the hard way:

  * The service knows which stage is running because it is the one that
    launched it — no log parsing, no guessing from timings.
  * A process that exits releases its VRAM. A model held warm across jobs
    saves about ten seconds of load and risks leaking several gigabytes, which
    makes the *next* run mysteriously slow rather than failing outright.

One GPU means one job at a time, so there is a single worker thread and a
plain queue. Nothing here needs Celery.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(PROJECT_ROOT, "work")

# The interpreter running the service is the one that has torch, VGGT and the
# rest of the pipeline's dependencies. Hard-coding "python" would find whatever
# is first on PATH, which on this machine is not the same environment.
PYTHON = sys.executable

# Last lines of a failed stage's output kept for the UI. Enough to carry a
# traceback, short enough to put in a JSON response.
LOG_TAIL = 40

# States, in the order a job passes through them.
#   queued           accepted, waiting for the worker
#   prep             stage 0 running
#   awaiting-framing stage 0 done, waiting for the user to accept the framing
#   running          stages 1-6 (or 3-6 on a recut)
#   awaiting-cut     measured UNCUT; the detected planes are published and the
#                    user has to approve or move them before the cut is applied
#   done             a confirmed cut has been measured
#   failed           a stage exited non-zero; `error` and `log` say which


@dataclass
class Job:
    id: str
    state: str = "queued"
    stage: int = 0
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    frames: int = 0
    error: str | None = None
    log: list[str] = field(default_factory=list)
    # True once stages 1-6 have produced a volume, so a recut knows stage 1's
    # predictions.npz is on disk and does not have to be recomputed.
    measured: bool = False

    @property
    def dir(self) -> str:
        return os.path.join(WORK, self.id)

    def touch(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.updated = time.time()
        self.save()

    def save(self):
        os.makedirs(self.dir, exist_ok=True)
        tmp = os.path.join(self.dir, "job.json.tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=2)
        os.replace(tmp, os.path.join(self.dir, "job.json"))

    def framing(self):
        """Stage 0's verdict on each submitted photo, once it has one."""
        path = os.path.join(self.dir, "00_prep", "framing.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None


class Registry:
    """Jobs in memory, mirrored to disk so a restart keeps finished runs.

    Only the mirror is authoritative across restarts; a job that was mid-flight
    when the service stopped comes back as failed rather than being resumed,
    because its subprocess is gone and its stage directories are half-written.
    """

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._run_forever, daemon=True)
        self._worker.start()
        self._restore()

    # ── lookup ──

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        # Not in memory: it may predate a restart.
        path = os.path.join(WORK, job_id, "job.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        job = Job(**{k: v for k, v in data.items()
                     if k in Job.__dataclass_fields__})
        with self._lock:
            self._jobs.setdefault(job_id, job)
        return job

    def put(self, job: Job):
        with self._lock:
            self._jobs[job.id] = job
        job.save()

    def _restore(self):
        if not os.path.isdir(WORK):
            return
        for name in os.listdir(WORK):
            path = os.path.join(WORK, name, "job.json")
            if not os.path.exists(path):
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            job = Job(**{k: v for k, v in data.items()
                         if k in Job.__dataclass_fields__})
            if job.state in ("queued", "prep", "running"):
                # Its subprocess died with the service.
                job.state = "failed"
                job.error = "the service restarted while this job was running"
            self._jobs[job.id] = job

    # ── scheduling ──

    def submit(self, job: Job, stages: str, extra: list[str],
               running_state: str, done_state: str):
        job.touch(state="queued", error=None, log=[])
        self._queue.put((job.id, stages, extra, running_state, done_state))

    def depth(self) -> int:
        return self._queue.qsize()

    def _run_forever(self):
        while True:
            item = self._queue.get()
            try:
                self._run_one(*item)
            except Exception as exc:  # a crash here must not kill the worker
                job = self.get(item[0])
                if job is not None:
                    job.touch(state="failed", error=repr(exc))
            finally:
                self._queue.task_done()

    def _run_one(self, job_id, stages, extra, running_state, done_state):
        job = self.get(job_id)
        if job is None:
            return
        first = int(stages.split("-")[0])
        job.touch(state=running_state, stage=first)

        for n in _expand(stages):
            job.touch(stage=n)
            code, tail = _run_stage(job, n, extra)
            if code != 0:
                job.touch(state="failed",
                          error=f"stage {n} exited with code {code}",
                          log=tail)
                return

        job.touch(state=done_state, log=[])


def _expand(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(spec)]


def _run_stage(job: Job, n: int, extra: list[str]):
    """Run one stage as a subprocess. Returns (exit code, last log lines)."""
    cmd = [PYTHON, "stagerun.py", str(n), "--name", job.id] + list(extra)
    log_path = os.path.join(job.dir, f"stage{n}.log")
    os.makedirs(job.dir, exist_ok=True)

    tail: list[str] = []
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            tail.append(line.rstrip("\n"))
            if len(tail) > LOG_TAIL:
                tail.pop(0)
        code = proc.wait()
    return code, tail


REGISTRY = Registry()
