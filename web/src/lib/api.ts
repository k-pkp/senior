import type { CutPlane, FramingReport, SampleDataset } from "./types";
import type { CameraData, LevellingData } from "./imagecamera";

/** Port the compute service listens on. */
const API_PORT = 8000;

/** Where the compute service is, from the point of view of whoever is looking.
 *
 * Defaulting to the page's own host rather than to localhost is what lets one
 * build serve both the laptop and a phone on the same wifi: the phone loaded
 * this page from some address, so that address is by construction reachable
 * from the phone, whereas `localhost` would send it back to itself.
 *
 * NEXT_PUBLIC_API_URL overrides it when the service lives somewhere else.
 * Empty during prerender, which is harmless — every call happens in the
 * browser, after this module is evaluated there. */
function defaultApi(): string {
  if (typeof window === "undefined") return "";
  return `${window.location.protocol}//${window.location.hostname}:${API_PORT}`;
}

export const API = process.env.NEXT_PUBLIC_API_URL || defaultApi();

/** What the service reports about one job.
 *
 * `stage` is which pipeline stage is running right now, straight from the
 * service — it launched the subprocess, so this is observed rather than
 * estimated. That is the whole point of polling instead of running a timer. */
export interface JobStatus {
  job_id: string;
  state:
    | "queued"
    | "prep"
    | "awaiting-framing"
    | "running"
    /** Measured uncut; the detected planes are published and waiting to be
     *  approved or moved before the cut is applied. */
    | "awaiting-cut"
    | "done"
    | "failed";
  stage: number;
  frames: number;
  measured: boolean;
  error: string | null;
  /** Last lines of the failed stage's output. Empty unless state is failed. */
  log: string[];
  queue: number;
  framing: FramingReport | null;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    // FastAPI puts the human-readable reason in `detail`, and those messages
    // are written for the person holding the camera — surface them verbatim
    // rather than replacing them with a status code.
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* not JSON; the status is all we have */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

/** Upload a photo set. Resolves once the service has the files; stage 0 then
 *  runs in the background, so poll `getJob` for its verdict. */
export function createJob(
  files: File[],
  onProgress?: (fraction: number) => void,
): Promise<{ job_id: string; frames: number }> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f, f.name));

  // XHR rather than fetch: a phone uploading a dozen HEICs over wifi is a slow
  // enough operation that a progress bar is the difference between "working"
  // and "frozen", and fetch still cannot report upload progress.
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API}/jobs`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let body: any = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* fall through to the status-only message */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body?.detail ?? `upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("could not reach the compute service"));
    xhr.send(form);
  });
}

// Fetches one job's current status, bypassing the browser cache.
export async function getJob(id: string): Promise<JobStatus> {
  return json<JobStatus>(await fetch(`${API}/jobs/${id}`, { cache: "no-store" }));
}

/** Measure a job. `strict` false is the user overruling the framing gate after
 *  seeing which photos failed and why. */
export async function runJob(id: string, strict: boolean) {
  return json(
    await fetch(`${API}/jobs/${id}/run`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ strict }),
    }),
  );
}

/** Re-measure with planes the user placed. Stages 3-6 only — stage 1's
 *  predictions are already on disk, which is why an edit costs seconds. */
export async function recut(id: string, planes: CutPlane[]) {
  return json(
    await fetch(`${API}/jobs/${id}/recut`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        planes: planes.map((p) => ({
          centroid: p.centroid,
          normal: p.normal,
          npts: p.npts,
        })),
      }),
    }),
  );
}

/** Dress a job up as a dataset.
 *
 * `SampleDataset` is nothing but a bag of URLs, so a live job differs from a
 * shipped sample only in where those URLs point. Result and Review therefore
 * need no knowledge of jobs at all — they keep loading a dataset. */
export function jobDataset(id: string, frames: number): SampleDataset {
  // Builds the artifact URL for one named file of this job.
  const f = (name: string) => `${API}/jobs/${id}/files/${name}`;
  return {
    id,
    label: "Your upload",
    subject: `${frames} photos, measured on this machine`,
    nominalMl: null, // an uploaded object has no known truth — never invent one
    frames,
    meshes: {
      leg: f("leg_mesh.ply"),
      box: f("box_mesh.ply"),
      scene: f("scene_mesh.ply"),
      legNoCut: f("leg_no_cut.ply"),
    },
    volumesCsv: f("volumes.csv"),
    cuttingLine: f("cutting_line.json"),
  };
}

// Base URL for this job's Stage 0 prep images.
export function jobPrepBase(id: string) {
  return `${API}/jobs/${id}/files/prep`;
}

/** URLs of the frames VGGT consumed, in order — Stage 0's 518px crops. */
export function jobFrameUrls(id: string, frames: number): string[] {
  const urls: string[] = [];
  for (let index = 0; index < frames; index++) {
    const name = `frame_${String(index).padStart(2, "0")}.png`;
    urls.push(`${API}/jobs/${id}/files/prep/${name}`);
  }
  return urls;
}

/**
 * VGGT's per-frame cameras for this job, or null if the run has not produced
 * them. Only Stage 1 writes them, so they are absent until inference has run.
 */
export async function loadCameras(id: string): Promise<CameraData | null> {
  try {
    const response = await fetch(`${API}/jobs/${id}/files/cameras.json`);
    if (!response.ok) return null;
    return (await response.json()) as CameraData;
  } catch {
    return null;
  }
}

/** Stage 3's levelling rotation for this job, or null if it has not run. */
export async function loadLevelling(id: string): Promise<LevellingData | null> {
  try {
    const response = await fetch(`${API}/jobs/${id}/files/levelling.json`);
    if (!response.ok) return null;
    return (await response.json()) as LevellingData;
  } catch {
    return null;
  }
}
