# Running the web app end to end

The site has two halves. `web/` is a page; it draws meshes and numbers but
computes nothing. `service/` is the half with the GPU: it accepts photos, runs
`stagerun.py`, and serves what the stages wrote. Without it the site still
works, but only on the two precomputed samples in `web/public/samples/`.

Nothing here can move into the browser. Stage 1 is VGGT-1B and needs about 8 GB
of VRAM; stage 0 additionally loads GroundingDINO and SAM. "End to end in the
web app" means the page drives a GPU on the local network.

## Quick start

```bash
./serve.sh
```

That builds the site if it has not been built, starts both halves bound to
`0.0.0.0`, and prints the addresses. Ctrl-C stops both.

Manually, if you would rather see the two processes separately:

```bash
# terminal 1 — the compute service, from the repo root (it needs work/)
~/miniconda3/envs/senior/bin/python -m uvicorn service.app:app --host 0.0.0.0 --port 8000

# terminal 2 — the site
cd web && npm run build && npx next start -H 0.0.0.0 -p 3111
```

The page finds the service by taking its own hostname and port 8000, so opening
it at `localhost:3111` looks for `localhost:8000` and opening it at
`192.168.1.34:3111` looks for `192.168.1.34:8000`. One build therefore serves
both the laptop and a phone. Set `NEXT_PUBLIC_API_URL` in `web/.env.local` only
if the service lives somewhere else.

## Letting a phone reach it

WSL2 sits behind its own NAT, so binding to `0.0.0.0` inside Linux does not put
the ports on the wifi. One of the following is needed, on Windows.

**Durable — mirrored networking.** Put this in `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

then run `wsl --shutdown` and start WSL again. From then on WSL shares the
Windows host's addresses and nothing else is required. This closes every
running WSL session, so do it before you start working rather than during.

**Immediate — port forwarding.** Works without restarting WSL, from an
administrator PowerShell. `serve.sh` prints these lines with the current
address already filled in:

```powershell
netsh interface portproxy add v4tov4 listenport=3111 listenaddress=0.0.0.0 connectport=3111 connectaddress=<wsl-ip>
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=<wsl-ip>
New-NetFirewallRule -DisplayName "cubit" -Direction Inbound -Protocol TCP -LocalPort 3111,8000 -Action Allow
```

The catch is that `<wsl-ip>` changes when WSL restarts, so the two `portproxy`
lines have to be re-pointed after a reboot (`netsh interface portproxy reset`
clears the old ones).

Either way, the phone opens `http://<windows-lan-ip>:3111`. Find that address
with `ipconfig` on Windows — it is the one on the same `192.168.x` network as
the phone, not the VirtualBox or Tailscale adapter.

## What happens when you upload

| Step | What runs | Roughly |
|---|---|---|
| Upload | Files land in `work/<job>/input/` | seconds |
| Framing check | Stage 0 alone: cube, limb and band per photo | 25 s |
| *you decide* | Continue, or re-take the rejected photos | — |
| Reconstructing | Stages 1-6 with `--no-cut`: the scene is reconstructed, the cutting plane is detected but **not applied**, and only the reference cube is measured | 65 s |
| Review | Nothing — the detected cut is shown for checking | — |
| *you decide* | Drag the plane, or accept it | — |
| Applying your cut | Stage 3 `--cut-only` plus 4-6. Stages 1 and 2 are not repeated, and the cube's mesh is reused | 10 s |

Nothing is computed twice. The limb is not reconstructed at all until the cut is
confirmed, so no volume is ever produced from a cut nobody agreed to. Applying the
*detected* plane through this split reproduces an unsplit run to the last digit
(`1126.1505024938106 cm³`), so the split changed when the cut happens, not what it
computes.

The Review step is the second gate. Until you confirm, the Result screen says
"Not measured yet" rather than showing a number — the object genuinely has no
volume until its extent is agreed.

The framing check is the first gate, not a progress step. A photo that clips the
reference cube corrupts the scale of every number the run goes on to report,
and does so with no visible sign, so the pipeline stops and says which photo to
re-take. "Measure anyway" exists because a pose is sometimes simply not
available — it hands the rejected frames to VGGT uncropped, which centre-crops
them itself, and whatever that crop removes is lost.

Each problem frame carries a severity, shown as a chip on its card:

| condition | verdict | what the frame is told |
|---|---|---|
| everything found and framed | **pass** | — |
| band missing | **warning** | marker missing — the cut must be placed by hand |
| band found but clipped | **warning** | marker out of window — the suggested cut may be off |
| cube found but clipped | **warning** | cube out of window — VGGT will centre-crop instead |
| cube not detected | **reject** | cube missing — the scale cannot be recovered |
| nothing detected | **reject** | nothing detected — no cube and no marker |
| file cannot be decoded | **reject** | file unreadable — cannot be decoded |

**A warning is a usable frame.** The distinction is what a defect *costs*. The
reference cube sets the scale of every number, so if it is not detected at all
there is nothing to recover from — that is a reject. Everything else degrades the
result without making it impossible: a clipped cube falls back to VGGT's own
centre crop, and a missing band only means the cut is placed by a person in the
review step, which it is anyway. Only rejects stop the run.

Which object left the window is not reported, because the window is not adjustable —
it is the largest square the photo allows. The remedy is the same either way: step
back and re-take.

## The API

```
GET  /health                     liveness; the page probes this
POST /jobs                       multipart, 6-12 photos -> {job_id}
GET  /jobs/{id}                  {state, stage, framing, error, log}
POST /jobs/{id}/run              {strict}  — stages 1-6 with --no-cut
POST /jobs/{id}/recut            {planes}  — stage 3 --cut-only, then 4-6
GET  /jobs/{id}/files/{name}     leg_mesh.ply, volumes.csv, prep/framing.json …
```

A job is a `work/<job_id>/` directory — the same layout every manual run uses,
so anything the service produced can be inspected or re-run from a terminal:

```bash
python stagerun.py 3-6 --name <job_id>                       # full stage 3, then measure
python stagerun.py 3 --name <job_id> --no-cut                # detect the plane, do not cut
python stagerun.py 3 --name <job_id> --cut-only --planes p.json   # apply a chosen cut
```

Each stage runs as its own subprocess. That is what lets the service report
which stage is in flight without parsing logs, and it guarantees the GPU memory
is released between stages rather than accumulating across jobs.

## Limits worth knowing before a demo

- **One job at a time.** There is one GPU, so uploads queue. The processing
  screen says so.
- **No authentication.** CORS is restricted to localhost and private address
  ranges, but anyone on the wifi who can reach the port can queue a job. Do not
  put this on a public network or a tunnel as it stands.
- **Jobs are never cleaned up.** `work/` grows by roughly 40 MB per run. Delete
  old job directories by hand.
- **A restarted service abandons running jobs.** Their subprocesses die with
  it; finished jobs survive, because each one mirrors its state to
  `work/<job>/job.json`.
- **Uploads are capped** at 12 photos and 25 MB each.
- **Stage 6 is currently main's version**, reverted pending review by its author
  (see `docs/stage06_experiments.md`). It writes different CSV columns
  (`ext_x/ext_y/ext_z/size_*_cm`) from the parked one
  (`obb_a/obb_b/obb_c/height_cm`). The viewer reads both, so runs display either
  way — but the *scale* is derived differently in each case, and under main's
  method the reference cube reports exactly 2744.00 cm³ by construction, which is
  an identity rather than a measurement. Treat volumes as provisional until
  Stage 6 settles.
- **The two shipped samples were measured with the parked Stage 6**, so they do
  not match what a live job now produces. `small_leg` ships as 1071.46 cm³ with
  the reference at 2692.89; uploading the same six photos today gives 1081.94 cm³
  with the reference at 2744.00. Both are the same capture — the gap is the two
  scale derivations, not a regression. The samples are kept as they are because
  regenerating them would lose the only worked example of the parked schema the
  viewer has to keep supporting.
