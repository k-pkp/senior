#!/usr/bin/env python3
"""Run pipeline stages one at a time, with inspectable output per stage.

Each stage reads the previous stage's artifacts from work/<input_name>/ and
writes its own, so a stage can be re-run and inspected without repeating the
expensive ones (inference in particular).

    python3 stagerun.py 1 -i inputs/small_leg          # inference only
    python3 stagerun.py 2 -i inputs/small_leg          # pointcloud, reuses stage 1
    python3 stagerun.py 1-3 -i inputs/small_leg        # range
    python3 stagerun.py 1 -i inputs/small_leg --force  # ignore cache

Layout:
    work/<name>/01_inference/   predictions.npz, input_frames.png, summary.txt
    work/<name>/02_pointcloud/  points.ply, summary.txt
    work/<name>/03_clean/       clean clouds, summary.txt
    work/<name>/04_recon/       meshes
    work/<name>/05_watertight/  watertight meshes
    work/<name>/06_volume/      volumes.csv
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORK = "work"
STAGE_DIRS = {
    0: "00_prep",
    1: "01_inference",
    2: "02_pointcloud",
    3: "03_clean",
    4: "04_recon",
    5: "05_watertight",
    6: "06_volume",
}


def stage_dir(name, n, create=True):
    """Returns the directory holding stage n's output for this run, creating it by default."""
    d = os.path.join(WORK, name, STAGE_DIRS[n])
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def src_dir(args, name, n):
    """Directory to read stage n from — --src run if given, else this run."""
    return stage_dir(args.src or name, n, create=False)


def _write_summary(d, lines):
    """Writes the summary lines to summary.txt in the stage directory and prints them."""
    path = os.path.join(d, "summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n  -> {path}")


def multiview_stats(preds, conf_pct=50, sample=40000, seed=42):
    """Geometric self-consistency: reproject each 3D point into every other view.

    For point p seen in view i, its depth in view j is known two ways — the
    geometric depth from transforming p by extrinsic[j], and the depth head's
    own prediction at that pixel. Disagreement means the model is not
    self-consistent about where the surface is.

    Agreement is counted at >=3 views: two views agreeing can both be wrong in
    the same way (they share the same depth head), whereas three corroborating
    is a meaningfully stronger claim about the surface.

    Ground-truth free, so it ranks variants but cannot prove correctness — a
    model can be consistently wrong.
    """
    wp = preds["world_points"]
    depth = preds["depth"]
    if depth.ndim == 4:
        depth = depth[..., 0]
    extr, intr = preds["extrinsic"], preds["intrinsic"]
    conf = preds["world_points_conf"]
    S, H, W = wp.shape[:3]
    if S <= 1:
        return None

    thr = np.percentile(conf, conf_pct)
    rng = np.random.default_rng(seed)
    rel_all, agree_counts = [], []

    for i in range(S):
        m = (conf[i] >= thr).reshape(-1)
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        if len(idx) > sample:
            idx = rng.choice(idx, sample, replace=False)
        pts = wp[i].reshape(-1, 3)[idx].astype(np.float64)
        agree = np.zeros(len(pts), dtype=np.int32)

        for j in range(S):
            if i == j:
                continue
            cam = np.hstack([pts, np.ones((len(pts), 1))]) @ extr[j].T
            dz = cam[:, 2]
            fx, fy = float(intr[j, 0, 0]), float(intr[j, 1, 1])
            cx, cy = float(intr[j, 0, 2]), float(intr[j, 1, 2])
            ok = dz > 1e-6
            safe = np.where(ok, dz, np.inf)
            u = np.rint(fx * cam[:, 0] / safe + cx)
            v = np.rint(fy * cam[:, 1] / safe + cy)
            u = np.nan_to_num(u, nan=-1, posinf=-1, neginf=-1).astype(np.int32)
            v = np.nan_to_num(v, nan=-1, posinf=-1, neginf=-1).astype(np.int32)
            ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not ok.any():
                continue
            d_pred = np.full(len(pts), np.inf)
            d_pred[ok] = depth[j, v[ok], u[ok]]
            rel = np.abs(d_pred - dz) / (np.abs(dz) + 1e-8)
            rel_all.append(rel[ok & np.isfinite(rel)])
            agree += ((rel < 0.05) & ok).astype(np.int32)

        agree_counts.append(agree)

    if not rel_all:
        return None
    rel = np.concatenate(rel_all)
    ag = np.concatenate(agree_counts)
    return {
        "median_rel": float(np.median(rel)),
        "p90_rel": float(np.percentile(rel, 90)),
        "frac_3views_5pct": float((ag >= 3).mean()),
        "frac_2views_at_1pct": None,
        "mean_agreeing_views": float(ag.mean()),
        "n_pairs": int(len(rel)),
    }


def border_contact(preds, conf_pct=50, band=8):
    """Fraction of confident geometry touching the image border.

    High values mean the subject runs off the edge of frame — the signature of
    a preprocessing crop cutting the object, which no downstream stage can undo.
    """
    conf = preds["world_points_conf"]
    S, H, W = conf.shape
    thr = np.percentile(conf, conf_pct)
    edge = np.zeros((H, W), dtype=bool)
    edge[:band, :] = edge[-band:, :] = True
    edge[:, :band] = edge[:, -band:] = True
    hi = conf >= thr
    tot = hi.sum()
    if tot == 0:
        return None
    return {
        "frac_conf_on_border": float((hi & edge[None]).sum() / tot),
        "top_edge": float((hi[:, :band, :]).sum() / tot),
        "bottom_edge": float((hi[:, -band:, :]).sum() / tot),
    }


def floor_planarity(path, band=0.03, cm_per_unit=None):
    """RMS residual of the floor to a fitted plane, in mm.

    The floor is real ceramic tile, so it is genuinely flat — its residual is
    VGGT's surface-localisation noise measured against physical truth, not a
    self-consistency proxy.
    """
    import open3d as o3d
    p = o3d.io.read_point_cloud(path)
    pts = np.asarray(p.points)
    if len(pts) < 5000:
        return None
    sub = pts[::max(1, len(pts) // 200000)]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(sub)
    model, _ = pc.segment_plane(distance_threshold=0.01, ransac_n=3,
                                num_iterations=2000)
    n, d = np.array(model[:3]), model[3]
    signed = pts @ n + d
    m = np.abs(signed) < band
    if m.sum() < 1000:
        return None
    r = signed[m] - np.median(signed[m])
    # 14 cm ArUco cube measures ~0.265 units across these scenes
    cm = cm_per_unit if cm_per_unit else 14 / 0.265
    return {
        "rms_mm": float(np.sqrt((r ** 2).mean()) * cm * 10),
        "p95_mm": float(np.percentile(np.abs(r), 95) * cm * 10),
        "n_floor": int(m.sum()),
        "frac_floor": float(m.mean()),
    }


def _pcd_stats(path):
    """Point count, extents, and colour presence for a PLY."""
    import open3d as o3d
    p = o3d.io.read_point_cloud(path)
    pts = np.asarray(p.points)
    if len(pts) == 0:
        return f"{os.path.basename(path):<24} EMPTY"
    e = pts.max(0) - pts.min(0)
    return (f"{os.path.basename(path):<24} {len(pts):>8,} pts  "
            f"extent=({e[0]:.4f},{e[1]:.4f},{e[2]:.4f})  "
            f"colours={'yes' if p.has_colors() else 'NO'}")


def _colorize(a, lo_pct=2, hi_pct=98):
    """Percentile-normalised heatmap for a 2D float array."""
    lo, hi = np.percentile(a, lo_pct), np.percentile(a, hi_pct)
    x = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    try:
        import matplotlib.cm as cm
        return (cm.turbo(x)[..., :3] * 255).astype(np.uint8)
    except Exception:
        return (np.stack([x] * 3, -1) * 255).astype(np.uint8)


def dump_raw(preds, out_dir):
    """Materialise every Stage 1 output as something inspectable.

    predictions.npz holds all nine arrays but is opaque; this writes each one as
    an image, PLY or JSON so it can be judged directly rather than through
    aggregate metrics.
    """
    import json
    from PIL import Image
    import trimesh

    raw = os.path.join(out_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    written = []

    imgs = preds["images"]                       # (S,3,H,W)
    S, _, H, W = imgs.shape
    depth = preds["depth"]
    depth = depth[..., 0] if depth.ndim == 4 else depth
    dconf = preds["depth_conf"]
    wconf = preds["world_points_conf"]

    for sub in ("images", "depth", "depth_conf", "world_points_conf"):
        os.makedirs(os.path.join(raw, sub), exist_ok=True)

    for i in range(S):
        rgb = (np.clip(imgs[i].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(f"{raw}/images/frame_{i:02d}.png")
        Image.fromarray(_colorize(depth[i])).save(f"{raw}/depth/frame_{i:02d}.png")
        Image.fromarray(_colorize(dconf[i])).save(f"{raw}/depth_conf/frame_{i:02d}.png")
        Image.fromarray(_colorize(wconf[i])).save(f"{raw}/world_points_conf/frame_{i:02d}.png")
    written += [f"images/           {S} PNG  — exactly what VGGT received",
                f"depth/            {S} PNG  — depth head, turbo colormap",
                f"depth_conf/       {S} PNG  — depth confidence",
                f"world_points_conf/{S} PNG  — pointmap confidence"]

    # Unfiltered clouds — shows what Stage 2's confidence filter throws away.
    cols = (np.clip(imgs.transpose(0, 2, 3, 1), 0, 1) * 255).astype(np.uint8).reshape(-1, 3)
    for key, fname in (("world_points", "cloud_pointmap.ply"),
                       ("world_points_from_depth", "cloud_from_depth.ply")):
        if key not in preds:
            continue
        pts = preds[key].reshape(-1, 3).astype(np.float32)
        trimesh.PointCloud(pts, colors=cols).export(os.path.join(raw, fname))
        written.append(f"{fname:<18} {len(pts):,} pts — RAW, no filtering")

    cams = {
        "pose_enc": preds["pose_enc"].tolist(),
        "extrinsic": preds["extrinsic"].tolist(),
        "intrinsic": preds["intrinsic"].tolist(),
        "note": ("extrinsic is 3x4 cam-from-world (OpenCV convention); "
                 "intrinsic is 3x3 pixel units for this resolution"),
    }
    with open(os.path.join(raw, "cameras.json"), "w") as f:
        json.dump(cams, f, indent=2)
    written.append("cameras.json       per-frame extrinsic / intrinsic / pose_enc")

    fx = [float(preds["intrinsic"][i, 0, 0]) for i in range(S)]
    cx = [float(preds["intrinsic"][i, 0, 2]) for i in range(S)]
    manifest = [
        "# Stage 1 raw outputs",
        "",
        "Everything VGGT produces, materialised. Generated from predictions.npz;",
        "delete this folder and re-run stage 1 to regenerate (no inference needed).",
        "",
    ] + [f"- {w}" for w in written] + [
        "",
        "## What is and is not used downstream",
        "",
        "| output | used by | note |",
        "|---|---|---|",
        "| `world_points` + conf | Stage 2 | the only 3D source in the live pipeline |",
        "| `depth` + `depth_conf` | nothing | second independent estimate, unused |",
        "| `world_points_from_depth` | nothing | computed every run, never read |",
        "| `extrinsic` / `intrinsic` | nothing | needed for any reprojection/consistency work |",
        "| `images` | Stage 2 (colours) | |",
        "",
        "## Camera sanity",
        "",
        f"- focal length fx per frame: {', '.join(f'{v:.1f}' for v in fx)}",
        f"- principal point cx: {', '.join(f'{v:.1f}' for v in cx)}  (image centre = {W/2:.1f})",
        "",
        "Focal lengths should be near-identical across frames for one phone camera;",
        "large spread means the pose head is unsure and downstream scale will drift.",
    ]
    with open(os.path.join(raw, "MANIFEST.md"), "w") as f:
        f.write("\n".join(manifest) + "\n")

    print(f"  raw dump -> {raw}/  ({len(written)} artifact groups)")
    return raw


# ── Stage 1 ────────────────────────────────────────────────────────────

def run_stage0(args, name):
    """Frame each photo around the subject before VGGT is given a square.

    Writes cropped frames that stage 1 can be pointed at with -i, rather than
    handing them over implicitly, so a prep run and a raw run stay directly
    comparable.
    """
    from pipeline.stages.prep import prepare_frames

    d = stage_dir(name, 0)
    images = os.path.join(d, "images")
    manifest = prepare_frames(args.image_folder, images,
                              band_heights=args.prep_band,
                              pad=args.prep_pad,
                              centre_on_subject=args.prep_recentre,
                              output_size=args.prep_size,
                              strict=args.prep_strict,
                              crop=args.prep_crop,
                              min_frames=args.prep_min_frames)
    frames = manifest["frames"]
    lines = [f"STAGE 0 — prep   (from {args.image_folder})", ""]
    lines.append(f"  band              : {manifest['band_cube_heights']} cube heights")
    lines.append(f"  pad               : {manifest['pad_frac']:.0%}")
    lines.append(f"  centred on        : "
                 f"{'subject' if manifest['centre_on_subject'] else 'frame'}")
    lines.append("")
    for r in frames:
        w = r["window"]
        # Take the verdict Stage 0 recorded rather than re-deriving one. This
        # line used to say REJECTED for anything that was not perfect, which
        # after the three-verdict change meant a frame the pipeline was happily
        # using was reported as refused.
        state = {"pass": "PASS", "warning": "WARNING",
                 "reject": "REJECT"}.get(r.get("verdict"),
                     "PASS" if (r["cube_ok"] and r["band_ok"]) else "REJECT")
        # The reasons come from the stage that decided them. This used to
        # re-derive its own, which drifted: the summary still said "cube not
        # contained" after Stage 0 had started distinguishing a cube that was
        # never seen from one the window cuts.
        notes = r.get("reasons") or []
        sev = r.get("severity")
        note = ("  " + "; ".join(notes)) if notes else ""
        sev = f"  [{sev}]" if sev else ""
        lines.append(f"  {r['source']:<16} {state:<9} {r['mode']:<13} "
                     f"{w[2] - w[0]}px  offset {r['offset_px']:.0f} px{sev}{note}")
    counts = {}
    for r in frames:
        counts[r.get("verdict", "?")] = counts.get(r.get("verdict", "?"), 0) + 1
    lines.append("")
    lines.append(f"  verdict: {counts.get('pass', 0)} pass, "
                 f"{counts.get('warning', 0)} warning, {counts.get('reject', 0)} reject")
    if manifest.get("warned"):
        lines.append(f"  warned  : {', '.join(manifest['warned'])}")
    if manifest.get("rejected"):
        lines.append(f"  rejected: {', '.join(manifest['rejected'])}")
    lines.append("")
    lines.append(f"  next: stagerun.py 1 -i {images} --name {name}")
    _write_summary(d, lines)
    print(f"\n  -> {images}")


def run_stage1(args, name):
    """Runs VGGT inference and caches predictions.npz, so later stages cost no inference time.

    Warns when Stage 0 produced framed images but this run is reading a
    different folder, because VGGT would then crop the frames itself and
    silently discard Stage 0's framing.
    """
    from vggt.utils.device import get_device

    # Guard the same trap when the stages are run separately. Stage 0's summary
    # tells you to re-point -i at its output; this says so again at the moment it
    # would otherwise be ignored, rather than leaving a silently unframed run.
    prep_images = os.path.join(stage_dir(name, 0, create=False), "images")
    if (os.path.isdir(prep_images)
            and os.path.abspath(prep_images) != os.path.abspath(args.image_folder)):
        print(f"  WARNING: stage 0 output exists at {prep_images}")
        print(f"           but this stage is reading {args.image_folder}")
        print(f"           VGGT will crop these frames itself — Stage 0's framing "
              f"is being ignored.")
    from pipeline.stages.inference import run_inference

    d = stage_dir(name, 1)
    npz = os.path.join(d, "predictions.npz")
    cached = os.path.exists(npz) and not args.force

    if cached:
        print(f"  cached: {npz}  (--force to redo)")
        save = dict(np.load(npz))
        preds, t = save, float("nan")
        device = "cached"
    else:
        device = get_device()
        preds, t = run_inference(args.image_folder, device, args.max_frames,
                                 preprocess_mode=args.preprocess_mode,
                                 input_res=args.input_res)
        save = {k: v for k, v in preds.items() if v is not None}
        np.savez_compressed(npz, **save)

    if args.dump:
        dump_raw(save, d)
    if cached:
        return

    # Dump what the model actually saw — catches framing/crop problems.
    try:
        from PIL import Image
        im = preds["images"]
        S = im.shape[0]
        picks = sorted(set([0, S // 2, S - 1]))
        tiles = [(np.clip(im[i].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                 for i in picks]
        Image.fromarray(np.hstack(tiles)).save(os.path.join(d, "input_frames.png"))
    except Exception as e:
        print(f"  (frame preview skipped: {e})")

    lines = [f"STAGE 1 — inference   input={args.image_folder}",
             f"  device            : {device}",
             f"  inference time    : {t:.1f}s",
             f"  frames            : {preds['images'].shape[0]}",
             f"  resolution        : {preds['images'].shape[-2]}x{preds['images'].shape[-1]}",
             f"  preprocess        : mode={args.preprocess_mode} res={args.input_res}",
             ""]
    for k, v in sorted(save.items()):
        lines.append(f"  {k:<26} {str(v.shape):<22} {v.dtype}")

    # Agreement between the two independent 3D estimates.
    wp, wpd = save.get("world_points"), save.get("world_points_from_depth")
    if wp is not None and wpd is not None:
        conf = save["world_points_conf"]
        keep = conf >= np.percentile(conf, 50)
        dist = np.linalg.norm(wp[keep] - wpd[keep], axis=-1)
        scale = np.percentile(np.linalg.norm(wp[keep] - wp[keep].mean(0), axis=-1), 90)
        lines += ["",
                  "  pointmap vs depth-unprojection disagreement (top-50% conf):",
                  f"    median {np.median(dist):.5f}   p90 {np.percentile(dist, 90):.5f}"
                  f"   (scene scale ~{scale:.3f})",
                  f"    median as % of scene: {np.median(dist)/scale*100:.2f}%"]

    mv = multiview_stats(save)
    if mv:
        lines += ["",
                  "  multi-view reprojection consistency (top-50% conf):",
                  f"    median rel depth error : {mv['median_rel']*100:.2f}%",
                  f"    p90 rel depth error    : {mv['p90_rel']*100:.2f}%",
                  f"    mean agreeing views    : {mv['mean_agreeing_views']:.2f}",
                  f"    pts with >=3 views @5% : {mv['frac_3views_5pct']*100:.1f}%"]

    bc = border_contact(save)
    if bc:
        lines += ["",
                  "  framing (confident geometry on image border):",
                  f"    any border : {bc['frac_conf_on_border']*100:.1f}%",
                  f"    top edge   : {bc['top_edge']*100:.1f}%",
                  f"    bottom edge: {bc['bottom_edge']*100:.1f}%"]

    _write_summary(d, lines)


# ── Stage 2 ────────────────────────────────────────────────────────────

def run_stage2(args, name):
    """Exports the confidence-filtered point cloud from Stage 1's predictions."""
    from pipeline.stages.pointcloud import export_ply

    src = os.path.join(src_dir(args, name, 1), "predictions.npz")
    if not os.path.exists(src):
        sys.exit(f"ERROR: stage 1 output missing ({src}) — run stage 1 first")

    d = stage_dir(name, 2)
    preds = dict(np.load(src))

    ply = export_ply(preds, d, args)
    outs = [ply]

    lines = [f"STAGE 2 — pointcloud",
             f"  conf_thres        : {args.conf_thres}",
             f"  prediction_mode   : {args.prediction_mode}",
             f"  src stage1        : {args.src or name}", ""]
    lines += ["  " + _pcd_stats(p) for p in outs if os.path.exists(p)]

    fp = floor_planarity(ply)
    if fp:
        lines += ["",
                  "  floor planarity (physical truth — tile floor is flat):",
                  f"    RMS      : {fp['rms_mm']:.2f} mm",
                  f"    p95      : {fp['p95_mm']:.2f} mm",
                  f"    n_floor  : {fp['n_floor']:,}  ({fp['frac_floor']*100:.0f}% of cloud)"]
    _write_summary(d, lines)


# ── Stage 3 ────────────────────────────────────────────────────────────

def _measured_marker_colour(args, name):
    """The marker colour Stage 0 measured for this run, if it ran.

    Falls back to None, which leaves Stage 3 on the config defaults — those
    describe one khaki band, so a differently coloured marker needs this.
    """
    for cand in (name, args.src):
        if not cand:
            continue
        path = os.path.join(stage_dir(cand, 0, create=False), "framing.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f).get("marker_colour")
    return None


def _detected_band_count(args, name):
    """How many marker bands Stage 0 counted, for --cut-mode auto.

    None when Stage 0 did not run, or ran before multi-band detection existed.
    That is "unknown", not "none": resolve_cut_mode falls back to the plane
    count alone rather than treating a missing report as evidence.
    """
    for cand in (name, args.src):
        if not cand:
            continue
        path = os.path.join(stage_dir(cand, 0, create=False), "framing.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f).get("bands")
    return None


def _projected_band_planes(args, name):
    """Cut planes projected from Stage 0's band boxes, if both stages ran.

    Empty when Stage 0 or Stage 1 is missing from this run, which leaves Stage 3
    on colour detection alone — exactly what it did before.
    """
    from pipeline.core.bands3d import band_planes_from_dirs
    for cand in (name, args.src):
        if not cand:
            continue
        prep = stage_dir(cand, 0, create=False)
        npz = os.path.join(stage_dir(cand, 1, create=False), "predictions.npz")
        if os.path.exists(os.path.join(prep, "manifest.json")) and os.path.exists(npz):
            return band_planes_from_dirs(
                prep, npz, height_axis=args.segment_height_axis)
    return []


def _review_planes(args):
    """Cutting planes supplied by an interactive review, if any.

    The file holds {"markers": [{"centroid": [x,y,z], "normal": [x,y,z]}, ...]}
    in LEVELLED space, which is the shape Stage 3 already publishes as
    cutting_line_levelled.json — so a review can read that file, move a plane
    and hand the same structure straight back.
    """
    if not getattr(args, "planes", None):
        return None
    with open(args.planes) as f:
        data = json.load(f)
    markers = data.get("markers", data) if isinstance(data, dict) else data
    return list(markers)


def run_stage3(args, name):
    """Segments the cloud, detects the marker planes, and exports the per-object clouds.

    With --cut-only it skips straight to applying already-confirmed planes to
    the uncut limb on disk, which avoids paying for a second clustering pass.
    """
    import glob
    from pipeline.stages.clean import clean_and_extract, cut_only

    # --cut-only is the second half of a Stage 3 that deferred its cut. Every
    # expensive step already ran and its result is on disk, so this reads the
    # uncut limb back and applies the confirmed planes to it. Doing that instead
    # of a full re-run is what keeps a review from costing a second clustering,
    # ghost filter, MLS and levelling pass.
    if getattr(args, "cut_only", False):
        d = stage_dir(name, 3)
        paths = cut_only(d, _review_planes(args) or [],
                         fill_enabled=not args.no_fill)
        lines = ["STAGE 3 — cut only   (planes confirmed by review)", ""]
        for p in sorted(glob.glob(os.path.join(d, "objects", "*.ply"))):
            lines.append("  " + _pcd_stats(p))
        lines += ["", f"  objects handed to stage 4: {len(paths or [])}"]
        for p in (paths or []):
            lines.append(f"    {p}")
        _write_summary(d, lines)
        return

    prev = src_dir(args, name, 2)
    ply = os.path.join(prev, "points.ply")
    if not os.path.exists(ply):
        sys.exit(f"ERROR: stage 2 output missing ({ply}) — run stage 2 first")


    d = stage_dir(name, 3)
    paths = clean_and_extract(
        ply, d, args.num_objects, seed=args.seed,
        segment_leg=args.segment_leg,
        segment_height_axis=args.segment_height_axis,
        fill_enabled=not args.no_fill,
        clean_ply_path=None,
        marker_colour=_measured_marker_colour(args, name),
        override_planes=_review_planes(args),
        cut_mode=getattr(args, "cut_mode", None),
        n_bands=_detected_band_count(args, name),
        band_planes=_projected_band_planes(args, name),
        apply_cut=not getattr(args, "no_cut", False))

    from pipeline.stages.clean import resolve_cut_mode
    lines = [f"STAGE 3 — clean   fill={not args.no_fill}  "
             f"segment_leg={args.segment_leg}  "
             f"cut_mode={resolve_cut_mode(getattr(args, 'cut_mode', None))}", ""]
    for p in sorted(glob.glob(os.path.join(d, "**", "*.ply"), recursive=True)):
        lines.append("  " + _pcd_stats(p))
    lines += ["", f"  objects handed to stage 4: {len(paths or [])}"]
    for p in (paths or []):
        lines.append(f"    {p}")
    _write_summary(d, lines)


# ── Stage 4 / 5 / 6 ────────────────────────────────────────────────────

def _mesh_stats(path):
    """Returns a one-line summary of a mesh: vertex and face counts, watertightness, volume."""
    import trimesh
    m = trimesh.load(path, process=False)
    m.merge_vertices()
    return (f"{os.path.basename(path):<24} {len(m.vertices):>7,} v  {len(m.faces):>7,} f  "
            f"watertight={str(m.is_watertight):<5}  vol={abs(m.volume):.6f}")


def _clear_meshes(d):
    """Remove every mesh in a stage's output directory.

    Right for stage 5, whose output is derived wholly from stage 4 and costs
    nothing to rebuild.
    """
    import glob as _glob
    mesh_dir = os.path.join(d, "mesh")
    for pat in ("*.ply", "*.stl"):
        for f in _glob.glob(os.path.join(mesh_dir, pat)):
            os.remove(f)


def _prune_meshes(d, object_paths):
    """Drop stage 4 meshes that no longer have an object behind them.

    A stage's output has to describe that stage's run: when the limb's cut is
    deferred there is no leg_cut.ply, and a leg_cut_recon.ply left from an
    earlier run would be picked up by stage 5 and measured as if it were this
    run's answer. That is exactly how a deferred cut still reported a cut limb.

    Meshes whose object IS still present are left alone, so reconstruct can
    reuse the ones whose cloud has not changed — the reference cube in
    particular, which no cut ever touches.
    """
    import glob as _glob
    from pipeline.stages.reconstruct import _recon_name

    mesh_dir = os.path.join(d, "mesh")
    keep = {_recon_name(p) for p in object_paths}
    keep.add("scene_recon")            # rebuilt every time from whatever survives
    for f in _glob.glob(os.path.join(mesh_dir, "*")):
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem not in keep:
            os.remove(f)


def run_stage4(args, name):
    """Reconstructs a surface mesh for every object cloud Stage 3 exported."""
    import glob
    from pipeline.stages.reconstruct import reconstruct_mesh_stage

    prev = src_dir(args, name, 3)
    objs = sorted(glob.glob(os.path.join(prev, "objects", "*.ply")))
    objs = [p for p in objs if os.path.basename(p) in ("box.ply", "leg_cut.ply", "obj.ply")]
    if not objs:
        sys.exit("ERROR: stage 3 objects missing — run stage 3 first")

    d = stage_dir(name, 4)
    _prune_meshes(d, objs)
    scene, recon = reconstruct_mesh_stage(
        objs, d, seed=args.seed, method=args.recon_method,
        box_method=args.box_recon_method, obj_method=args.obj_recon_method)

    lines = [f"STAGE 4 — recon   method={args.recon_method} "
             f"box={args.box_recon_method} obj={args.obj_recon_method}", ""]
    lines += ["  " + _mesh_stats(p) for p in (recon or [])]
    _write_summary(d, lines)


def run_stage5(args, name):
    """Repairs Stage 4's meshes into watertight solids, skipping the merged scene mesh."""
    import glob
    from pipeline.stages.watertight import watertight_stage

    prev = os.path.join(src_dir(args, name, 4), "mesh")
    # scene_recon.ply is the MERGE of the per-object meshes, not another object.
    # Feeding it back in makes Stage 5 merge every object twice, and the
    # coincident duplicate geometry turns two closed surfaces into thousands of
    # non-manifold fragments (measured: 2 components -> 7,554, euler 4 -> 3113).
    # The orchestrator passes recon_mesh_paths, which already excludes it; only
    # this glob had to rediscover that.
    recon = sorted(p for p in glob.glob(os.path.join(prev, "*_recon.ply"))
                   if os.path.basename(p) != "scene_recon.ply")
    if not recon:
        sys.exit("ERROR: stage 4 recon meshes missing — run stage 4 first")

    d = stage_dir(name, 5)
    _clear_meshes(d)
    scene, wt = watertight_stage(recon, d)

    lines = ["STAGE 5 — watertight", ""]
    lines += ["  " + _mesh_stats(p) for p in (wt or [])]
    _write_summary(d, lines)


def run_stage6(args, name):
    """Computes real-world volumes, preferring Stage 5's meshes and falling back to Stage 4's."""
    import glob
    from pipeline.stages.volume import compute_volumes

    for n in (5, 4):
        prev = os.path.join(src_dir(args, name, n), "mesh")
        meshes = sorted(glob.glob(os.path.join(prev, "*.ply")))
        meshes = [p for p in meshes
                  if not os.path.basename(p).startswith("scene")]
        if meshes:
            break
    if not meshes:
        sys.exit("ERROR: no meshes from stage 4/5 — run those first")

    d = stage_dir(name, 6)
    df = compute_volumes(meshes, voxel_res=args.voxel_res,
                         auto_res=args.auto_res,
                         clean_dir=src_dir(args, name, 3))
    lines = [f"STAGE 6 — volume   (from {prev})", ""]
    if df is not None:
        df.to_csv(os.path.join(d, "volumes.csv"), index=False)
        lines.append(df.to_string(index=False))
    _write_summary(d, lines)


RUNNERS = {0: run_stage0, 1: run_stage1, 2: run_stage2, 3: run_stage3,
           4: run_stage4, 5: run_stage5, 6: run_stage6}


def parse_stages(spec):
    """Expands a stage spec into a list of stage numbers. Accepts '3' or a '2-6' range."""
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(spec)]


def main():
    """Parses the stage spec and runs each requested stage in order."""
    p = argparse.ArgumentParser(description="Run pipeline stages individually")
    p.add_argument("stages", help="stage number or range, e.g. 1 or 1-3")
    p.add_argument("-i", "--image_folder", default="./inputs/small_leg/")
    p.add_argument("--name", default=None, help="work dir name (default: input folder name)")
    p.add_argument("--force", action="store_true", help="ignore cached stage 1")
    p.add_argument("--prep-band", dest="prep_band", type=float, default=1.6,
                   help="stage 0: how far above the floor the limb reaches, in "
                        "cube heights (the cube stands on the floor and fixes "
                        "both the level and the scale)")
    p.add_argument("--prep-pad", dest="prep_pad", type=float, default=0.05,
                   help="stage 0: margin around the subject before squaring")
    p.add_argument("--prep-size", dest="prep_size", type=int, default=518,
                   help="stage 0: side of the emitted square. 518 is what VGGT "
                        "resizes to anyway, and for a square input its own "
                        "arithmetic is exact, so nothing is lost by doing the "
                        "reduction here. 0 emits at native resolution.")
    p.add_argument("--prep-min-frames", dest="prep_min_frames", type=int,
                   default=6, help="stage 0: frames required before the "
                                   "pipeline may run")
    p.add_argument("--no-prep-crop", dest="prep_crop",
                   action="store_false",
                   help="hand VGGT the original frames instead of stage 0's crop.\n"
                        "VGGT then centre-crops them itself, which discards 44%% of\n"
                        "a 9:16 photo without regard for where the reference is.")
    p.add_argument("--continue-on-rejected", dest="prep_strict",
                   action="store_false",
                   help="run the pipeline even though stage 0 rejected "
                        "frames. Off by default: a clipped reference "
                        "corrupts the scale of every reported volume "
                        "and leaves no visible sign that it did.")
    p.add_argument("--prep-lenient", dest="prep_strict", action="store_false",
                   help=argparse.SUPPRESS)  # old name for --continue-on-rejected
    p.add_argument("--prep-frame-centred", dest="prep_recentre",
                   action="store_false",
                   help="stage 0: keep the window concentric with the frame "
                        "instead of centring it on the subject. Rarely useful: "
                        "it degenerates to the old centre crop and left 1 of 6 "
                        "frames usable against 4.")
    p.add_argument("--inference", default=None,
                   help="run dir holding 01_inference, for the Stage 6 marker "
                        "cross-check. Needed when stage 1 lives in another run "
                        "and --src would also redirect the meshes.")
    p.add_argument("--src", default=None,
                   help="run name to read the PREVIOUS stage from (default: --name)")
    p.add_argument("--no-dump", dest="dump", action="store_false", default=True,
                   help="skip materialising raw stage 1 outputs (images/depth/PLY/cameras)")
    p.add_argument("--conf_thres", type=float, default=45.0)
    p.add_argument("--prediction_mode", default="pointmap", choices=["pointmap", "depth"])
    p.add_argument("--mask_black_bg", action="store_true")
    p.add_argument("--mask_white_bg", action="store_true")
    p.add_argument("--num_objects", type=int, default=2)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--preprocess-mode", dest="preprocess_mode", default="crop",
                   choices=["crop", "pad"],
                   help="crop (default, centre-crops height) or pad (keeps whole frame)")
    p.add_argument("--input-res", dest="input_res", type=int, default=518,
                   help="VGGT input resolution, must be divisible by 14 (518 native, 1022 hi-res)")
    p.add_argument("--no-fill", action="store_true")
    p.add_argument("--no-segment-leg", action="store_false", dest="segment_leg")
    p.add_argument("--segment-height-axis", default="z", choices=["x", "y", "z"])
    p.add_argument("--cut-mode", dest="cut_mode", default=None,
                   choices=["upper", "span", "auto"],
                   help="stage 3: 'upper' measures everything below the highest "
                        "valid band (one-band capture); 'span' measures the "
                        "segment between the outermost two (upper and lower "
                        "band); 'auto' follows the bands, cutting a span only "
                        "when Stage 0's band count and Stage 3's plane count "
                        "both say two. Default: config.MARKER_CUT_MODE (auto).")
    p.add_argument("--cut-only", dest="cut_only", action="store_true",
                   help="stage 3: skip straight to the cut, reusing the uncut "
                        "result a previous --no-cut run left on disk. Needs "
                        "--planes.")
    p.add_argument("--no-cut", dest="no_cut", action="store_true",
                   help="stage 3: detect the cutting planes and publish them, "
                        "but do not cut. The uncut cloud is measured instead, "
                        "so a review can approve the cut before it is applied.")
    p.add_argument("--planes", default=None,
                   help="JSON of cutting planes in levelled space to use instead "
                        "of the detected ones (stage 3) — the shape stage 3 "
                        "writes as cutting_line_levelled.json")
    p.add_argument("--recon-method", dest="recon_method", default="poisson")
    p.add_argument("--box-recon-method", dest="box_recon_method", default=None)
    p.add_argument("--obj-recon-method", dest="obj_recon_method", default=None)
    p.add_argument("--voxel-res", dest="voxel_res", type=int, default=150)
    p.add_argument("--no-auto-res", dest="auto_res", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    name = args.name or os.path.basename(os.path.normpath(args.image_folder))

    from pipeline.utils.seeding import seed_everything
    seed_everything(args.seed)

    for n in parse_stages(args.stages):
        print()
        print("=" * 64)
        print(f"  STAGE {n}  ({STAGE_DIRS[n]})   work/{name}/")
        print("=" * 64)
        t0 = time.time()
        RUNNERS[n](args, name)
        print(f"\n  stage {n} took {time.time() - t0:.1f}s")

        # Stage 0 rewrites the frames, so every later stage in this run must read
        # what it produced rather than the folder the user typed.
        #
        # Without this, `stagerun.py 0-6 -i inputs/foo` framed every photo, wrote
        # the crops, and then handed VGGT inputs/foo anyway -- Stage 0 ran and its
        # entire output was discarded, silently. The run still looked correct: the
        # framing report was written, the marker colour was learned and used, and
        # only the pixels the model saw were wrong.
        if n == 0:
            prep_images = os.path.join(stage_dir(name, 0, create=False), "images")
            if os.path.isdir(prep_images):
                args.image_folder = prep_images
                print(f"  -> later stages will read {prep_images}")

        # --src redirects only the first stage of a range; once a stage has run
        # under `name`, later stages must read what it just wrote.
        args.src = None


if __name__ == "__main__":
    main()
