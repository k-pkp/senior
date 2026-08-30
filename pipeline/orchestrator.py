"""Top-level pipeline orchestrator — runs Stages 1-7 end to end."""
import glob
import os
import shutil
import sys
import time

import numpy as np

from vggt.utils.device import get_device

from pipeline.cli import parse_args
from pipeline.config import IMAGE_EXTENSIONS
from pipeline.utils.runlog import RunLogger
from pipeline.utils.seeding import seed_everything

from pipeline.stages.inference import run_inference
from pipeline.stages.pointcloud import export_ply
from pipeline.stages.clean import clean_and_extract
from pipeline.stages.reconstruct import reconstruct_mesh_stage
from pipeline.stages.watertight import watertight_stage
from pipeline.stages.volume import compute_volumes


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

STAGE_DIRS = {
    0: "00_prep",
    1: "01_inference",
    2: "02_pointcloud",
    3: "03_clean",
    4: "04_recon",
    5: "05_watertight",
    6: "06_volume",
}

# Final deliverables, promoted out of for_debug/ to the top of output/.
FINAL_NAMES = {"leg_cut": "leg_mesh", "obj": "leg_mesh",
               "box": "box_mesh", "scene_colour": "scene_mesh"}


def _print_banner(args, device):
    """Prints the run's configuration banner before any stage starts."""
    print(f"╔{'═' * 58}╗")
    print(f"║  VGGT Full Pipeline                                      ║")
    print(f"╠{'═' * 58}╣")
    print(f"║  Device        : {device:<40}║")
    print(f"║  Input         : {args.image_folder:<40}║")
    print(f"║  Pred. mode    : {args.prediction_mode:<40}║")
    print(f"║  Conf. thresh  : {args.conf_thres:<40}║")
    print(f"║  Skip mesh     : {str(args.skip_mesh):<40}║")
    recon_label = args.recon_method
    if args.box_recon_method and args.box_recon_method != args.recon_method:
        recon_label += f" (box={args.box_recon_method})"
    if args.obj_recon_method and args.obj_recon_method != args.recon_method:
        recon_label += f" (obj={args.obj_recon_method})"
    print(f"║  Recon method  : {recon_label:<40}║")
    print(f"║  Watertight    : {str(not args.no_watertight):<40}║")
    print(f"║  Seed          : {args.seed:<40}║")
    print(f"║  Leg segment   : {str(args.segment_leg):<40}║")
    from pipeline.stages.clean import resolve_cut_mode
    cut_label = {"upper": "upper (below top band)",
                 "span": "span (between bands)",
                 "auto": "auto (follows the bands found)"}[
        resolve_cut_mode(getattr(args, "cut_mode", None))]
    print(f"║  Cut mode      : {cut_label:<40}║")
    print(f"╚{'═' * 58}╝")


def _publish_final_meshes(output_dir, wt_mesh_paths, recon_mesh_paths, scene_path):
    """Promote the deliverables to output/ as leg_mesh / box_mesh / scene_mesh.

    Everything else stays under for_debug/. Falls back to the Stage 4 recon when
    watertight repair was skipped, so the three files always exist if a mesh was
    produced at all.
    """
    import open3d as o3d

    sources = list(wt_mesh_paths or recon_mesh_paths or [])
    if scene_path:
        sources.append(scene_path)

    published = []
    for src in sources:
        if not src or not os.path.exists(src):
            continue
        stem = os.path.splitext(os.path.basename(src))[0]
        stem = stem[:-len("_recon")] if stem.endswith("_recon") else stem
        final = FINAL_NAMES.get(stem)
        if final is None:
            continue
        dst_ply = os.path.join(output_dir, f"{final}.ply")
        shutil.copy2(src, dst_ply)
        mesh = o3d.io.read_triangle_mesh(dst_ply)
        mesh.compute_vertex_normals()
        dst_stl = os.path.join(output_dir, f"{final}.stl")
        o3d.io.write_triangle_mesh(dst_stl, mesh)
        published.append((final, len(mesh.vertices), len(mesh.triangles)))

    if published:
        print()
        print("  Final meshes:")
        for name, nv, nf in sorted(set(published)):
            print(f"    {name}.ply / .stl   {nv:,} verts, {nf:,} faces")
    return published


def _copy_images_to_target(image_folder, target_images_dir):
    """Copy input images to target/images/ for demo_gradio compatibility."""
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    image_files = [p for p in image_files if p.lower().endswith(IMAGE_EXTENSIONS)]
    for src in image_files:
        dst = os.path.join(target_images_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)


def _print_summary(total_time, inference_time, ply_path, scene_recon_path,
                   recon_mesh_paths, scene_wt_path, wt_mesh_paths,
                   npz_path2, target_dir, target_images_dir):
    """Prints the closing summary: timings, every output path, and viewer commands."""
    print()
    print(f"╔{'═' * 58}╗")
    print(f"║  Pipeline Complete                                       ║")
    print(f"╠{'═' * 58}╣")
    print(f"║  Total time    : {total_time:>6.1f}s{' ' * 33}║")
    print(f"║  Inference     : {inference_time:>6.1f}s{' ' * 33}║")
    print(f"╠{'═' * 58}╣")
    print(f"║  Outputs:                                                ║")
    print(f"║    PLY         : {ply_path:<40}║")
    if scene_recon_path:
        print(f"║    Scene recon : {scene_recon_path:<40}║")
    for p in recon_mesh_paths:
        name = os.path.basename(p)
        print(f"║    Recon       : {name:<40}║")
    if scene_wt_path:
        print(f"║    Scene wt    : {scene_wt_path:<40}║")
    for p in wt_mesh_paths:
        name = os.path.basename(p)
        print(f"║    Wt          : {name:<40}║")
    print(f"║    Predictions : {npz_path2:<40}║")
    print(f"║    Target dir  : {target_dir:<40}║")
    print(f"╚{'═' * 58}╝")
    print()
    print("To view results interactively:")
    print(f"  python pipeline/tools/viewer.py {ply_path}")
    if scene_recon_path:
        print(f"  python pipeline/tools/viewer.py {scene_recon_path}  # recon scene")
    for p in recon_mesh_paths:
        print(f"  python pipeline/tools/viewer.py {p}")
    if scene_wt_path:
        print(f"  python pipeline/tools/viewer.py {scene_wt_path}  # watertight scene")
    for p in wt_mesh_paths:
        print(f"  python pipeline/tools/viewer.py {p}")
    print()
    print("To use with demo_gradio.py:")
    print(f"  The predictions are saved at: {target_dir}/predictions.npz")
    print(f"  Images are at: {target_images_dir}/")


def main():
    """Runs Stages 0-6 end to end, then publishes the final meshes and writes the run log."""
    args = parse_args()
    device = get_device()
    total_t0 = time.time()

    seed_everything(args.seed)

    _print_banner(args, device)

    if not os.path.isdir(args.image_folder):
        print(f"\nERROR: Input folder not found: {args.image_folder}")
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = os.path.join(_PROJECT_ROOT, "output")
    os.makedirs(args.output_dir, exist_ok=True)

    logger = None
    inference_time = None
    obj_vol_cm3 = None
    box_vol_cm3 = None
    if args.log:
        logger = RunLogger(os.path.join(_PROJECT_ROOT, "log.csv"), device)
        logger.start()

    try:
        # Everything intermediate lands under for_debug/<stage>/; only the three
        # final meshes sit at the top of output/.
        debug_root = os.path.join(args.output_dir, "for_debug")
        stage_dirs = {n: os.path.join(debug_root, d) for n, d in STAGE_DIRS.items()}
        for d in stage_dirs.values():
            os.makedirs(d, exist_ok=True)

        target_dir = os.path.join(stage_dirs[1], "target")
        target_images_dir = os.path.join(target_dir, "images")
        os.makedirs(target_images_dir, exist_ok=True)

        # ── Stage 0: Framing ──
        #
        # Runs before anything else and refuses to continue on a capture that
        # cannot be framed. A frame that clips the reference cube corrupts the
        # scale for every number the pipeline goes on to report, and does so
        # invisibly, so the only safe response is to stop and say which images
        # to re-take.
        inference_input = args.image_folder
        marker_colour = None
        n_bands = None
        manifest = None
        if getattr(args, "prep", True):
            from pipeline.stages.prep import prepare_frames
            prep_images = os.path.join(stage_dirs[0], "images")
            # crop= and output_size= are threaded through deliberately.
            # They used to be omitted here while stagerun.py passed both, so
            # `run.py --no-prep-crop` parsed the flag, stored it, and cropped
            # anyway -- the same flag worked on one entry point and not the
            # other, silently. getattr keeps this in step if either flag is
            # added to only one of the two parsers again.
            manifest = prepare_frames(args.image_folder, prep_images,
                           band_heights=args.prep_band, pad=args.prep_pad,
                           centre_on_subject=args.prep_recentre,
                           strict=args.prep_strict,
                           crop=getattr(args, "prep_crop", True),
                           output_size=getattr(args, "prep_size", 518),
                           min_frames=args.prep_min_frames)
            inference_input = prep_images
            # Stage 3 uses the colour Stage 0 measured, so a marker of any
            # colour works without editing the config's khaki defaults.
            marker_colour = manifest.get("marker_colour")
            # How many bands Stage 0 counted, for --cut-mode auto. Absent on a
            # manifest written before multi-band detection, and None there means
            # "unknown", not "none" — see clean.resolve_cut_mode.
            n_bands = manifest.get("bands")

        # Record what VGGT was actually shown. This used to copy the submitted
        # folder, which stopped being the same thing once Stage 0 could rewrite
        # the frames -- the record then quietly disagreed with the run.
        _copy_images_to_target(inference_input, target_images_dir)

        # ── Stage 1: Inference ──
        predictions, inference_time = run_inference(inference_input, device, args.max_frames)
        print(f"[DBG-stage] stage1 inference: {inference_time:.2f}s")

        # Save predictions (compatible with demo_gradio)
        npz_path = os.path.join(target_dir, "predictions.npz")
        save_dict = {k: v for k, v in predictions.items() if v is not None}
        np.savez_compressed(npz_path, **save_dict)
        print(f"  Saved predictions: {npz_path}")

        npz_path2 = os.path.join(stage_dirs[1], "predictions.npz")
        shutil.copy2(npz_path, npz_path2)

        # ── Stage 2: Export PLY ──
        _dbg_t = time.time()
        ply_path = export_ply(predictions, stage_dirs[2], args)
        print(f"[DBG-stage] stage2 export_ply: {time.time() - _dbg_t:.2f}s")

        # Planes projected from Stage 0's band boxes through Stage 1's own
        # pointmap. A second source for the cut, independent of colour, merged
        # with the colour planes inside Stage 3 — see core/bands3d.py.
        band_planes = []
        if getattr(args, "prep", True) and manifest is not None:
            from pipeline.core.bands3d import band_planes_from_arrays
            band_planes = band_planes_from_arrays(
                manifest, predictions["world_points"],
                predictions["world_points_conf"],
                height_axis=args.segment_height_axis)

        # ── Stages 3-5: Clean + Reconstruct + Watertight ──
        scene_recon_path = None
        recon_mesh_paths = []
        scene_wt_path = None
        wt_mesh_paths = []

        if not args.skip_mesh:
            _dbg_t = time.time()
            object_paths = clean_and_extract(
                ply_path, stage_dirs[3], args.num_objects, seed=args.seed,
                segment_leg=args.segment_leg,
                segment_height_axis=args.segment_height_axis,
                fill_enabled=not args.no_fill,
                marker_colour=marker_colour,
                cut_mode=getattr(args, "cut_mode", None),
                n_bands=n_bands,
                band_planes=band_planes)
            print(f"[DBG-stage] stage3 clean_and_extract: {time.time() - _dbg_t:.2f}s")
            if object_paths:
                _dbg_t = time.time()
                scene_recon_path, recon_mesh_paths = reconstruct_mesh_stage(
                    object_paths, stage_dirs[4], seed=args.seed,
                    method=args.recon_method,
                    box_method=args.box_recon_method,
                    obj_method=args.obj_recon_method)
                print(f"[DBG-stage] stage4 reconstruct: {time.time() - _dbg_t:.2f}s")

                if recon_mesh_paths and not args.no_watertight:
                    _dbg_t = time.time()
                    # Stage 3 detects the planes and publishes them; Stage 5 is
                    # where they are actually applied, to the repaired solid.
                    detected_planes_path = os.path.join(
                        stage_dirs[3], "debug", "cutting_line_levelled.json")
                    cut_planes = None
                    if os.path.exists(detected_planes_path):
                        import json as _json
                        with open(detected_planes_path) as planes_file:
                            cut_planes = _json.load(planes_file).get("markers", [])
                    scene_wt_path, wt_mesh_paths = watertight_stage(
                        recon_mesh_paths, stage_dirs[5],
                        cut_planes=cut_planes,
                        height_axis=args.segment_height_axis)
                    print(f"[DBG-stage] stage5 watertight: {time.time() - _dbg_t:.2f}s")
        else:
            print("\n  (Skipping mesh stages — --skip_mesh was set)")

        # ── Stage 6: Volumes ──
        vol_objects = wt_mesh_paths or recon_mesh_paths
        if vol_objects:
            _dbg_t = time.time()
            vol_df = compute_volumes(vol_objects,
                                     voxel_res=args.voxel_res,
                                     auto_res=args.auto_res,
                                     clean_dir=stage_dirs[3])
            print(f"[DBG-stage] stage6 volumes: {time.time() - _dbg_t:.2f}s")
            if vol_df is not None:
                vol_df.to_csv(os.path.join(stage_dirs[6], "volumes.csv"), index=False)
                box_rows = vol_df[vol_df["is_ref"]]
                obj_rows = vol_df[~vol_df["is_ref"]]
                if not box_rows.empty:
                    box_vol_cm3 = float(box_rows.iloc[0]["real_vol_cm3"])
                if not obj_rows.empty:
                    obj_vol_cm3 = float(obj_rows.iloc[0]["real_vol_cm3"])

        _publish_final_meshes(args.output_dir, wt_mesh_paths, recon_mesh_paths,
                              scene_wt_path or scene_recon_path)

        total_time = time.time() - total_t0
        _print_summary(total_time, inference_time, ply_path, scene_recon_path,
                       recon_mesh_paths, scene_wt_path, wt_mesh_paths,
                       npz_path2, target_dir, target_images_dir)
    finally:
        if logger is not None:
            logger.stop_and_write(args.image_folder, args.output_dir, inference_time,
                                  obj_vol_cm3=obj_vol_cm3, box_vol_cm3=box_vol_cm3)
