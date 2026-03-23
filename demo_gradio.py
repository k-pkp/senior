# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
import io
from contextlib import redirect_stdout, redirect_stderr

sys.path.append("vggt/")
sys.path.append("clean")

from visual_util import predictions_to_glb, export_point_cloud_to_ply
from clean_ply import clean_point_cloud
from recons import reconstruct_mesh
from com_vol import compute_volume_from_mesh
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")
# model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model

model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))


model.eval()
model = model.to(device)


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def run_model(target_dir, model) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None # remove pose_enc_list

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    return predictions


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # 1 frame/sec

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_video, input_images):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_video and not input_images:
        return None, None, None, None, "Pipeline idle.", None, None, {"next_step": -1}
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return (
        None,
        target_dir,
        image_paths,
        "Upload complete. Click 'Reconstruct' to begin 3D processing.",
        "GLB step not started. Click Reconstruct.",
        None,
        None,
        {"next_step": -1},
    )


def update_gallery_on_upload_with_logs(input_video, input_images):
    """Capture upload handler terminal logs into UI log panel."""
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        recon, target_dir, gallery, log_msg, volume_msg, mesh_file, mesh_view, state = update_gallery_on_upload(
            input_video, input_images
        )
    combined_log = f"{log_msg}\n\n{_format_terminal_log(buffer.getvalue())}"
    return recon, target_dir, gallery, combined_log, volume_msg, mesh_file, mesh_view, state


def run_volume_pipeline(
    predictions,
    target_dir,
    conf_thres,
    frame_filter,
    mask_black_bg,
    mask_white_bg,
    mask_sky,
    prediction_mode,
    cube_size_cm,
):
    """Run clean/reconstruct/volume pipeline and return final volume message and mesh PLY path."""
    clean_base_dir = "clean"
    input_dir = os.path.join(clean_base_dir, "input")
    clean_input_dir = os.path.join(clean_base_dir, "clean_input_ply")
    output_dir = os.path.join(clean_base_dir, "output_ply")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(clean_input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Step 2: Save as clean/input/input.ply
    input_ply_path = os.path.join(input_dir, "input.ply")
    export_point_cloud_to_ply(
        predictions,
        input_ply_path,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )

    # Step 3: Clean point cloud and save to clean_input_ply
    cleaned_ply_path = clean_point_cloud(
        input_path=input_ply_path,
        output_folder=clean_input_dir,
        output_name="clean_input.ply",
    )

    # Step 4: Reconstruct and save to output_ply (PLY + STL)
    mesh_ply_path, _ = reconstruct_mesh(
        input_path=cleaned_ply_path,
        output_folder=output_dir,
        base_name="input",
    )

    # Step 5: Compute volume from reconstructed mesh (PLY)
    volume_result = compute_volume_from_mesh(
        mesh_path=mesh_ply_path,
        reference_ply_path=cleaned_ply_path,
        real_cube_size_m=float(cube_size_cm) / 100.0,
    )

    volume_msg = (
        f"Volume result: {volume_result['real_volume_cm3']:.2f} cm^3 "
        f"({volume_result['real_volume_m3']:.6f} m^3). "
        f"Scale factor: {volume_result['scale_factor']:.6f}. "
        f"Watertight: {volume_result['is_watertight']}"
    )

    return volume_msg, mesh_ply_path


def init_pipeline_state(target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, mask_sky, prediction_mode, cube_size_cm):
    """Initialize step state right after GLB is generated."""
    return {
        "next_step": 0,
        "target_dir": target_dir,
        "conf_thres": conf_thres,
        "frame_filter": frame_filter,
        "mask_black_bg": mask_black_bg,
        "mask_white_bg": mask_white_bg,
        "mask_sky": mask_sky,
        "prediction_mode": prediction_mode,
        "cube_size_cm": cube_size_cm,
        "input_ply_path": None,
        "cleaned_ply_path": None,
        "mesh_ply_path": None,
    }


def _prepare_step_paths():
    clean_base_dir = "clean"
    input_dir = os.path.join(clean_base_dir, "input")
    clean_input_dir = os.path.join(clean_base_dir, "clean_input_ply")
    output_dir = os.path.join(clean_base_dir, "output_ply")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(clean_input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return input_dir, clean_input_dir, output_dir


def _load_predictions(target_dir):
    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        raise FileNotFoundError("No predictions found. Please run Reconstruct first.")

    key_list = [
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "images",
        "extrinsic",
        "intrinsic",
        "world_points_from_depth",
    ]
    loaded = np.load(predictions_path)
    return {key: np.array(loaded[key]) for key in key_list}


def continue_pipeline_step(
    pipeline_state,
    target_dir,
    conf_thres,
    frame_filter,
    mask_black_bg,
    mask_white_bg,
    mask_sky,
    prediction_mode,
    cube_size_cm,
):
    """Run exactly one next pipeline step after GLB generation."""
    if not isinstance(pipeline_state, dict):
        pipeline_state = {"next_step": -1}

    next_step = pipeline_state.get("next_step", -1)
    if next_step < 0:
        return pipeline_state, "Please click Reconstruct first.", None, None, "Pipeline not initialized."

    pipeline_state.update(
        {
            "target_dir": target_dir,
            "conf_thres": conf_thres,
            "frame_filter": frame_filter,
            "mask_black_bg": mask_black_bg,
            "mask_white_bg": mask_white_bg,
            "mask_sky": mask_sky,
            "prediction_mode": prediction_mode,
            "cube_size_cm": cube_size_cm,
        }
    )

    input_dir, clean_input_dir, output_dir = _prepare_step_paths()

    try:
        if next_step == 0:
            predictions = _load_predictions(target_dir)
            input_ply_path = os.path.join(input_dir, "input.ply")
            export_point_cloud_to_ply(
                predictions,
                input_ply_path,
                conf_thres=conf_thres,
                filter_by_frames=frame_filter,
                mask_black_bg=mask_black_bg,
                mask_white_bg=mask_white_bg,
                mask_sky=mask_sky,
                target_dir=target_dir,
                prediction_mode=prediction_mode,
            )
            pipeline_state["input_ply_path"] = input_ply_path
            pipeline_state["next_step"] = 1
            msg = (
                f"Step 1 done. Input used: {target_dir}/predictions.npz\n"
                f"Output: {input_ply_path}\n"
                "Click Continue Step."
            )
            return pipeline_state, msg, input_ply_path, None, "Step 1 completed."

        if next_step == 1:
            input_ply_path = pipeline_state.get("input_ply_path")
            if not input_ply_path or not os.path.exists(input_ply_path):
                raise FileNotFoundError("Step 1 output missing. Click Redo to rerun Step 1.")
            cleaned_ply_path = clean_point_cloud(
                input_path=input_ply_path,
                output_folder=clean_input_dir,
                output_name="clean_input.ply",
            )
            pipeline_state["cleaned_ply_path"] = cleaned_ply_path
            pipeline_state["next_step"] = 2
            msg = (
                f"Step 2 done. Input used: {input_ply_path}\n"
                f"Output: {cleaned_ply_path}\n"
                "Click Continue Step."
            )
            return pipeline_state, msg, cleaned_ply_path, None, "Step 2 completed."

        if next_step == 2:
            cleaned_ply_path = pipeline_state.get("cleaned_ply_path")
            if not cleaned_ply_path or not os.path.exists(cleaned_ply_path):
                raise FileNotFoundError("Step 2 output missing. Click Redo to rerun Step 2.")
            mesh_ply_path, _ = reconstruct_mesh(
                input_path=cleaned_ply_path,
                output_folder=output_dir,
                base_name="input",
            )
            pipeline_state["mesh_ply_path"] = mesh_ply_path
            pipeline_state["next_step"] = 3
            msg = (
                f"Step 3 done. Input used: {cleaned_ply_path}\n"
                f"Output mesh PLY: {mesh_ply_path}\n"
                "Click Continue Step."
            )
            return pipeline_state, msg, mesh_ply_path, mesh_ply_path, "Step 3 completed."

        if next_step == 3:
            mesh_ply_path = pipeline_state.get("mesh_ply_path")
            cleaned_ply_path = pipeline_state.get("cleaned_ply_path")
            if not mesh_ply_path or not os.path.exists(mesh_ply_path):
                raise FileNotFoundError("Step 3 output missing. Click Redo to rerun Step 3.")
            if not cleaned_ply_path or not os.path.exists(cleaned_ply_path):
                raise FileNotFoundError("Step 2 output missing. Click Redo to rerun Step 2.")

            volume_result = compute_volume_from_mesh(
                mesh_path=mesh_ply_path,
                reference_ply_path=cleaned_ply_path,
                real_cube_size_m=float(cube_size_cm) / 100.0,
            )
            pipeline_state["next_step"] = 4
            msg = (
                f"Step 4 done. Inputs used: mesh={mesh_ply_path}, reference={cleaned_ply_path}\n"
                f"Step 4 done: volume {volume_result['real_volume_cm3']:.2f} cm^3 "
                f"({volume_result['real_volume_m3']:.6f} m^3). "
                f"Scale factor: {volume_result['scale_factor']:.6f}. "
                f"Watertight: {volume_result['is_watertight']}"
            )
            return pipeline_state, msg, mesh_ply_path, mesh_ply_path, "Pipeline finished."

        mesh_ply_path = pipeline_state.get("mesh_ply_path")
        return pipeline_state, "All steps already completed. Use Redo if needed.", mesh_ply_path, mesh_ply_path, "Pipeline finished."

    except Exception as exc:
        mesh_ply_path = pipeline_state.get("mesh_ply_path")
        return pipeline_state, f"Step failed: {exc}", mesh_ply_path, mesh_ply_path, "Pipeline step failed."


def redo_pipeline_step(
    pipeline_state,
    target_dir,
    conf_thres,
    frame_filter,
    mask_black_bg,
    mask_white_bg,
    mask_sky,
    prediction_mode,
    cube_size_cm,
):
    """Redo the most recently completed step."""
    if not isinstance(pipeline_state, dict):
        pipeline_state = {"next_step": -1}

    next_step = pipeline_state.get("next_step", -1)
    if next_step <= 0:
        return pipeline_state, "No completed post-GLB step to redo. Click Reconstruct, then Continue.", None, None, "Redo unavailable."

    pipeline_state["next_step"] = next_step - 1
    return continue_pipeline_step(
        pipeline_state,
        target_dir,
        conf_thres,
        frame_filter,
        mask_black_bg,
        mask_white_bg,
        mask_sky,
        prediction_mode,
        cube_size_cm,
    )


# -------------------------------------------------------------------------
# 4) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
    cube_size_cm=14.0,
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return (
            None,
            "No valid target directory found. Please upload first.",
            None,
            "GLB step failed. No valid target directory.",
            None,
            None,
            {"next_step": -1},
        )

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)

    pipeline_state = init_pipeline_state(
        target_dir,
        conf_thres,
        frame_filter,
        mask_black_bg,
        mask_white_bg,
        mask_sky,
        prediction_mode,
        cube_size_cm,
    )
    volume_msg = "GLB saved. Click Continue Step to run Step 1."
    mesh_ply_path = None

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

    return (
        glbfile,
        log_msg,
        gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True),
        volume_msg,
        mesh_ply_path,
        mesh_ply_path,
        pipeline_state,
    )


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None


def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."


def _format_terminal_log(log_text):
    """Format captured stdout/stderr for Markdown display."""
    if not log_text.strip():
        return "(no terminal output captured)"
    return f"```\n{log_text.strip()}\n```"


def reconstruct_with_logs(
    target_dir,
    conf_thres,
    frame_filter,
    mask_black_bg,
    mask_white_bg,
    show_cam,
    mask_sky,
    prediction_mode,
    cube_size_cm,
):
    """Run reconstruction and capture terminal logs into UI."""
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        recon_out, log_msg, dropdown, volume_msg, mesh_file, mesh_view, state = gradio_demo(
            target_dir,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            cube_size_cm,
        )
    combined_log = f"{log_msg}\n\n{_format_terminal_log(buffer.getvalue())}"
    return recon_out, combined_log, dropdown, volume_msg, mesh_file, mesh_view, state


def continue_pipeline_step_with_logs(
    pipeline_state,
    target_dir,
    conf_thres,
    frame_filter,
    mask_black_bg,
    mask_white_bg,
    mask_sky,
    prediction_mode,
    cube_size_cm,
):
    """Run one continue step and capture terminal logs into UI."""
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        state, volume_msg, mesh_file, mesh_view, log_msg = continue_pipeline_step(
            pipeline_state,
            target_dir,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            mask_sky,
            prediction_mode,
            cube_size_cm,
        )
    combined_log = f"{log_msg}\n\n{_format_terminal_log(buffer.getvalue())}"
    return state, volume_msg, mesh_file, mesh_view, combined_log


def redo_pipeline_step_with_logs(
    pipeline_state,
    target_dir,
    conf_thres,
    frame_filter,
    mask_black_bg,
    mask_white_bg,
    mask_sky,
    prediction_mode,
    cube_size_cm,
):
    """Redo one step and capture terminal logs into UI."""
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        state, volume_msg, mesh_file, mesh_view, log_msg = redo_pipeline_step(
            pipeline_state,
            target_dir,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            mask_sky,
            prediction_mode,
            cube_size_cm,
        )
    combined_log = f"{log_msg}\n\n{_format_terminal_log(buffer.getvalue())}"
    return state, volume_msg, mesh_file, mesh_view, combined_log


def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer.
    """

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    key_list = [
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "images",
        "extrinsic",
        "intrinsic",
        "world_points_from_depth",
    ]

    loaded = np.load(predictions_path)
    predictions = {key: np.array(loaded[key]) for key in key_list}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"


def update_visualization_with_logs(
    target_dir,
    conf_thres,
    frame_filter,
    mask_black_bg,
    mask_white_bg,
    show_cam,
    mask_sky,
    prediction_mode,
):
    """Capture visualization-update terminal logs into UI log panel."""
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        recon, log_msg = update_visualization(
            target_dir,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        )
    combined_log = f"{log_msg}\n\n{_format_terminal_log(buffer.getvalue())}"
    return recon, combined_log


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(
    theme=theme,
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }

    .example-log * {
        font-style: italic;
        font-size: 16px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }
    
    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }

    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
    """,
) as demo:
    gr.HTML(
        """
    <h1>🏛️ VGGT: Visual Geometry Grounded Transformer</h1>
    <p>
    <a href="https://github.com/facebookresearch/vggt">🐙 GitHub Repository</a> |
    <a href="#">Project Page</a>
    </p>

    <div style="font-size: 16px; line-height: 1.5;">
    <p>Upload a video or a set of images to create a 3D reconstruction of a scene or object. VGGT takes these images and generates a 3D point cloud, along with estimated camera poses.</p>

    <h3>Getting Started:</h3>
    <ol>
        <li><strong>Upload Data:</strong> Upload a video or image set. Video input is sampled into frames at 1 FPS.</li>
        <li><strong>Reconstruct:</strong> Click "Reconstruct" to run VGGT and generate the GLB preview.</li>
        <li><strong>Continue Steps:</strong> Click "Continue Step" to run stages in order: export PLY, clean PLY, reconstruct STL, compute volume.</li>
        <li><strong>Redo If Needed:</strong> Click "Redo Step" to rerun the latest completed stage if a step fails.</li>
        <li><strong>Read Results:</strong> View final volume in the result panel and download STL/PLY outputs from the file panels.</li>
    </ol>
    <p><strong style="color: #0ea5e9;">Please note:</strong> <span style="color: #0ea5e9; font-weight: bold;">VGGT typically reconstructs a scene in less than 1 second. However, visualizing 3D points may take tens of seconds due to third-party rendering, which are independent of VGGT's processing time. </span></p>
    </div>
    """
    )

    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")
    pipeline_state = gr.State(value={"next_step": -1})

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)

            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                show_download_button=True,
                object_fit="contain",
                preview=True,
            )

        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                log_output = gr.Markdown(
                    "Please upload a video or images, then click Reconstruct.", elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

            with gr.Row():
                mesh_ply_output = gr.File(label="Reconstructed Mesh PLY (download)", visible=True)

            with gr.Row():
                mesh_ply_viewer = gr.Model3D(height=320, label="Reconstructed Mesh Preview (PLY)")

            with gr.Row():
                cube_size_cm = gr.Number(label="Reference Cube Size (cm)", value=14.0, precision=3)

            with gr.Row():
                volume_output = gr.Markdown("Volume not computed.")

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                continue_btn = gr.Button("Continue Step", scale=1, variant="secondary")
                redo_btn = gr.Button("Redo Step", scale=1, variant="secondary")
                clear_btn = gr.ClearButton(
                    [
                        input_video,
                        input_images,
                        reconstruction_output,
                        log_output,
                        target_dir_output,
                        image_gallery,
                        volume_output,
                        mesh_ply_output,
                        mesh_ply_viewer,
                    ],
                    scale=1,
                )

            with gr.Row():
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Pointmap Branch"],
                    label="Select a Prediction Mode",
                    value="Pointmap Branch",
                    scale=1,
                    elem_id="my_radio",
                )

            with gr.Row():
                conf_thres = gr.Slider(minimum=0, maximum=100, value=60, step=0.1, label="Confidence Threshold (%)")
                frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                with gr.Column():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky = gr.Checkbox(label="Filter Sky", value=True)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=True)
                    mask_white_bg = gr.Checkbox(label="Filter White Background", value=True)

    # -------------------------------------------------------------------------
    # "Reconstruct" button logic:
    #  - Clear fields
    #  - Update log
    #  - gradio_demo(...) with the existing target_dir
    #  - Initialize staged pipeline state for Continue/Redo
    # -------------------------------------------------------------------------
    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=reconstruct_with_logs,
        inputs=[
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            cube_size_cm,
        ],
        outputs=[reconstruction_output, log_output, frame_filter, volume_output, mesh_ply_output, mesh_ply_viewer, pipeline_state],
    )

    continue_btn.click(
        fn=continue_pipeline_step_with_logs,
        inputs=[
            pipeline_state,
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            mask_sky,
            prediction_mode,
            cube_size_cm,
        ],
        outputs=[pipeline_state, volume_output, mesh_ply_output, mesh_ply_viewer, log_output],
    )

    redo_btn.click(
        fn=redo_pipeline_step_with_logs,
        inputs=[
            pipeline_state,
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            mask_sky,
            prediction_mode,
            cube_size_cm,
        ],
        outputs=[pipeline_state, volume_output, mesh_ply_output, mesh_ply_viewer, log_output],
    )

    # -------------------------------------------------------------------------
    # Real-time Visualization Updates
    # -------------------------------------------------------------------------
    conf_thres.change(
        update_visualization_with_logs,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        [reconstruction_output, log_output],
    )
    frame_filter.change(
        update_visualization_with_logs,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        [reconstruction_output, log_output],
    )
    mask_black_bg.change(
        update_visualization_with_logs,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        [reconstruction_output, log_output],
    )
    mask_white_bg.change(
        update_visualization_with_logs,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        [reconstruction_output, log_output],
    )
    show_cam.change(
        update_visualization_with_logs,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        [reconstruction_output, log_output],
    )
    mask_sky.change(
        update_visualization_with_logs,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        [reconstruction_output, log_output],
    )
    prediction_mode.change(
        update_visualization_with_logs,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        [reconstruction_output, log_output],
    )

    # -------------------------------------------------------------------------
    # Auto-update gallery whenever user uploads or changes their files
    # -------------------------------------------------------------------------
    input_video.change(
        fn=update_gallery_on_upload_with_logs,
        inputs=[input_video, input_images],
        outputs=[
            reconstruction_output,
            target_dir_output,
            image_gallery,
            log_output,
            volume_output,
            mesh_ply_output,
            mesh_ply_viewer,
            pipeline_state,
        ],
    )
    input_images.change(
        fn=update_gallery_on_upload_with_logs,
        inputs=[input_video, input_images],
        outputs=[
            reconstruction_output,
            target_dir_output,
            image_gallery,
            log_output,
            volume_output,
            mesh_ply_output,
            mesh_ply_viewer,
            pipeline_state,
        ],
    )

    demo.queue(max_size=20).launch(show_error=True, share=True)
