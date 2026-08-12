"""Stage 1 — Load VGGT model and run inference to produce a predictions dict."""
import glob
import os
import sys
import time

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images, load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.device import is_mps, autocast_on, aggressive_cleanup

from pipeline.config import (
    DEFAULT_MAX_FRAMES_MPS,
    IMAGE_EXTENSIONS,
    VGGT_COMMERCIAL_FILE,
    VGGT_COMMERCIAL_REPO,
    VGGT_MODEL_URL,
    VGGT_USE_COMMERCIAL,
)


def _load_weights():
    """Fetch the VGGT state dict, preferring the commercially licensed checkpoint.

    The commercial repo is gated, so this needs an accepted licence plus a token
    in HF_TOKEN / HUGGINGFACE_HUB_TOKEN (or `huggingface-cli login`). Without one
    the download fails, and silently continuing on the CC BY-NC-SA checkpoint
    would hand back a non-commercial model — so the fallback is loud.
    """
    if VGGT_USE_COMMERCIAL:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(VGGT_COMMERCIAL_REPO, VGGT_COMMERCIAL_FILE)
            print(f"  Checkpoint: {VGGT_COMMERCIAL_REPO} (commercial licence)")
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"  WARNING: commercial checkpoint unavailable ({type(e).__name__}: "
                  f"{str(e)[:120]})")
            print("  WARNING: falling back to facebook/VGGT-1B — CC BY-NC-SA 4.0, "
                  "NOT licensed for commercial use.")

    print("  Checkpoint: facebook/VGGT-1B (non-commercial)")
    return torch.hub.load_state_dict_from_url(VGGT_MODEL_URL, map_location="cpu")


def _select_frames(image_names, max_frames):
    """Uniformly subsample frames if there are more than max_frames, keeping first and last."""
    n = len(image_names)
    if max_frames is None or n <= max_frames:
        return image_names
    indices = np.linspace(0, n - 1, max_frames, dtype=int)
    indices = sorted(set(indices))
    return [image_names[i] for i in indices]


def run_inference(image_folder, device, max_frames=None, preprocess_mode="crop",
                  input_res=518):
    """Load model, run inference, return predictions dict (numpy) and timings.

    preprocess_mode:
        "crop" — width to 518, centre-crop the height. On 9:16 phone photos this
                 discards ~44% of every frame, which can amputate the subject.
        "pad"  — fit the whole frame inside 518 and pad. Keeps all content at
                 lower effective resolution.

    input_res:
        518 is VGGT's native size. Anything else routes through the square
        loader (black-padded, so pad-style regardless of preprocess_mode) and
        must be divisible by 14 — 1022 works, 1024 does not. Token count grows
        with res², and global attention across frames grows with its square, so
        1022 costs roughly 15x the attention memory of 518.
    """
    t0 = time.time()
    print("=" * 60)
    print("STAGE 1: Loading model and running inference")
    print("=" * 60)

    aggressive_cleanup(device)

    # Auto-limit frames on MPS to avoid OOM
    # 9 frames @ 518×518 → global attention over 12k tokens → ~4.9GB just for attn scores
    # 7 frames → ~3.0GB → fits in 30GB MPS with model + activations
    if max_frames is None and is_mps(device):
        max_frames = DEFAULT_MAX_FRAMES_MPS
        print(f"  MPS detected: auto-limiting to {max_frames} frames (override with --max_frames)")

    print("  Loading VGGT model...")
    model = VGGT()
    model.load_state_dict(_load_weights())
    model.eval()
    model = model.to(device)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    image_names = sorted(glob.glob(os.path.join(image_folder, "*")))
    image_names = [p for p in image_names if p.lower().endswith(IMAGE_EXTENSIONS)]
    if not image_names:
        print(f"ERROR: No images found in {image_folder}")
        sys.exit(1)

    original_count = len(image_names)
    image_names = _select_frames(image_names, max_frames)
    if len(image_names) < original_count:
        print(f"  Found {original_count} images → selected {len(image_names)} (uniformly spaced)")
    else:
        print(f"  Found {len(image_names)} images")

    if int(input_res) == 518:
        images = load_and_preprocess_images(image_names, mode=preprocess_mode).to(device)
        mode_label = preprocess_mode
    else:
        if int(input_res) % 14 != 0:
            print(f"ERROR: input_res {input_res} is not divisible by 14 "
                  f"(nearest valid: {round(input_res / 14) * 14})")
            sys.exit(1)
        images, _coords = load_and_preprocess_images_square(image_names,
                                                            target_size=int(input_res))
        images = images.to(device)
        mode_label = "square/pad"
    print(f"  Preprocessed shape: {images.shape}  (mode={mode_label}, res={input_res})")

    t1 = time.time()
    print("  Running inference...")
    with torch.no_grad():
        with autocast_on(device):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert all tensors to numpy
    for key in list(predictions.keys()):
        v = predictions[key]
        if isinstance(v, torch.Tensor):
            predictions[key] = v.cpu().float().numpy().squeeze(0)
        elif isinstance(v, list):
            predictions[key] = None
    predictions["pose_enc_list"] = None

    depth_map = predictions["depth"]
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"]
    )

    inference_time = time.time() - t1
    print(f"  Inference done in {inference_time:.1f}s")

    del model
    aggressive_cleanup(device)

    return predictions, inference_time
