import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from rembg import remove
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def _build_mask_generator(checkpoint_path: str) -> SamAutomaticMaskGenerator:
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=500,
    )


def _segment_single_image(image: np.ndarray, mask_generator: SamAutomaticMaskGenerator) -> np.ndarray:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image_rgb)

    h, w, _ = image.shape
    final_mask = np.zeros((h, w), dtype=bool)

    for mask_info in masks:
        seg = mask_info["segmentation"]
        area = mask_info["area"]

        if area < 2000:
            continue

        y_indices, x_indices = np.where(seg)
        cx = np.mean(x_indices)
        cy = np.mean(y_indices)

        if 0.2 * w < cx < 0.8 * w and 0.2 * h < cy < 0.9 * h:
            final_mask |= seg

    white_bg = np.ones_like(image, dtype=np.uint8) * 255
    masked = np.where(final_mask[..., None], image, white_bg)

    rgba = remove(masked)
    if rgba.shape[-1] == 4:
        alpha = (rgba[:, :, 3:4] / 255.0).astype(np.float32)
        rgb = rgba[:, :, :3].astype(np.float32)
        blended = rgb * alpha + 255.0 * (1.0 - alpha)
        return blended.astype(np.uint8)

    return rgba


def _iter_image_files(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for item in sorted(folder.iterdir()):
        if item.is_file() and item.suffix.lower() in exts:
            yield item


def process_numbered_inputs(input_root: str, output_root: str, checkpoint_path: str) -> list[str]:
    input_root_path = Path(input_root)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    mask_generator = _build_mask_generator(checkpoint_path)
    saved_outputs: list[str] = []

    input_dirs = sorted([p for p in input_root_path.iterdir() if p.is_dir() and p.name.startswith("input")])
    for input_dir in input_dirs:
        idx = input_dir.name.replace("input", "")
        output_dir = output_root_path / f"output{idx}"
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(_iter_image_files(input_dir))
        if not image_files:
            print(f"[WARN] No image found in {input_dir}")
            continue

        src_img = image_files[0]
        image = cv2.imread(str(src_img))
        if image is None:
            print(f"[WARN] Failed to read image: {src_img}")
            continue

        segmented = _segment_single_image(image, mask_generator)
        output_path = output_dir / "segmented.png"
        cv2.imwrite(str(output_path), segmented)
        saved_outputs.append(str(output_path))
        print(f"[OK] {src_img} -> {output_path}")

    return saved_outputs


def clear_folder(folder: str) -> None:
    folder_path = Path(folder)
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_root = str(base_dir / "input")
    output_root = str(base_dir / "output")
    checkpoint = str(base_dir / "sam_vit_b_01ec64.pth")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

    saved = process_numbered_inputs(input_root=input_root, output_root=output_root, checkpoint_path=checkpoint)
    print(f"Done. Saved {len(saved)} segmented image(s).")