import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from rembg import remove 
import numpy as np
# =========================
# LOAD MODEL
# =========================
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=500
)

# =========================
# LOAD IMAGE
# =========================
image = cv2.imread("input/IMG_4025.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# =========================
# GENERATE ALL MASKS
# =========================
print("🔄 Generating masks...")
masks = mask_generator.generate(image_rgb)

print(f"🔍 Found {len(masks)} masks")

# =========================
# FILTER MASKS
# =========================
h, w, _ = image.shape
final_mask = np.zeros((h, w), dtype=bool)

for m in masks:
    seg = m["segmentation"]
    area = m["area"]

    # 🔹 Rule 1: keep large objects
    if area < 2000:
        continue

    # 🔹 Rule 2: keep objects near center (leg + cube likely here)
    y_indices, x_indices = np.where(seg)
    cx = np.mean(x_indices)
    cy = np.mean(y_indices)

    if 0.2*w < cx < 0.8*w and 0.2*h < cy < 0.9*h:
        final_mask |= seg

# =========================
# APPLY MASK
# =========================
white_bg = np.ones_like(image) * 255
output = np.where(final_mask[..., None], image, white_bg)

# load image 
img = output
# remove background 
result = remove(img) 
result2= remove(result) 
# save (transparent PNG) 
cv2.imwrite("output/no_bg5.png" , result2)
print("✅ Background removed")