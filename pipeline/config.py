"""Pipeline-wide configuration constants."""

# Stage 7 volume calibration
# Hard-coded: object at REFERENCE_OBJECT_INDEX is the ArUco marker.
# The unknown object whose volume we want to measure is at the other index.
REFERENCE_OBJECT_INDEX = 1        # which object is the ArUco reference
REFERENCE_REAL_SIZE_CM = 14.0     # real linear size of reference in cm

# Stage 1 frame limits
DEFAULT_MAX_FRAMES_MPS = 6

# Model weights
VGGT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

# Image extensions accepted by the input loader
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
