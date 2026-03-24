# New SAM-Based Volume Measurement Pipeline

## Overview

This pipeline measures the real-world volume of a leg (or any target object) using multi-view images, a reference cube of known size, and two AI models:

| Model | Role |
|-------|------|
| **VGGT** (Visual Geometry Grounded Transformer) | 3D reconstruction from images → per-pixel world coordinates |
| **SAM** (Segment Anything Model) | 2D image segmentation → separates box, leg, and floor |

## Pipeline Flow

```
Input Images (with reference cube + leg + floor)
        │
        ▼
┌──────────────────────┐
│  1. VGGT Inference   │  → 3D world points per pixel, camera poses, depth
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  2. SAM Segmentation │  → 2D masks for: box, leg, floor
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  3. Mask → 3D Split  │  → Separate point clouds: box.ply, leg.ply
│     (floor discarded)│     (floor points removed)
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│ 4. Box  │ │ 6. Leg  │
│  Clean  │ │  Clean  │
└────┬────┘ └────┬────┘
     ▼           ▼
┌─────────┐ ┌─────────┐
│ 5. Box  │ │ 7. Leg  │
│  Recon  │ │  Recon  │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌──────────────┐
│ Scale   │ │ 8. Volume    │
│ Factor  │─│  Computation │
│         │ │  (with scale)│
└─────────┘ └──────────────┘

Scale Factor = real_cube_size / measured_cube_size
Leg Volume (cm³) = mesh_volume × scale_factor³ × 1e6
```

## How It Works

### Step 1: VGGT 3D Reconstruction
- Input: Multiple images of the scene from different angles
- Output: Per-pixel 3D world coordinates `(S, H, W, 3)`, confidence maps, camera poses
- The model outputs all frames in a shared world coordinate system

### Step 2: SAM Segmentation (ArUco-guided)
- **ArUco detection** (OpenCV, `DICT_5X5_50`) finds the reference cube's markers → bounding box
- **SAM prompted** with that bounding box → precise box mask
- **SAM auto masks** → largest mask that doesn't overlap the box → target object (leg/bowl)
- Floor / background is implicitly everything not in box or target
- Falls back to 3D geometry heuristics if no ArUco markers are detected

### Step 3: Mask-to-3D Projection
- Maps 2D SAM masks to 3D point cloud regions
- For the segmented frame: direct mask application
- For other frames: 3D bounding-box proximity matching
- Exports separate PLY files per class; floor points are discarded

### Step 4-5: Box Processing
- Cleans box point cloud (outlier removal, downsampling — **plane removal is skipped** since SAM already isolated the object)
- Poisson surface reconstruction → watertight mesh
- Measures bounding box dimensions in mesh units
- Computes: `scale_factor = real_size / mesh_size`

### Step 6-8: Leg/Target Processing
- Cleans target point cloud (plane removal skipped for same reason)
- Poisson surface reconstruction → watertight mesh
- Applies scale factor: `real_volume = mesh_volume × scale³`
- Reports volume in cm³ and m³

## Usage

### Option 1: Gradio Web UI

```bash
python demo_gradio.py
```

1. Upload images/video of the scene (must include reference cube and leg)
2. Click **Reconstruct** → generates GLB 3D preview
3. Click **Continue Step** three times:
   - Step 1/3: SAM segments the scene
   - Step 2/3: Box processed → scale factor computed
   - Step 3/3: Leg processed → volume reported
4. View results in the volume panel; download mesh files

### Option 2: Terminal CLI

```bash
python run_pipeline.py --images ./my_photos/ --cube-size 14.0
```

**Required arguments:**
| Argument | Description |
|----------|-------------|
| `--images` | Directory containing input images |
| `--cube-size` | Real reference cube side length in **cm** |

**Optional arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `pipeline_output` | Output directory |
| `--sam-model` | `vit_b` | SAM model size: `vit_b`, `vit_l`, `vit_h` |
| `--sam-checkpoint` | auto-download | Path to SAM `.pth` checkpoint |
| `--conf-threshold` | `60.0` | Confidence percentile threshold |
| `--seg-frame` | `0` | Which frame to use for SAM segmentation |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |
| `--prediction-mode` | `Pointmap Branch` | VGGT output branch |

**Example with all options:**
```bash
python run_pipeline.py \
    --images ./photos/ \
    --cube-size 14.0 \
    --output-dir ./results/ \
    --sam-model vit_h \
    --seg-frame 2 \
    --conf-threshold 50
```

### Output Files

```
pipeline_output/
├── predictions.npz           # VGGT raw predictions
├── segmentation_overlay.png  # Visualisation of SAM masks
├── results.txt               # Summary metrics
├── box/
│   ├── box_raw.ply           # Raw box point cloud
│   ├── clean/
│   │   └── box_clean.ply     # Cleaned box points
│   └── mesh/
│       ├── mesh_box.ply      # Reconstructed box mesh
│       └── mesh_box.stl
└── leg/
    ├── leg_raw.ply           # Raw leg point cloud
    ├── clean/
    │   └── leg_clean.ply     # Cleaned leg points
    └── mesh/
        ├── mesh_leg.ply      # Reconstructed leg mesh
        └── mesh_leg.stl      # STL for 3D printing / CAD
```

## Scale Calibration Example

```
Real cube = 14 cm
Box mesh bounding box median side = 0.30 mesh units

Scale factor = 0.14 m / 0.30 units = 0.4667 m/unit

Leg mesh raw volume = 0.05 mesh-units³
Leg real volume = 0.05 × (0.4667)³ = 0.00509 m³ = 5,086 cm³
```

## Requirements

Additional dependency beyond the base project:

```bash
pip install segment-anything
```

SAM checkpoints are auto-downloaded on first use (~375 MB for `vit_b`, ~1.2 GB for `vit_l`, ~2.4 GB for `vit_h`). Or download manually:

```bash
# vit_b (fastest, recommended for testing)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/

# vit_h (most accurate)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P checkpoints/
```

## Tips

- **Image quality matters**: Clear, well-lit photos from multiple angles produce better 3D reconstructions.
- **Reference cube placement**: Place the cube close to the leg, visible in most frames.
- **SAM frame selection**: Use `--seg-frame` to pick a frame where all three objects (box, leg, floor) are clearly visible.
- **Confidence threshold**: Lower values include more points (noisier); higher values are stricter (cleaner but may miss detail).
- **SAM model choice**: `vit_b` is fast and sufficient for most scenes; `vit_h` is more accurate for complex scenes.
