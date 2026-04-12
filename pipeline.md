# VGGT 3D Reconstruction Pipeline

End-to-end pipeline for reconstructing 3D meshes from multi-view images using the VGGT (Visual Geometry Grounded Transformer) model.

## Pipeline Stages

```
Input Images → [Stage 1] Inference → [Stage 2] PLY Export → [Stage 3] Clean & Extract → [Stage 4] Reconstruct & Repair → [Stage 5] Evaluate
```

### Stage 1: Model Inference
- **File**: `run.py` → `run_inference()`
- **Model**: VGGT-1B (facebook/VGGT-1B from HuggingFace)
- **Input**: Folder of images (JPG/PNG)
- **Output**: Predictions dict containing:
  - `world_points` — direct 3D point regression (S, H, W, 3)
  - `world_points_conf` — per-point confidence scores
  - `depth` / `depth_conf` — depth maps and confidence
  - `extrinsic` / `intrinsic` — camera parameters
- **Device**: Auto-detects CUDA, MPS (Apple Silicon), or CPU
- **Frame limit**: Auto-limits to 6 frames on MPS to avoid OOM

### Stage 2: PLY Point Cloud Export
- **File**: `run.py` → `export_ply()`
- **Process**:
  1. Adaptive confidence filtering (percentile-based with distribution-aware adjustment)
  2. Optional background masking (black/white)
  3. Statistical outlier removal (Open3D, k=20, std_ratio=2.5)
- **Output**: `output/points.ply` — colored point cloud

### Stage 3: Clean & Extract Objects
- **File**: `clean/clean_ply.py` → `clean_and_extract_objects()`
- **Process**:
  1. Statistical outlier removal (std_ratio=2.0)
  2. Voxel downsampling (voxel=0.002)
  3. RANSAC plane removal (removes dominant ground plane)
  4. DBSCAN clustering with adaptive epsilon
  5. Object selection: top-k clusters by score (points x density)
- **Output**: `output/clean_objects/object_0.ply`, `object_1.ply`, ...

### Stage 4: Reconstruct & Watertight Repair
- **File**: `clean/recons.py` → `reconstruct_multiple_objects()`
- **Process** (per object):
  1. **Poisson Surface Reconstruction** (Open3D)
     - Adaptive depth: 7-11 based on point count
     - Automatic downsampling for >300k points
     - Normal estimation with adaptive radius
     - Fallback depth reduction if Poisson fails
  2. **Density filtering** — removes low-density Poisson artifacts (1st percentile)
  3. **Mesh cleanup** — degenerate/duplicate triangles, non-manifold edges
  4. **Watertight repair** (optional, `--watertight` flag)
     - Bbox crop instead of density filter keeps the Poisson mesh connected
     - PyMeshFix (Attene 2010) closes gaps and base
     - Vertex colors transferred back via nearest-neighbor KDTree lookup (in subprocess)
     - Normal estimation + orientation run in subprocess for stability
- **Output per object**: `output/mesh/object_0.ply`, `object_0.stl`
- **Scene output**: `output/mesh/scene.ply`, `scene.stl` (merged)

### Stage 5: Multi-Perspective Evaluation
- **File**: `run.py` → `evaluate_with_viewer()`, `viewer.py`
- **Process**: Captures 16 camera perspectives for each output:
  - 8 orbital views at mid elevation (every 45 degrees)
  - 4 high-angle views (steep downward, every 90 degrees)
  - 2 low-angle views (front and back, slightly upward)
  - Top and bottom views
- **Uses**: Open3D OffscreenRenderer (headless-compatible)
- **Output**: `output/evaluation/` with subdirectories:
  - `pointcloud/` — 6 views of the raw point cloud
  - `scene/` — 6 views of the merged scene mesh
  - `object_0/`, `object_1/`, ... — 6 views per object mesh

## Usage

```bash
# Full pipeline — preserves color and surface detail (default)
conda run -n vggt python run.py --evaluate

# Watertight meshes — for 3D printing (loses color/detail)
conda run -n vggt python run.py --evaluate --watertight

# Custom input folder
conda run -n vggt python run.py --image_folder ./my_images/ --evaluate

# PLY only (skip mesh reconstruction)
conda run -n vggt python run.py --skip_mesh

# Adjust confidence threshold (lower = more points)
conda run -n vggt python run.py --conf_thres 30 --evaluate

# Use depth-based unprojection instead of pointmap
conda run -n vggt python run.py --prediction_mode depth --evaluate
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--image_folder` | `./baam/` | Input image directory |
| `--output_dir` | `./output/` | Output directory |
| `--conf_thres` | `45.0` | Confidence filter percentile (0-100) |
| `--prediction_mode` | `pointmap` | `pointmap` or `depth` |
| `--mask_black_bg` | off | Mask dark background pixels |
| `--mask_white_bg` | off | Mask bright background pixels |
| `--skip_mesh` | off | Skip stages 3-4, export PLY only |
| `--num_objects` | `2` | Number of objects to extract |
| `--max_frames` | auto | Max input frames (auto=6 on MPS) |
| `--evaluate` | off | Capture multi-perspective screenshots |
| `--watertight` | off | Make meshes watertight via PyMeshFix (loses color/detail) |

## Output Structure

```
output/
  points.ply                    # Full filtered point cloud
  predictions.npz               # Raw model predictions
  clean_objects/
    object_0.ply                # Cleaned point cloud (object 0)
    object_1.ply                # Cleaned point cloud (object 1)
  mesh/
    scene.ply / scene.stl       # Merged scene mesh
    object_0.ply / object_0.stl # Object 0 mesh (watertight if --watertight)
    object_1.ply / object_1.stl # Object 1 mesh (watertight if --watertight)
  evaluation/
    pointcloud/                 # 6-view screenshots of point cloud
    scene/                      # 6-view screenshots of scene mesh
    object_0/                   # 6-view screenshots of object 0
    object_1/                   # 6-view screenshots of object 1
  target/
    predictions.npz             # Copy for demo_gradio compatibility
    images/                     # Copy of input images
```

## Interactive Viewing

```bash
# View point cloud
conda run -n vggt python viewer.py output/points.ply

# View a specific object mesh
conda run -n vggt python viewer.py output/mesh/object_0.ply

# View scene mesh
conda run -n vggt python viewer.py output/mesh/scene.ply

# Print mesh info
conda run -n vggt python viewer.py output/mesh/object_0.ply --info

# Custom multi-view screenshots
conda run -n vggt python viewer.py output/mesh/object_0.ply --multi-view --screenshot my_views/
```

## Dependencies

Core:
- `torch` >= 2.5.1, `torchvision` >= 0.20.1
- `numpy`, `open3d`, `trimesh`, `scipy`, `opencv-python`

Watertight repair:
- `pymeshfix` — mesh hole filling and manifold repair
- `pyvista` — mesh I/O for PyMeshFix

## Key Files

| File | Purpose |
|---|---|
| `run.py` | Main pipeline orchestrator |
| `viewer.py` | 3D viewer with multi-perspective screenshot support |
| `clean/clean_ply.py` | Point cloud cleaning and object extraction |
| `clean/recons.py` | Poisson reconstruction + watertight repair |
| `clean/meshfix_worker.py` | Subprocess worker for PyMeshFix (OpenGL isolation) |
| `vggt/` | VGGT model and utilities |
