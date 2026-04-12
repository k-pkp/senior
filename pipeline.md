# VGGT 3D Reconstruction Pipeline

End-to-end pipeline for reconstructing 3D meshes from multi-view images using the VGGT (Visual Geometry Grounded Transformer) model, then computing real-world volumes using a reference cube.

## Pipeline Stages

```
Images → [1] Inference → [2] PLY Export → [3] Clean & Extract
       → [4] Poisson Reconstruction → [5] Watertight Repair
       → [6] Multi-Perspective Evaluation → [7] Volume Computation
```

### Stage 1: Model Inference
- **File**: `run.py` → `run_inference()`
- **Model**: VGGT-1B (`facebook/VGGT-1B` from HuggingFace)
- **Input**: Folder of images (JPG/PNG)
- **Output**: Predictions dict with `world_points`, `world_points_conf`, `depth`, `depth_conf`, `extrinsic`, `intrinsic`
- **Device**: Auto-detects CUDA, MPS, or CPU
- **Frame limit**: Auto-limits to 6 frames on MPS to avoid OOM

### Stage 2: PLY Point Cloud Export
- **File**: `run.py` → `export_ply()`
- **Process**:
  1. Adaptive confidence filtering (percentile-based, distribution-aware)
  2. Optional background masking (`--mask_black_bg`, `--mask_white_bg`)
  3. Statistical outlier removal (Open3D)
- **Output**: `output/points.ply` — colored point cloud

### Stage 3: Clean & Extract Objects
- **File**: `clean/clean_ply.py` → `clean_and_extract_objects()`
- **Process**:
  1. Statistical outlier removal (adaptive std_ratio by density)
  2. Voxel downsampling (voxel=0.002)
  3. RANSAC plane removal (`distance_threshold=0.015`)
  4. DBSCAN clustering with adaptive epsilon
  5. Select top-k clusters by score (points × density)
- **Output**: `output/clean_objects/object_N.ply` (one per object)

### Stage 4: Poisson Reconstruction (non-watertight)
- **File**: `clean/recons.py` → `reconstruct_multiple_objects()` → subprocess `recons_worker.py`
- **Process per object** (runs in an isolated subprocess for stability):
  1. Downsample to ≤ 165K points (Poisson stability ceiling)
  2. Normal estimation with tangent-plane orientation (progressive k fallback: 50 → 20 → 10 → camera-based)
  3. Poisson reconstruction with depth fallback 9 → 8 → 7 (each depth runs in a sub-subprocess to survive C++ crashes)
  4. Density filter at 5% quantile (removes far-field artifacts)
  5. Cleanup (degenerate, duplicate, non-manifold triangles)
  6. **Largest-connected-component filter** — keeps the main surface only, removing noise fragments that would confuse the watertight stage
- **Output per object**: `output/mesh/object_N_recon.ply` and `.stl`
- **Scene output**: `output/mesh/scene_recon.ply` and `.stl` (merged)

### Stage 5: Watertight Repair (default; skip with `--no-watertight`)
- **File**: `clean/recons.py` → `make_watertight_meshes()` → subprocess `meshfix_worker.py`
- **Process per object**:
  1. PyMeshFix repair (Attene 2010: "A lightweight approach to repairing digitized polygon meshes") — removes self-intersections, fills holes/base, enforces manifold topology
  2. Vertex color transfer from the recon mesh via nearest-neighbor KDTree lookup
  3. Runs in a separate subprocess to isolate PyMeshFix/pyvista from Open3D
  4. If MeshFix fails, copies the recon mesh as a non-watertight fallback
- **Output per object**: `output/mesh/object_N.ply` and `.stl` (watertight)
- **Scene output**: `output/mesh/scene.ply` and `.stl` (merged watertight scene)

### Stage 6: Multi-Perspective Evaluation
- **File**: `run.py` → `evaluate_with_viewer()`, `viewer.py`
- **Process**: 38 screenshots per output using Open3D OffscreenRenderer (headless-capable):
  - **36 orbital views**: 3 elevations (high, mid, low) × 12 azimuth angles (every 30°)
  - **Top** and **bottom** views
- **Output**: `output/evaluation/`
  - `pointcloud/` — 38 views of the raw point cloud
  - `scene/` — 38 views of the watertight scene
  - `object_N/` — 38 views per object
- **Enabled by**: `--evaluate` flag

### Stage 7: Real-World Volume Computation
- **File**: `run.py` → `compute_volumes()` (also standalone: `volume.py`)
- **Input**: Watertight object meshes (falls back to convex hull if non-watertight)
- **Formula** (uses reference cube of known real size):
  ```
  mesh_bbox_vol_ref  = X_ref × Y_ref × Z_ref       # product of ref bbox extents
  real_ref_vol       = real_size³                   # e.g. 14³ = 2744 cm³
  k                  = real_ref_vol / mesh_bbox_vol_ref

  # For any object i:
  real_X_i = mesh_X_i × k^(1/3)
  real_Y_i = mesh_Y_i × k^(1/3)
  real_Z_i = mesh_Z_i × k^(1/3)
  real_volume_i = real_X_i × real_Y_i × real_Z_i   # = mesh_bbox_vol_i × k
  ```
- **Configuration**: Edit the constants at the top of the Stage 7 section in `run.py`:
  ```python
  REFERENCE_OBJECT_INDEX = 1        # which object is the reference
  REFERENCE_REAL_SIZE_CM = 14.0     # real edge length of the reference cube
  ```

## Usage

```bash
# Full pipeline (watertight by default)
conda run -n vggt python run.py --image_folder ./baam/ --evaluate

# Skip watertight repair — export only the Poisson recon mesh
conda run -n vggt python run.py --image_folder ./baam/ --evaluate --no-watertight

# PLY only (skip mesh reconstruction + watertight)
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
| `--skip_mesh` | off | Skip stages 3-5, export PLY only |
| `--num_objects` | `2` | Number of objects to extract |
| `--max_frames` | auto | Max input frames (auto=6 on MPS) |
| `--evaluate` | off | Capture 38-view screenshots |
| `--no-watertight` | off | Skip Stage 5 (export only recon meshes) |

## Output Structure

```
output/
  points.ply                         # Full filtered point cloud
  predictions.npz                    # Raw model predictions
  clean_objects/
    object_N.ply                     # Cleaned point cloud (per object)
  mesh/
    scene_recon.ply / .stl           # Merged Poisson recon (non-watertight)
    object_N_recon.ply / .stl        # Per-object Poisson recon
    scene.ply / .stl                 # Merged watertight scene
    object_N.ply / .stl              # Per-object watertight mesh
  evaluation/
    pointcloud/                      # 38 views
    scene/                           # 38 views
    object_N/                        # 38 views per object
  target/
    predictions.npz                  # Copy for demo_gradio compatibility
    images/                          # Copy of input images
```

## Interactive Viewing & Volume Tools

```bash
# View any mesh or point cloud (interactive Open3D window)
conda run -n vggt python viewer.py output/mesh/object_0.ply

# Print mesh info (watertight status, vertex/face count, bounds)
conda run -n vggt python viewer.py output/mesh/object_0.ply --info

# Overlay multiple files
conda run -n vggt python viewer.py output/points.ply output/mesh/scene.ply

# Batch 38-view screenshots (headless-capable)
conda run -n vggt python viewer.py output/mesh/object_0.ply --multi-view --screenshot my_views/

# Standalone volume tool (interactive reference selection)
conda run -n vggt python volume.py output/mesh/object_0.ply output/mesh/object_1.ply

# Volume with explicit reference (no interaction)
conda run -n vggt python volume.py output/mesh/object_0.ply output/mesh/object_1.ply \
    --ref-index 1 --ref-size 14
```

## Dependencies

Core:
- `torch` ≥ 2.5.1, `torchvision` ≥ 0.20.1
- `numpy`, `open3d`, `trimesh`, `scipy`, `opencv-python`

Watertight repair:
- `pymeshfix` — mesh hole filling and manifold repair
- `pyvista` — mesh I/O for PyMeshFix

## Key Files

| File | Purpose |
|---|---|
| `run.py` | Main pipeline orchestrator (Stages 1-7) |
| `volume.py` | Standalone volume-from-reference tool |
| `viewer.py` | 3D viewer + 38-view screenshot tool |
| `clean/clean_ply.py` | Point cloud cleaning and object extraction (Stage 3) |
| `clean/recons.py` | Stage 4 + Stage 5 orchestration |
| `clean/recons_worker.py` | Subprocess worker for Poisson reconstruction |
| `clean/meshfix_worker.py` | Subprocess worker for PyMeshFix + color transfer |
| `vggt/` | VGGT model and utilities |

## Design Notes

- **Subprocess isolation**: Open3D, PyMeshFix, and pyvista each link against their own OpenGL/VTK stacks that conflict when combined in the same process. Each unstable step (Poisson, normal orientation, MeshFix) runs in an isolated subprocess so a segfault in one doesn't break the whole pipeline.
- **Largest-component filter**: Open3D's Poisson + density filter produces hundreds of disconnected fragments. Keeping only the largest component before MeshFix gives it a clean single surface to close (holes at base, top), instead of forcing it to discard real geometry as noise.
- **Color preservation**: MeshFix strips vertex colors during repair. We transfer them back from the recon mesh via nearest-neighbor KDTree lookup inside the MeshFix worker subprocess.
- **Watertight is default**: `--no-watertight` opts out. When on, `object_N.ply` is watertight; when off, it's the raw Poisson output with density filter (better visual detail but has boundary holes).
