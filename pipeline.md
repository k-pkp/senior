# VGGT 3D Reconstruction Pipeline

End-to-end pipeline for reconstructing watertight 3D meshes from multi-view
images using the VGGT (Visual Geometry Grounded Transformer) model and
computing real-world volumes via an ArUco scale reference.

## Pipeline Stages

```
Input Images
  → [1] Inference
  → [2] PLY Export
  → [3] Clean & Extract
  → [4] Poisson Reconstruction
  → [5] Watertight Repair
  → [6] Multi-view Evaluation
  → [7] Real-world Volume
```

### Stage 1: Model Inference
- **File**: `run.py` → `run_inference()`
- **Model**: VGGT-1B (`facebook/VGGT-1B` from HuggingFace)
- **Input**: Folder of images (JPG/PNG/BMP/TIFF/WEBP)
- **Output**: Predictions dict
  - `world_points` (S, H, W, 3) — direct 3D point regression
  - `world_points_conf` — per-point confidence
  - `depth` / `depth_conf` — depth maps and confidence
  - `extrinsic` / `intrinsic` — camera parameters
  - `world_points_from_depth` — unprojected depth-based points
- **Device**: Auto-detects CUDA, MPS, or CPU
- **Frame limit**: Auto-limits to 6 frames on MPS to avoid OOM

### Stage 2: PLY Point Cloud Export
- **File**: `run.py` → `export_ply()`
- **Process**:
  1. Adaptive confidence filter (percentile + distribution-aware absolute threshold)
  2. Optional background masking (`--mask_black_bg`, `--mask_white_bg`)
  3. Statistical outlier removal (Open3D, k=20, std_ratio=2.5)
- **Output**: `output/points.ply` — colored point cloud

### Stage 3: Clean & Extract Objects
- **File**: `clean/clean_ply.py` → `clean_and_extract_objects()`
- **Process**:
  1. Statistical outlier removal (std_ratio=2.0)
  2. Voxel downsampling (voxel=0.002)
  3. RANSAC plane removal (drops dominant ground plane)
  4. DBSCAN clustering with adaptive epsilon
  5. Top-k selection by score (points × density)
- **Output**: `output/clean_objects/object_0.ply`, `object_1.ply`, …

### Stage 4: Poisson Reconstruction (non-watertight)
- **File**: `clean/recons.py` → `reconstruct_multiple_objects()`
  → `clean/recons_worker.py` (subprocess)
- **Process** (per object):
  1. Adaptive Poisson depth 7–11 by point count
  2. Auto-downsample for >300k points
  3. Adaptive normal estimation
  4. Density filter (low-density artifact removal)
  5. Cleanup: degenerate / duplicate triangles, non-manifold edges
- **Output per object**: `output/mesh/object_N_recon.ply`, `object_N_recon.stl`
- **Scene merge**: `output/mesh/scene_recon.ply`, `scene_recon.stl`

### Stage 5: Watertight Repair
- **File**: `clean/recons.py` → `make_watertight_meshes()`
  → `clean/meshfix_worker.py` (subprocess)
- **Process** (per recon mesh):
  1. **PyMeshFix `MeshFix.fill_holes()`** — closes boundary loops, never deletes faces
     (shape preserved; retention typically >100% as faces are added)
  2. **Open3D `TriangleMesh.fill_holes()`** fallback for residual gaps (deterministic earcut)
  3. **Color transfer** — cKDTree nearest-neighbor from recon mesh vertices
  4. Subprocess exit codes: `0` watertight, `2` written-not-watertight, `1` hard fail
  5. Up to **3 retries** on hard fail (transient C++ crash isolation)
- **Verification**: `trimesh.load(path, process=False)` + `is_watertight`
  (`process=False` is required — default merge welds intentional PyMeshFix
  seam duplicates and breaks edge-manifoldness)
- **Output per object**: `output/mesh/object_N.ply`, `object_N.stl` (watertight)
- **Scene merge (geometry only)**: `output/mesh/scene.ply`, `scene.stl`
- **Scene merge (with vertex colors)**: `output/mesh/scene_colour.ply` +
  `scene_colour.stl` (`trimesh.util.concatenate` of per-object watertight meshes,
  `process=False` to keep PyMeshFix seam dups intact and preserve per-object
  watertightness; STL is geometry-only — colors not stored in the format)
- **Determinism**: PyMeshFix + Open3D `fill_holes` are pure C++ geometric algorithms,
  no RNG → identical md5 across repeated runs (verified ×3)

### Stage 6: Multi-Perspective Evaluation
- **File**: `run.py` → `evaluate_with_viewer()`, `viewer.py`
- **Process**: Captures **38 screenshots** per output:
  - 3 elevation rings × 12 azimuths (every 30°) = 36 orbital views
  - 1 top + 1 bottom view
- **Renderer**: Open3D `OffscreenRenderer` (EGL headless on Linux)
- **Output** (`output/evaluation/`):
  - `pointcloud/` — point cloud (38 views)
  - `scene/` — merged scene mesh (38 views)
  - `object_0/`, `object_1/`, … — per-object meshes (38 views each)

### Stage 7: Real-world Volume (ArUco-referenced)
- **File**: `run.py` → `compute_volumes()`
- **Reference**: `object_1` is the ArUco marker, a **14 × 14 × 14 cm cube**
  (configurable via `REFERENCE_OBJECT_INDEX`, `REFERENCE_REAL_SIZE_CM`)
- **Formula**:
  ```
  k          = REAL_VOL / mesh_bbox_vol_ref      # volume scale
  k^(1/3)    = linear scale
  real_X     = mesh_X * k^(1/3)
  real_vol_i = mesh_vol_i * k                    # cm^3
  ```
- **Volume source**: `mesh.volume` if watertight, else `convex_hull.volume`
- **Mesh load**: `trimesh.load(path, force="mesh", process=False)` (same reason as Stage 5)

## Usage

```bash
# Full pipeline (default — watertight ON, evaluation OFF)
conda run -n vggt python run.py

# Full pipeline + multi-view screenshots
conda run -n vggt python run.py --evaluate

# Skip watertight repair (keep Poisson recon as final mesh)
conda run -n vggt python run.py --no-watertight

# Custom input folder
conda run -n vggt python run.py --image_folder ./my_images/ --evaluate

# PLY only — skip stages 3–7
conda run -n vggt python run.py --skip_mesh

# Lower confidence threshold (more points retained)
conda run -n vggt python run.py --conf_thres 30 --evaluate

# Depth-based unprojection instead of pointmap regression
conda run -n vggt python run.py --prediction_mode depth --evaluate
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--image_folder` | `./baam/` | Input image directory |
| `--output_dir` | `./output/` | Output directory |
| `--conf_thres` | `45.0` | Confidence filter percentile (0–100) |
| `--prediction_mode` | `pointmap` | `pointmap` or `depth` |
| `--mask_black_bg` | off | Mask dark background pixels |
| `--mask_white_bg` | off | Mask bright background pixels |
| `--skip_mesh` | off | Skip stages 3–7, export PLY only |
| `--num_objects` | `2` | Number of objects to extract |
| `--max_frames` | auto | Max input frames (auto=6 on MPS) |
| `--evaluate` | off | Capture multi-perspective screenshots |
| `--no-watertight` | off | Skip Stage 5; final mesh is Poisson recon only |
| `--seed` | `42` | Seed for `random`, NumPy, PyTorch, Open3D |

## Output Structure

```
output/
  points.ply                       # Filtered point cloud (Stage 2)
  predictions.npz                  # Raw model predictions
  clean_objects/
    object_0.ply / object_1.ply    # Cleaned point clouds (Stage 3)
  mesh/
    object_N_recon.ply / .stl      # Poisson recon (Stage 4, non-watertight)
    scene_recon.ply / .stl         # Merged recon scene
    object_N.ply / .stl            # Watertight repair (Stage 5)
    scene.ply / .stl               # Merged watertight scene (no colors)
    scene_colour.ply / .stl        # Merged watertight scene (PLY: vertex colors; STL: geometry only)
  evaluation/                      # Stage 6 (only if --evaluate)
    pointcloud/                    # 38 views
    scene/                         # 38 views
    object_0/, object_1/, …        # 38 views per object
  target/
    predictions.npz                # Copy for demo_gradio compatibility
    images/                        # Copy of input images
```

## Interactive Viewing

```bash
# Point cloud
conda run -n vggt python viewer.py output/points.ply

# Watertight object mesh
conda run -n vggt python viewer.py output/mesh/object_0.ply

# Scene mesh
conda run -n vggt python viewer.py output/mesh/scene.ply

# Mesh info (vertices, faces, watertight, bounds)
conda run -n vggt python viewer.py output/mesh/object_0.ply --info

# Multi-view screenshots
conda run -n vggt python viewer.py output/mesh/object_0.ply --multi-view --screenshot my_views/
```

## Determinism

All stages run with the seed from `--seed` (default `42`):

- `random`, `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`,
  `open3d.utility.random.seed` are all set
- `viewer.py` uses an internal `np.random.default_rng(42)` for any subsampling
- Stage 5 PyMeshFix + Open3D `fill_holes` have no RNG; identical input bytes
  produce byte-identical output (verified by md5 across 3 sequential runs)

Reproducibility check (Stage 5 only, recon meshes already on disk):

```bash
for i in 1 2 3; do
  conda run -n vggt python -c "
import sys; sys.path.insert(0,'clean')
from recons import make_watertight_meshes
make_watertight_meshes(['output/mesh/object_0_recon.ply',
                        'output/mesh/object_1_recon.ply'],
                       output_folder=f'/tmp/wt_run$i', base_name='scene')"
done
md5sum /tmp/wt_run{1,2,3}/object_0.ply
md5sum /tmp/wt_run{1,2,3}/object_1.ply
```

## Dependencies

Core:
- `torch` ≥ 2.5.1, `torchvision` ≥ 0.20.1
- `numpy`, `open3d`, `trimesh`, `scipy`, `opencv-python`, `networkx`,
  `matplotlib`, `Pillow`

Watertight repair:
- `pymeshfix` (≥ 0.18) — boundary hole filling
- `open3d` ≥ 0.19 — `t.geometry.TriangleMesh.fill_holes` fallback

## Key Files

| File | Purpose |
|---|---|
| `run.py` | Main pipeline orchestrator (Stages 1, 2, 6, 7) |
| `viewer.py` | 3D viewer + multi-perspective screenshot capture |
| `clean/clean_ply.py` | Point cloud cleaning + object extraction (Stage 3) |
| `clean/recons.py` | Poisson recon + watertight repair drivers (Stages 4, 5) |
| `clean/recons_worker.py` | Subprocess worker for Poisson reconstruction |
| `clean/meshfix_worker.py` | Subprocess worker for PyMeshFix + Open3D fill_holes |
| `clean/com_vol.py` | Volume helpers |
| `vggt/` | VGGT model + utilities |
