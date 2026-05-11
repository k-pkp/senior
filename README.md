# VGGT Pipeline

Self-contained 3D reconstruction pipeline: images → point cloud → cleaned objects → watertight meshes → real-world volumes (ArUco-calibrated).

## Layout

```
new/
├── run.py                       # entry point (executable)
├── viewer.py                    # PLY/STL viewer (executable)
├── requirements.txt
├── README.md
├── vggt/                        # VGGT model package (bundled)
├── pipeline/
│   ├── cli.py                   # argparse
│   ├── config.py                # constants (ArUco index, real size, model URL)
│   ├── orchestrator.py          # main() — runs Stages 1-7
│   ├── stages/
│   │   ├── inference.py         # Stage 1 — VGGT model inference
│   │   ├── pointcloud.py        # Stage 2 — filter + export PLY
│   │   ├── clean.py             # Stage 3 — level / outlier / floor / cluster
│   │   ├── reconstruct.py       # Stage 4 — Poisson reconstruction
│   │   ├── watertight.py        # Stage 5 — PyMeshFix watertight repair
│   │   ├── evaluate.py          # Stage 6 — multi-view screenshots
│   │   └── volume.py            # Stage 7 — ArUco-scaled real-world volumes
│   ├── core/
│   │   ├── plane.py             # RANSAC + leveling primitives
│   │   ├── cluster.py           # DBSCAN + ArUco-aware ranking
│   │   ├── filters.py           # confidence + spatial outlier filters
│   │   └── mesh.py              # merge_meshes, verify_watertight, cleanup
│   └── utils/
│       └── seeding.py           # seed_everything (random / numpy / torch / o3d)
├── workers/
│   ├── recons_worker.py         # Poisson reconstruction subprocess (executable)
│   └── meshfix_worker.py        # PyMeshFix subprocess (executable)
└── tools/
    └── com_vol.py               # standalone mesh-vs-reference volume tool
```

## Install

```bash
# Recommended: virtual env or conda env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
./run.py                                          # uses ./baam/ as input
./run.py --image_folder ./baam/
./run.py --image_folder ./baam/ --output_dir output/
./run.py --image_folder ./baam/ --skip_mesh       # PLY only
./run.py --image_folder ./baam/ --evaluate        # multi-view screenshots
./run.py --image_folder ./vase/ --conf_thres 30
```

Or:
```bash
python run.py --image_folder ./baam/
```

Backends auto-detected: CUDA → MPS (Apple Silicon) → CPU.

## Object identification

After Stage 3, two objects are saved:
- `object_0.ply` = target (unknown volume)
- `object_1.ply` = ArUco reference cube (known: 14×14×14 cm)

Ranking by combined score: `cubeness × 0.6 + bw_ratio × 0.4`. Most cube-like + most black/white → ArUco.

Adjust constants in `pipeline/config.py`:
- `REFERENCE_OBJECT_INDEX` — which output index is the reference (default 1)
- `REFERENCE_REAL_SIZE_CM` — real edge length in cm (default 14.0)

## Outputs

Default `output/`:
```
output/
├── points.ply                   # filtered point cloud
├── predictions.npz              # VGGT raw predictions
├── clean_objects/
│   ├── object_0.ply
│   └── object_1.ply
├── mesh/
│   ├── object_0_recon.ply       # Poisson
│   ├── object_0.ply             # watertight
│   ├── scene_recon.ply
│   └── scene.ply
├── evaluation/                  # multi-view screenshots (--evaluate)
└── target/                      # demo_gradio-compatible layout
```

## View

```bash
./viewer.py output/points.ply
./viewer.py output/mesh/object_0.ply
```
