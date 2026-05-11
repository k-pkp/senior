#!/usr/bin/env python3
"""VGGT Run Script — automated terminal inference → PLY → clean → reconstruct → evaluate.

Usage:
    python run.py                                          # uses ./baam/ as input
    python run.py --image_folder ./baam/
    python run.py --image_folder ./baam/ --output_dir output/
    python run.py --image_folder ./baam/ --skip_mesh        # PLY only, skip clean+reconstruct
    python run.py --image_folder ./baam/ --evaluate          # auto-screenshot with viewer.py

Supports CUDA, MPS (Apple Silicon), and CPU backends automatically.
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from pipeline.orchestrator import main

if __name__ == "__main__":
    main()
