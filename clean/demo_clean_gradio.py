import os
import shutil
from pathlib import Path

import gradio as gr

from clean_ply import clean_point_cloud
from recons import reconstruct_mesh
from com_vol import compute_volume_from_mesh


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
CLEAN_INPUT_DIR = BASE_DIR / "clean_input_ply"
OUTPUT_DIR = BASE_DIR / "output_ply"


def ensure_dirs():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline(uploaded_ply_path: str, cube_size_cm: float):
    """Run full pipeline:
    1) Save uploaded ply as input/input.ply
    2) Clean and save to clean_input_ply
    3) Reconstruct and save to output_ply
    4) Compute volume from STL and show in UI
    """
    if not uploaded_ply_path:
        return None, None, None, "Please upload a PLY file first."

    ensure_dirs()

    src_path = Path(uploaded_ply_path)
    if src_path.suffix.lower() != ".ply":
        return None, None, None, "Uploaded file must be .ply"

    try:
        # Step 2: Save as input/input.ply
        saved_input = INPUT_DIR / "input.ply"
        shutil.copyfile(src_path, saved_input)

        # Step 3: Clean point cloud -> clean_input_ply/clean_input.ply
        cleaned_path = clean_point_cloud(
            input_path=str(saved_input),
            output_folder=str(CLEAN_INPUT_DIR),
            output_name="clean_input.ply",
        )

        # Step 4: Reconstruction -> output_ply/mesh_input.ply + output_ply/mesh_input.stl
        _, mesh_stl_path = reconstruct_mesh(
            input_path=cleaned_path,
            output_folder=str(OUTPUT_DIR),
            base_name="input",
        )

        # Step 5: Volume from STL + cleaned reference point cloud
        result = compute_volume_from_mesh(
            mesh_path=mesh_stl_path,
            reference_ply_path=cleaned_path,
            real_cube_size_m=float(cube_size_cm) / 100.0,
        )

        report = (
            "Pipeline completed successfully\n\n"
            f"Saved input: {saved_input}\n"
            f"Cleaned point cloud: {cleaned_path}\n"
            f"Reconstructed STL: {mesh_stl_path}\n\n"
            f"Real volume: {result['real_volume_m3']:.6f} m^3\n"
            f"Real volume: {result['real_volume_cm3']:.2f} cm^3\n"
            f"Scale factor: {result['scale_factor']:.6f}\n"
            f"Watertight: {result['is_watertight']}"
        )

        return str(saved_input), str(cleaned_path), str(mesh_stl_path), report

    except Exception as exc:
        return None, None, None, f"Pipeline failed: {exc}"


with gr.Blocks(title="PLY Clean and Volume Pipeline") as demo:
    gr.Markdown("## Automatic PLY Pipeline")
    gr.Markdown(
        "Upload a .ply file. The app will save it as input/input.ply, clean it, reconstruct STL, and compute volume."
    )

    with gr.Row():
        uploaded_ply = gr.File(label="Upload Input PLY", file_types=[".ply"], type="filepath")
        cube_size_cm = gr.Number(label="Real Cube Size (cm)", value=14.0, precision=3)

    run_btn = gr.Button("Run Full Pipeline", variant="primary")

    with gr.Row():
        saved_input_out = gr.File(label="Step 2 Output: input/input.ply")
        cleaned_out = gr.File(label="Step 3 Output: clean_input_ply/clean_input.ply")
        stl_out = gr.File(label="Step 4 Output: output_ply/mesh_input.stl")

    result_text = gr.Textbox(label="Step 5 Result", lines=10)

    run_btn.click(
        fn=run_pipeline,
        inputs=[uploaded_ply, cube_size_cm],
        outputs=[saved_input_out, cleaned_out, stl_out, result_text],
    )


if __name__ == "__main__":
    demo.launch()
