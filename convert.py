import trimesh
import os

def convert_glb_to_stl(input_file, output_file):
    try:
        # Load the GLB file
        # GLB files often contain a 'scene' with multiple meshes
        scene = trimesh.load(input_file)
        
        # Check if it's a scene (common for GLB) and concatenate into one mesh
        if isinstance(scene, trimesh.Scene):
            print(f"Combining {len(scene.geometry)} mesh geometries...")
            mesh = trimesh.util.concatenate([
                geom for geom in scene.geometry.values()
            ])
        else:
            mesh = scene

        # Export as STL
        mesh.export(output_file)
        print(f"Successfully converted! Saved to: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Change these filenames to match yours
input_path = "st.glb" 
output_path = "st.stl"

if os.path.exists(input_path):
    convert_glb_to_stl(input_path, output_path)
else:
    print(f"File {input_path} not found. Check the file name!")