import bpy
import os

def import_bvh_to_blender(bvh_file_path, global_scale=0.01, frame_start=1):
    """
    Imports a BVH file into Blender.

    Args:
        bvh_file_path (str): The absolute path to the BVH file.
        global_scale (float): Factor to scale the imported armature.
        frame_start (int): The start frame in Blender for the animation.
    """
    if not os.path.exists(bvh_file_path):
        print(f"Error: BVH file not found at {bvh_file_path}")
        return

    try:
        # Ensure the BVH import add-on is enabled
        # This is usually enabled by default, but it's good practice to check.
        # This check is for interactive Blender sessions, not background runs.
        if not bpy.app.background:
            if "import_export_bvh" not in bpy.context.preferences.addons:
                bpy.ops.preferences.addon_enable(module="import_export_bvh")
                print("Enabled 'BioVision Motion Capture (BVH)' add-on.")

        bpy.ops.import_anim.bvh(
            filepath=bvh_file_path,
            filter_glob="*.bvh",
            target='ARMATURE',
            global_scale=global_scale,
            frame_start=frame_start,
            use_fps_scale=False,
            use_cyclic=False,
            rotate_mode='NATIVE'
        )
        print(f"Successfully imported BVH file: {bvh_file_path}")

    except Exception as e:
        print(f"Error importing BVH file: {e}")

# --- Usage Example ---
if __name__ == "__main__":
    # IMPORTANT: Replace this with the ACTUAL ABSOLUTE PATH to your generated BVH file.
    # You can find your BVH files in the 'output_data' directory after running the pipeline.
    # Example: /mnt/d/progress/ani_bender/output_data/boxing_mp4_person0.bvh
    your_bvh_file = "/path/to/your/generated/animation.bvh"

    # Call the import function
    import_bvh_to_blender(
        bvh_file_path=your_bvh_file,
        global_scale=0.01,  # Adjust scale as needed (0.01 is a common starting point)
        frame_start=1       # Start animation at frame 1
    )
