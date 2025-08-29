import os
import subprocess
import argparse
import glob

def run_command(command, description):
    print(f"\n--- {description} ---")
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(process.stdout)
    print(process.stderr)
    if process.returncode != 0:
        print(f"Error during {description}. Exit code: {process.returncode}")
        exit(process.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run the VIBE pipeline for 3D pose and shape estimation to BVH animation.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--output_base_dir', type=str, default="output_data", help="Base directory for all output files.")

    args = parser.parse_args()

    video_filename_base = os.path.basename(args.video_path).replace('.', '_')

    # Step 1: Run VIBE Demo
    vibe_output_base_folder = os.path.join(args.output_base_dir, "vibe_output")
    vibe_output_person_folder = os.path.join(vibe_output_base_folder, os.path.basename(args.video_path).replace('.mp4', ''))
    vibe_pkl_path = os.path.join(vibe_output_person_folder, "vibe_output.pkl")
    
    # Ensure VIBE data is prepared (models, etc.) - this needs to be run once manually or integrated here carefully
    # For now, assuming prepare_data.sh has been run successfully by the user.

    run_command(
        f"venv/bin/python models/VIBE/demo.py --vid_file '{args.video_path}' --output_folder '{vibe_output_base_folder}' --no_render",
        "Running VIBE Demo"
    )

    # Step 2: Process VIBE Output (extract 3D keypoints)
    output_3d_json = os.path.join(args.output_base_dir, f'{video_filename_base}_vibe_3d_keypoints.json')
    run_command(
        f"venv/bin/python scripts/process_vibe_output.py --vibe_pkl_path '{vibe_pkl_path}' --output_dir '{args.output_base_dir}' --video_filename_base '{video_filename_base}'",
        "Processing VIBE Output"
    )

    # Step 3: Temporal Smoothing
    output_smoothed_3d_json = os.path.join(args.output_base_dir, f'{video_filename_base}_vibe_smoothed_3d_keypoints.json')
    run_command(
        f"venv/bin/python scripts/apply_smoothing.py --input_json_path '{output_3d_json}' --output_dir '{args.output_base_dir}'",
        "Applying Temporal Smoothing"
    )

    # Step 4: Convert to BVH (generates multiple BVH files if multiple people detected)
    run_command(
        f"venv/bin/python scripts/convert_to_bvh.py --input_json_path '{output_smoothed_3d_json}' --output_dir '{args.output_base_dir}'",
        "Converting to BVH"
    )

    # Step 5: Process each generated BVH file
    bvh_files = glob.glob(os.path.join(args.output_base_dir, f'{video_filename_base}_person*.bvh'))
    
    if not bvh_files:
        print("No BVH files found for processing.")
        return

    for bvh_file_path in bvh_files:
        person_id = os.path.basename(bvh_file_path).split('_person')[1].replace('.bvh', '')
        print(f"\n--- Processing BVH for Person {person_id} ---")

        # Step 5.1: Parse BVH
        parsed_json_path = os.path.join(args.output_base_dir, f'{video_filename_base}_person{person_id}_parsed_positions.json')
        run_command(
            f"venv/bin/python scripts/parse_bvh.py --bvh_path '{bvh_file_path}' --output_dir '{args.output_base_dir}'",
            f"Parsing BVH for Person {person_id}"
        )

        # Step 5.2: Visualize BVH frames
        output_frames_dir = os.path.join(args.output_base_dir, f'frames_vibe_person{person_id}_3d')
        run_command(
            f"venv/bin/python scripts/visualize_bvh.py --parsed_json_path '{parsed_json_path}' --output_frames_dir '{output_frames_dir}'",
            f"Generating Visualization Frames for Person {person_id}"
        )

        # Step 5.3: Create Video from Frames
        output_video_path = os.path.join(args.output_base_dir, f'bvh_animation_vibe_person{person_id}_3d.mp4')
        run_command(
            f"venv/bin/python scripts/create_video_from_frames.py --input_frames_dir '{output_frames_dir}' --output_video_path '{output_video_path}'",
            f"Creating Video for Person {person_id}"
        )
        print(f"Full pipeline completed for Person {person_id}. Video saved to {output_video_path}")

if __name__ == "__main__":
    main()
