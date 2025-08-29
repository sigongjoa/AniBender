import os
import subprocess
import argparse
import glob

def run_command(command, description):
    print(f"\n--- {description} ---")
    # Use venv\Scripts\python.exe for robustness on Windows
    full_command = f"venv\\Scripts\\python.exe {command}"
    process = subprocess.run(full_command, shell=True, capture_output=True, text=True)
    print(process.stdout)
    print(process.stderr)
    if process.returncode != 0:
        print(f"Error during {description}. Exit code: {process.returncode}")
        exit(process.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run the full pose estimation to BVH animation pipeline.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file (e.g., input_videos\\your_video.mp4).")
    parser.add_argument('--output_base_dir', type=str, default="output_data", help="Base directory for all output files.")

    args = parser.parse_args()

    video_filename_base = os.path.basename(args.video_path).replace('.', '_')

    # Step 1: Run MediaPipe Pose Estimation
    output_mediapipe_3d_json = os.path.join(args.output_base_dir, f'{video_filename_base}_mediapipe_3d_keypoints.json')
    run_command(
        f"scripts\\run_pose_estimation_mediapipe.py --video_path \"{args.video_path}\" --output_dir \"{args.output_base_dir}\" --output_annotated_frames_dir \"{args.output_base_dir}\\annotated_frames_mediapipe\"",
        "Running MediaPipe Pose Estimation"
    )

    # Step 2: Uplift to 3D (Pass-through for MediaPipe 3D output)
    # MediaPipe already outputs 3D, so this is a pass-through step.
    # The output filename is the same as input for this step, but it ensures consistency.
    run_command(
        f"scripts\\uplift_to_3d.py --input_json_path \"{output_mediapipe_3d_json}\" --output_dir \"{args.output_base_dir}\"",
        "Passing through MediaPipe 3D data"
    )

    # Step 3: Temporal Smoothing
    output_smoothed_3d_json = os.path.join(args.output_base_dir, f'{video_filename_base}_mediapipe_smoothed_3d_keypoints.json')
    run_command(
        f"scripts\\apply_smoothing.py --input_json_path \"{output_mediapipe_3d_json}\" --output_dir \"{args.output_base_dir}\"",
        "Applying Temporal Smoothing"
    )

    # Step 4: Convert to BVH (generates multiple BVH files if multiple people detected)
    run_command(
        f"scripts\\convert_to_bvh.py --input_json_path \"{output_smoothed_3d_json}\" --output_dir \"{args.output_base_dir}\"",
        "Converting to BVH"
    )

    # Step 5: Process each generated BVH file
    # Find all BVH files generated for this video (e.g., video_base_person1.bvh, video_base_person2.bvh)
    bvh_files = glob.glob(os.path.join(args.output_base_dir, f'{video_filename_base}_person*.bvh'))
    
    if not bvh_files:
        print("No BVH files found for processing.")
        return

    for bvh_file_path in bvh_files:
        person_id = os.path.basename(bvh_file_path).split('_person')[1].replace('.bvh', '')
        print(f"\n--- Processing BVH for Person {person_id} ---")

        # Step 5.1: Parse BVH
        parsed_json_path = os.path.join(args.output_base_dir, f'{video_filename_base}_mediapipe_person{person_id}_parsed_positions.json')
        run_command(
            f"scripts\\parse_bvh.py --bvh_path \"{bvh_file_path}\" --output_dir \"{args.output_base_dir}\"",
            f"Parsing BVH for Person {person_id}"
        )

        # Step 5.2: Visualize BVH frames
        output_frames_dir = os.path.join(args.output_base_dir, f'frames_mediapipe_person{person_id}_3d')
        run_command(
            f"scripts\\visualize_bvh.py --parsed_json_path \"{parsed_json_path}\" --output_frames_dir \"{output_frames_dir}\"",
            f"Generating Visualization Frames for Person {person_id}"
        )

        # Step 5.3: Create Video from Frames
        output_video_path = os.path.join(args.output_base_dir, f'bvh_animation_mediapipe_person{person_id}_3d.mp4')
        run_command(
            f"scripts\\create_video_from_frames.py --input_frames_dir \"{output_frames_dir}\" --output_video_path \"{output_video_path}\"",
            f"Creating Video for Person {person_id}"
        )
        print(f"Full pipeline completed for Person {person_id}. Video saved to {output_video_path}")

if __name__ == "__main__":
    main()
