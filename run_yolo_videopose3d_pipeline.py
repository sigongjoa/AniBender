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
    parser = argparse.ArgumentParser(description="Run the YOLO-VideoPose3D pipeline for 3D pose estimation to BVH animation.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--output_base_dir', type=str, default="output_data", help="Base directory for all output files.")

    args = parser.parse_args()

    video_filename_base = os.path.basename(args.video_path).replace('.', '_')

    # Step 1: Run YOLO 2D Pose Estimation
    yolo_2d_json_path = os.path.join(args.output_base_dir, f'{video_filename_base}_2d_keypoints.json')
    run_command(
        f"venv/bin/python scripts/run_pose_estimation.py --video_path '{args.video_path}' --output_dir '{args.output_base_dir}'",
        "Running YOLO 2D Pose Estimation"
    )

    # Step 2: Prepare YOLO output for VideoPose3D
    # Need video width and height for prepare_yolo_for_videopose3d.py
    # For now, hardcode or get from video metadata if possible. Let's assume 1920x1080 for now.
    # A more robust solution would be to read video metadata.
    video_width = 1920
    video_height = 1080
    run_command(
        f"venv/bin/python scripts/prepare_yolo_for_videopose3d.py --yolo_json '{yolo_2d_json_path}' --video_width {video_width} --video_height {video_height} --output_dir '{args.output_base_dir}'",
        "Preparing YOLO output for VideoPose3D"
    )

    # The NPZ file is now saved directly to data/data_2d_custom_yolo.npz by prepare_yolo_for_videopose3d.py
    videopose3d_input_npz_path = "data/data_2d_custom_yolo.npz"

    # Step 3: Uplift to 3D using VideoPose3D (modified uplift_to_3d.py will handle this)
    output_3d_json = os.path.join(args.output_base_dir, f'{video_filename_base}_videopose3d_3d_keypoints.json')
    run_command(
        f"venv/bin/python scripts/uplift_to_3d.py --input_npz_path '{videopose3d_input_npz_path}' --output_dir '{args.output_base_dir}' --video_filename_base '{video_filename_base}'",
        "Uplifting to 3D using VideoPose3D"
    )

    # Step 4: Temporal Smoothing
    output_smoothed_3d_json = os.path.join(args.output_base_dir, f'{video_filename_base}_videopose3d_smoothed_3d_keypoints.json')
    run_command(
        f"venv/bin/python scripts/apply_smoothing.py --input_json_path '{output_3d_json}' --output_dir '{args.output_base_dir}'",
        "Applying Temporal Smoothing"
    )

    # Step 5: Convert to BVH (generates multiple BVH files if multiple people detected)
    run_command(
        f"venv/bin/python scripts/convert_to_bvh.py --input_json_path '{output_smoothed_3d_json}' --output_dir '{args.output_base_dir}'",
        "Converting to BVH"
    )

    # Step 6: Process each generated BVH file
    bvh_files = glob.glob(os.path.join(args.output_base_dir, f'{video_filename_base}_person*.bvh'))
    
    if not bvh_files:
        print("No BVH files found for processing.")
        return

    for bvh_file_path in bvh_files:
        person_id = os.path.basename(bvh_file_path).split('_person')[1].replace('.bvh', '')
        print(f"\n--- Processing BVH for Person {person_id} ---")

        # Step 6.1: Parse BVH
        parsed_json_path = os.path.join(args.output_base_dir, f'{video_filename_base}_person{person_id}_parsed_positions.json')
        run_command(
            f"venv/bin/python scripts/parse_bvh.py --bvh_path '{bvh_file_path}' --output_dir '{args.output_base_dir}'",
            f"Parsing BVH for Person {person_id}"
        )

        # Step 6.2: Visualize BVH frames
        output_frames_dir = os.path.join(args.output_base_dir, f'frames_videopose3d_person{person_id}_3d')
        run_command(
            f"venv/bin/python scripts/visualize_bvh.py --parsed_json_path '{parsed_json_path}' --output_frames_dir '{output_frames_dir}'",
            f"Generating Visualization Frames for Person {person_id}"
        )

        # Step 6.3: Create Video from Frames
        output_video_path = os.path.join(args.output_base_dir, f'bvh_animation_videopose3d_person{person_id}_3d.mp4')
        run_command(
            f"venv/bin/python scripts/create_video_from_frames.py --input_frames_dir '{output_frames_dir}' --output_video_path '{output_video_path}'",
            f"Creating Video for Person {person_id}"
        )
        print(f"Full pipeline completed for Person {person_id}. Video saved to {output_video_path}")

if __name__ == "__main__":
    main()
