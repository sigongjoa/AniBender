import os
import subprocess
import argparse
import glob

def run_command(command, description, cwd=None):
    print(f"\n--- {description} ---")
    # Use venv/bin/python for robustness on Linux
    full_command = f"/mnt/d/progress/ani_bender/venv/bin/python {command}"
    process = subprocess.run(full_command, shell=True, capture_output=True, text=True, cwd=cwd)
    print(process.stdout)
    print(process.stderr)
    if process.returncode != 0:
        print(f"Error during {description}. Exit code: {process.returncode}")
        exit(process.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run the full pose estimation to BVH animation pipeline.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file (e.g., input_videos/your_video.mp4).")
    parser.add_argument('--output_base_dir', type=str, default="output_data", help="Base directory for all output files.")

    args = parser.parse_args()

    absolute_video_path = os.path.abspath(args.video_path)
    video_filename_base = os.path.basename(absolute_video_path).replace('.', '_')

    # Step 1: Run Lightweight 3D Human Pose Estimation Demo
    output_3d_json = os.path.join(args.output_base_dir, f'{video_filename_base}_lightweight_3d_keypoints.json')
    run_command(
        f'demo.py -m human-pose-estimation-3d.pth --video "{absolute_video_path}" --output-json-path "{output_3d_json}"',
        "Running Lightweight 3D Human Pose Estimation",
        cwd='models/lightweight-human-pose-estimation-3d-demo/'
    )

    # Step 2: Temporal Smoothing (now uses output from Lightweight demo)
    output_smoothed_3d_json = os.path.join(args.output_base_dir, f'{video_filename_base}_lightweight_smoothed_3d_keypoints.json')
    run_command(
        f"scripts/apply_smoothing.py --input_json_path \"{output_3d_json}\" --output_dir \"{args.output_base_dir}\"",
        "Applying Temporal Smoothing"
    )

    # Step 3: Convert to BVH (generates multiple BVH files if multiple people detected)
    run_command(
        f"scripts/convert_to_bvh.py --input_json_path \"{output_smoothed_3d_json}\" --output_dir \"{args.output_base_dir}\"",
        "Converting to BVH"
    )

    # Step 4: Process each generated BVH file
    # Find all BVH files generated for this video (e.g., video_base_person1.bvh, video_base_person2.bvh)
    bvh_files = glob.glob(os.path.join(args.output_base_dir, f'{video_filename_base}_person*.bvh'))
    
    if not bvh_files:
        print("No BVH files found for processing.")
        return

    for bvh_file_path in bvh_files:
        person_id = os.path.basename(bvh_file_path).split('_person')[1].replace('.bvh', '')
        print(f"\n--- Processing BVH for Person {person_id} ---")

        # Step 4.1: Parse BVH
        parsed_json_path = os.path.join(args.output_base_dir, f'{video_filename_base}_lightweight_person{person_id}_parsed_positions.json')
        run_command(
            f"scripts/parse_bvh.py --bvh_path \"{bvh_file_path}\" --output_dir \"{args.output_base_dir}\"",
            f"Parsing BVH for Person {person_id}"
        )

        # Step 4.2: Visualize BVH frames
        output_frames_dir = os.path.join(args.output_base_dir, f'frames_lightweight_person{person_id}_3d')
        run_command(
            f"scripts/visualize_bvh.py --parsed_json_path \"{parsed_json_path}\" --output_frames_dir \"{output_frames_dir}\"",
            f"Generating Visualization Frames for Person {person_id}"
        )

        # Step 4.3: Create Video from Frames
        output_video_path = os.path.join(args.output_base_dir, f'bvh_animation_lightweight_person{person_id}_3d.mp4')
        run_command(
            f"scripts/create_video_from_frames.py --input_frames_dir \"{output_frames_dir}\" --output_video_path \"{output_video_path}\"",
            f"Creating Video for Person {person_id}"
        )
        print(f"Full pipeline completed for Person {person_id}. Video saved to {output_video_path}")

if __name__ == "__main__":
    main()