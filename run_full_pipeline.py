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

    os.makedirs(args.output_base_dir, exist_ok=True)

    absolute_video_path = os.path.abspath(args.video_path)
    video_filename_base = os.path.basename(absolute_video_path).replace('.', '_')

    # Step 1: Run Lightweight 3D Human Pose Estimation Demo
    output_3d_json = os.path.join(args.output_base_dir, f'{video_filename_base}_lightweight_3d_keypoints.json')
    absolute_output_3d_json = os.path.abspath(output_3d_json)
    run_command(
        f'demo.py -m human-pose-estimation-3d.pth --video "{absolute_video_path}" --output-json-path "{absolute_output_3d_json}" --no-display',
        "Running Lightweight 3D Human Pose Estimation",
        cwd='models/lightweight-human-pose-estimation-3d-demo/'
    )

    # Step 2: Temporal Smoothing (now uses output from Lightweight demo)
    output_smoothed_3d_json = os.path.join(args.output_base_dir, f'{video_filename_base}_lightweight_smoothed_3d_keypoints.json')
    run_command(
        f"scripts/apply_smoothing.py --input_json_path \"{output_3d_json}\" --output_dir \"{args.output_base_dir}\"",
        "Applying Temporal Smoothing"
    )

    # Step 3: Convert to OpenPose JSON format
    output_openpose_json = os.path.join(args.output_base_dir, f'{video_filename_base}_openpose_3d_keypoints.json')
    run_command(
        f"scripts/convert_lightweight_to_openpose_json.py --input_json_path \"{output_smoothed_3d_json}\" --output_dir \"{args.output_base_dir}\"",
        "Converting to OpenPose JSON format"
    )
    print(f"Pipeline completed. OpenPose JSON saved to {output_openpose_json}")

if __name__ == "__main__":
    main()