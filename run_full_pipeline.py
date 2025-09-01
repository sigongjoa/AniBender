import os
import subprocess
import argparse

def run_command(command, description, cwd=None):
    print(f"\n--- {description} ---")
    # Always use the project's main venv python for all scripts to ensure consistency
    full_command = f"/mnt/d/progress/ani_bender/venv/bin/python {command}"

    process = subprocess.run(full_command, shell=True, capture_output=True, text=True, cwd=cwd)
    print(process.stdout)
    if process.stderr:
        print("--- Stderr ---")
        print(process.stderr)
    if process.returncode != 0:
        print(f"Error during: {description}. Exit code: {process.returncode}")
        exit(process.returncode) # Exit on error to prevent cascading failures

def main():
    parser = argparse.ArgumentParser(description="Run the full pose estimation and visualization pipeline.")
    parser.add_argument('--video', type=str, required=True, help="Path to the input video file (e.g., input_videos/your_video.mp4).")
    parser.add_argument('--output_dir', type=str, default="output_data", help="Base directory for all output files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_filename_base = os.path.splitext(os.path.basename(args.video))[0]
    absolute_video_path = os.path.abspath(args.video)
    absolute_output_dir = os.path.abspath(args.output_dir)

    # --- Step 1: Run Lightweight 3D Human Pose Estimation to get 2D/3D keypoints ---
    keypoints_json_path = os.path.join(absolute_output_dir, f'{video_filename_base}_keypoints.json')
    run_command(
        f'demo.py -m human-pose-estimation-3d.pth --video "{absolute_video_path}" --output-json-path "{keypoints_json_path}" --no-display',
        "Running Lightweight 3D Human Pose Estimation",
        cwd='models/lightweight-human-pose-estimation-3d-demo/'
    )

    # --- Step 2: Run Visualization Script ---
    if os.path.exists(keypoints_json_path):
        run_command(
            f'scripts/visualize_data.py --video "{absolute_video_path}" --json "{keypoints_json_path}" --output_dir "{absolute_output_dir}"',
            "Creating 2D/3D Overlay Videos and Logs"
        )
    else:
        print(f"Error: Keypoint JSON file not found at {keypoints_json_path}. Skipping visualization.")

    print("\n--- Pipeline Finished ---")
    print(f"Check the '{args.output_dir}' directory for results:")
    print(f"- {os.path.basename(keypoints_json_path)}")
    print(f"- 2d_keypoints.txt")
    print(f"- 3d_keypoints.txt")
    print(f"- video_2d_overlay.mp4")
    print(f"- video_3d_overlay.mp4")

if __name__ == "__main__":
    main()