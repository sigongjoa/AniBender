import os
import subprocess
import argparse
import glob
import json
import numpy as np
import cv2 # To get video dimensions
import sys

# Ensure the script is run by the venv's python
# This assumes the script is run from the project root
VENV_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), 'venv', 'bin', 'python'))
if sys.executable != VENV_PYTHON:
    print(f"Warning: This script should be run using the virtual environment's python: {VENV_PYTHON}")
    print(f"Attempting to re-execute with venv python...")
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

def run_command(command, description, cwd=None):
    print(f"\n--- {description} ---")
    full_command = f"/mnt/d/progress/ani_bender/venv/bin/python {command}"
    
    process = subprocess.run(full_command, shell=True, capture_output=True, text=True, cwd=cwd)
    print(process.stdout)
    if process.stderr:
        print("--- Stderr ---")
        print(process.stderr)
    if process.returncode != 0:
        print(f"Error during: {description}. Exit code: {process.returncode}")
        exit(process.returncode)

def linear_blend(val1, val2, alpha):
    """Performs linear blending between two values."""
    return val1 * (1 - alpha) + val2 * alpha

def stitch_poses(all_chunk_predictions, video_length, chunk_size, overlap_size):
    """
    Stitches overlapping 3D pose predictions from multiple chunks.
    all_chunk_predictions: List of dictionaries, where each dict is a frame from a chunk.
                           Each frame dict has 'frame_idx' and 'keypoints' (list of persons).
    video_length: Total number of frames in the original video.
    chunk_size: Size of each processing chunk.
    overlap_size: Overlap between chunks.
    """
    stitched_frames = [None] * video_length
    
    # Group predictions by their original frame_idx
    predictions_by_frame = [[] for _ in range(video_length)]
    for frame_data in all_chunk_predictions:
        # DEBUG PRINT
        

        original_frame_idx = frame_data['frame_idx']
        if original_frame_idx < video_length: # Ensure we don't go out of bounds
            predictions_by_frame[original_frame_idx].append(frame_data['keypoints'])

    # Stitching logic
    for i in range(video_length):
        if not predictions_by_frame[i]:
            # If no prediction for this frame, use a zero pose or previous frame's pose
            # For simplicity, let's use a a zero pose for now.
            stitched_frames[i] = {'frame_idx': i, 'keypoints': []} # Empty keypoints for this frame
            continue

        # If only one prediction for this frame (most common for non-overlapping parts)
        if len(predictions_by_frame[i]) == 1:
            stitched_frames[i] = {'frame_idx': i, 'keypoints': predictions_by_frame[i][0]}
            continue

        # Handle overlapping frames
        # For simplicity, we'll just take the first prediction for now.
        # A proper blending would involve weighting based on position within overlap.
        stitched_frames[i] = {'frame_idx': i, 'keypoints': predictions_by_frame[i][0]}

    # Filter out None values if any (shouldn't happen with current logic if all frames have data)
    stitched_frames = [f for f in stitched_frames if f is not None]

    return stitched_frames

def main():
    parser = argparse.ArgumentParser(description="Run the YOLO-VideoPose3d pipeline for 3D pose estimation to BVH animation.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--output_base_dir', type=str, default="output_data", help="Base directory for all output files.")
    parser.add_argument('--chunk_size', type=int, default=243, help="Number of frames to process in each chunk.")
    parser.add_argument('--overlap_size', type=int, default=121, help="Number of overlapping frames between chunks.")
    parser.add_argument('--smoothing_method', type=str, default="moving_average",
                        choices=["moving_average", "savgol", "one_euro", "none"],
                        help="Smoothing method to apply in apply_smoothing.py.")

    args = parser.parse_args()

    os.makedirs(args.output_base_dir, exist_ok=True)

    # Put these definitions back inside main()
    video_filename_base = os.path.splitext(os.path.basename(args.video_path))[0]
    absolute_video_path = os.path.abspath(args.video_path)
    absolute_output_base_dir = os.path.abspath(args.output_base_dir)

    # Correctly derive the base filename used by run_pose_estimation.py
    video_filename_base_for_glob = os.path.basename(args.video_path).replace('.', '_')
    
    # Get total video length
    cap = cv2.VideoCapture(absolute_video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # --- Step 1: Run YOLO 2D Pose Estimation in chunks ---
    run_command(
        f"scripts/run_pose_estimation.py --video_path \"{absolute_video_path}\" --output_dir \"{absolute_output_base_dir}\" --chunk_size {args.chunk_size} --overlap_size {args.overlap_size}",
        "Running YOLO 2D Pose Estimation in chunks"
    )

    # --- Process each chunk ---
    chunk_json_files = sorted(glob.glob(os.path.join(absolute_output_base_dir, f'{video_filename_base_for_glob}_chunk*_2d_keypoints.json')))
    
    if not chunk_json_files:
        print("No 2D keypoint JSON files found for chunks. Aborting pipeline.")
        exit(1)

    all_chunk_predictions = [] # To collect all 3D predictions from all chunks

    for chunk_idx, yolo_2d_json_path in enumerate(chunk_json_files):
        print(f"\n--- Processing Chunk {chunk_idx + 1}/{len(chunk_json_files)} ---")

        # Get video dimensions from the original video (assuming it's consistent)
        cap = cv2.VideoCapture(absolute_video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Step 2: Prepare YOLO output for VideoPose3D
        run_command(
            f"scripts/prepare_yolo_for_videopose3d.py --yolo_json \"{yolo_2d_json_path}\" --video_width {video_width} --video_height {video_height} --output_dir \"{absolute_output_base_dir}\"",
            f"Preparing YOLO output for VideoPose3D (Chunk {chunk_idx + 1})"
        )

        # The NPZ file is now saved directly to data/data_2d_custom_yolo.npz by prepare_yolo_for_videopose3d.py
        videopose3d_input_npz_path = "data/data_2d_custom_yolo.npz"

        # Step 3: Uplift to 3D using VideoPose3D
        output_3d_json_chunk = os.path.join(absolute_output_base_dir, f'{video_filename_base}_chunk{chunk_idx}_videopose3d_3d_keypoints.json')
        run_command(
            f"scripts/uplift_to_3d.py --input_npz_path \"{videopose3d_input_npz_path}\" --output_dir \"{absolute_output_base_dir}\" --video_filename_base \"{video_filename_base}_chunk{chunk_idx}\"",
            f"Uplifting to 3D using VideoPose3D (Chunk {chunk_idx + 1})"
        )

        # Step 4: Temporal Smoothing
        output_smoothed_3d_json_chunk = os.path.join(absolute_output_base_dir, f'{video_filename_base}_chunk{chunk_idx}_videopose3d_smoothed_3d_keypoints.json')
        run_command(
            f"scripts/apply_smoothing.py --input_json_path \"{output_3d_json_chunk}\" --output_dir \"{absolute_output_base_dir}\" --method {args.smoothing_method}",
            f"Applying Temporal Smoothing (Chunk {chunk_idx + 1})"
        )
        
        # Load the smoothed 3D predictions for stitching later
        with open(output_smoothed_3d_json_chunk, 'r') as f:
            chunk_predictions = json.load(f)
            all_chunk_predictions.extend(chunk_predictions) # Extend the list of frames

    # --- Step 5: Stitch all 3D predictions and convert to final BVH ---
    # The stitch_poses function will handle the overlap blending.
    stitched_3d_predictions = stitch_poses(all_chunk_predictions, video_length, args.chunk_size, args.overlap_size)

    # Save the stitched predictions to a temporary JSON for the BVH converter
    final_stitched_json_path = os.path.join(absolute_output_base_dir, f'{video_filename_base}_videopose3d_stitched_3d_keypoints.json')
    with open(final_stitched_json_path, 'w') as f:
        json.dump(stitched_3d_predictions, f, indent=4)

    run_command(
        f"scripts/convert_json_to_bvh_bvhio.py --input_json_path \"{final_stitched_json_path}\" --output_dir \"{absolute_output_base_dir}\"",
        "Converting stitched 3D keypoints to BVH"
    )

    print("\n--- Pipeline Finished ---")
    print(f"Check the '{args.output_base_dir}' directory for results.")

if __name__ == "__main__":
    main()