import json
import os
import numpy as np
import argparse
import subprocess
import cv2 # To get video dimensions

def uplift_to_3d(input_npz_path, output_dir, video_filename_base):
    """
    Uplifts 2D keypoints from an NPZ file to 3D using VideoPose3D.
    """
    # Define paths
    videopose3d_inference_script = "models/VideoPose3D/inference_script.py"
    checkpoint_path = "models/VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin"
    
    # Output 3D NPY from VideoPose3D inference
    output_3d_npy_path = os.path.join(output_dir, f'{video_filename_base}_videopose3d_3d_keypoints.npy')

    # Get video dimensions from the original video path (assuming it's passed or derivable)
    # The video_width and video_height are passed to prepare_yolo_for_videopose3d.py
    # and stored in the metadata of the NPZ file. Let's extract them from there.
    keypoints_npz = np.load(input_npz_path, allow_pickle=True)
    video_width = keypoints_npz['metadata'].item()['w']
    video_height = keypoints_npz['metadata'].item()['h']

    # Construct the VideoPose3D inference command
    command = [
        "venv/bin/python", videopose3d_inference_script,
        "--input_npz_path", input_npz_path,
        "--checkpoint_path", checkpoint_path,
        "--output_npy_path", output_3d_npy_path,
        "--video_width", str(video_width),
        "--video_height", str(video_height)
    ]

    print(f"Running VideoPose3D inference: {' '.join(command)}")
    process = subprocess.run(command, capture_output=True, text=True) # Removed check=True for debugging
    print(process.stdout)
    print(process.stderr)

    # Read the generated .npy file and convert to JSON
    if not os.path.exists(output_3d_npy_path):
        raise FileNotFoundError(f"VideoPose3D did not generate the expected output file: {output_3d_npy_path}")

    # The .npy file contains 3D keypoints in a numpy array format.
    # We need to convert this to a JSON format similar to MediaPipe's output for consistency.
    # VideoPose3D output shape is typically (frames, num_joints, 3)
    # We need to create a list of dictionaries, where each dict represents a frame
    # and contains a list of persons, each with their keypoints.
    # For simplicity, assuming single person for now.

    # Load the 3D keypoints from the .npy file
    keypoints_3d_np = np.load(output_3d_npy_path)

    # Convert to JSON format
    # Assuming keypoints_3d_np has shape (num_frames, num_joints, 3)
    # We need to create a structure like:
    # [
    #   { "frame_idx": 0, "keypoints": [ [ [x,y,z,visibility], ... ] ] },
    #   { "frame_idx": 1, "keypoints": [ [ [x,y,z,visibility], ... ] ] },
    #   ...
    # ]
    # Visibility is not directly available from VideoPose3D output, so we can set it to 1.0 (visible)

    json_output_data = []
    for frame_idx, frame_keypoints in enumerate(keypoints_3d_np):
        person_keypoints_list = []
        # Assuming single person output from VideoPose3D for now
        # Each keypoint is [x, y, z]
        person_keypoints_with_visibility = [[float(kp[0]), float(kp[1]), float(kp[2]), 1.0] for kp in frame_keypoints]
        person_keypoints_list.append(person_keypoints_with_visibility)
        
        json_output_data.append({
            "frame_idx": frame_idx,
            "keypoints": person_keypoints_list
        })

    output_json_path = os.path.join(output_dir, f'{video_filename_base}_videopose3d_3d_keypoints.json')
    with open(output_json_path, 'w') as f:
        json.dump(json_output_data, f, indent=4)
    
    print(f"3D keypoints uplifted by VideoPose3D and saved to {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uplift 2D pose keypoints to 3D using VideoPose3D.")
    parser.add_argument('--input_npz_path', type=str, required=True, help="Path to the input 2D keypoints NPZ file for VideoPose3D.")
    parser.add_argument('--output_dir', type=str, default="/mnt/d/progress/ani_bender/output_data", help="Directory to save the output 3D keypoints JSON file.")
    parser.add_argument('--video_filename_base', type=str, required=True, help="Base filename of the video for consistent output naming.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    uplift_to_3d(args.input_npz_path, args.output_dir, args.video_filename_base)