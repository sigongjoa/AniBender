
import cv2
import numpy as np
import argparse
import os
import sys

# Add the model's root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'lightweight-human-pose-estimation-3d-demo')))
# Add the pose_extractor/build directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'lightweight-human-pose-estimation-3d-demo', 'pose_extractor', 'build')))

from modules.input_reader import VideoReader, ImageReader
from modules.inference_engine_pytorch import InferenceEnginePyTorch
from modules.parse_poses import parse_poses
from modules.pose import Pose # For Pose.num_kpts and kpt_names

# Define the skeleton edges for 19 joints (from modules/draw.py)
SKELETON_EDGES = np.array([[11, 10], [10, 9], [9, 0], [0, 3], [3, 4], [4, 5], [0, 6], [6, 7], [7, 8], [0, 12],
                           [12, 13], [13, 14], [0, 1], [1, 15], [15, 16], [1, 17], [17, 18]])

def run_lightweight_pose_estimation(video_path, model_path, output_npz_path):
    # Default arguments from demo.py, some are hardcoded for simplicity
    args_device = 'GPU'
    args_use_tensorrt = False
    args_height_size = 256
    args_fx = -1

    stride = 8
    net = InferenceEnginePyTorch(model_path, args_device, use_tensorrt=args_use_tensorrt)

    frame_provider = VideoReader(video_path)
    is_video = True
    base_height = args_height_size
    fx = args_fx

    all_frames_poses = []
    previous_poses_2d = [] # For tracking

    frame_idx = 0
    for frame in frame_provider:
        if frame is None:
            break

        input_scale = base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % stride)]
        if fx < 0:  # Focal length is unknown
            fx = np.float32(0.8 * frame.shape[1])

        inference_result = net.infer(scaled_img)
        # parse_poses returns translated_poses_3d and poses_2d_scaled
        # poses_2d_scaled contains Pose objects with IDs after propagate_ids
        translated_poses_3d, poses_2d_scaled = parse_poses(inference_result, input_scale, stride, fx, is_video=True)

        current_frame_data = []
        for pose_idx, pose_obj in enumerate(poses_2d_scaled):
            person_id = pose_obj.id
            # Find the corresponding 3D pose. translated_poses_3d is already in the correct format (N, 19, 3)
            # after the reshape in demo.py
            # The order of poses in translated_poses_3d corresponds to poses_2d_scaled
            pose_3d_data = translated_poses_3d[pose_idx].reshape((19, 3))
            current_frame_data.append({'person_id': person_id, 'pose_3d': pose_3d_data.tolist()})
        
        all_frames_poses.append({'frame_idx': frame_idx, 'persons': current_frame_data})
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

    # Save all_frames_poses to a single NPZ file
    np.savez_compressed(output_npz_path, data=all_frames_poses)
    print(f"Successfully saved 3D poses to {output_npz_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Lightweight 3D Pose Estimation and save results.")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--model', type=str, required=True, help='Path to the pre-trained model checkpoint.')
    parser.add_argument('--output', type=str, required=True, help='Path for the output .npz file.')
    args = parser.parse_args()

    run_lightweight_pose_estimation(args.video, args.model, args.output)
