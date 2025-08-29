import numpy as np
import json
import os
import argparse

def convert_yolo_to_videopose3d_format(yolo_json_path, video_width, video_height, output_dir):
    """
    Converts YOLO pose estimation JSON output to the .npz format required by VideoPose3D.
    Saves a single .npz file containing the first detected person's keypoints for the entire video.
    """
    with open(yolo_json_path, 'r') as f:
        data = json.load(f)

    # VideoPose3D expects a dictionary structure like:
    # {'subject_name': {'action_name': [keypoints_for_camera_0, keypoints_for_camera_1, ...]}}
    # For our custom data, we'll use 'custom' for subject and 'yolo' for action/keypoints type.
    # We'll assume a single camera (index 0).

    all_frames_first_person_keypoints = []

    for frame in data:
        persons_in_frame = frame['persons']
        
        if persons_in_frame: # If at least one person is detected
            # Take the first detected person
            person_data = persons_in_frame[0]
            keypoints_xy = np.array(person_data['keypoints'])[:, :2] # (17, 2)
            keypoints_conf = np.array(person_data['keypoints'])[:, 2:3] # (17, 1)
            
            # Combine x, y, and confidence into (17, 3) array
            kps_array = np.concatenate((keypoints_xy, keypoints_conf), axis=1) # (17, 3)
            all_frames_first_person_keypoints.append(kps_array)
        else:
            # If no person detected, append a zero array of the expected shape (17, 3)
            all_frames_first_person_keypoints.append(np.zeros((17, 3)))

    # Stack all frames' keypoints into a single numpy array
    # Shape will be (num_frames, 17, 3)
    final_keypoints_array = np.stack(all_frames_first_person_keypoints)

    # Create the dictionary structure for VideoPose3D
    positions_2d = {
        'custom': { # Subject name
            'yolo': [ # Action name / Keypoints type
                final_keypoints_array # Keypoints for camera 0
            ]
        }
    }

    metadata = {
        'layout': 'coco', # Assuming YOLO output is COCO-like
        'num_joints': 17,
        'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]], # Example symmetry
        'w': video_width,
        'h': video_height
    }

    # Ensure the output directory for VideoPose3D data exists
    videopose3d_data_dir = "data"
    os.makedirs(videopose3d_data_dir, exist_ok=True)

    output_npz_path = os.path.join(videopose3d_data_dir, "data_2d_custom_yolo.npz")
    
    np.savez_compressed(
        output_npz_path,
        positions_2d=positions_2d,
        metadata=metadata
    )
    print(f"Successfully created VideoPose3D input NPZ at {output_npz_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_json', type=str, required=True, help='Path to the YOLO JSON output file.')
    parser.add_argument('--video_width', type=int, required=True, help='Width of the video.')
    parser.add_argument('--video_height', type=int, required=True, help='Height of the video.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for the output .npz files (not used for final output path, but for consistency).')
    args = parser.parse_args()

    # The output_dir argument is not directly used for the final NPZ path, as it's hardcoded for VideoPose3D.
    # However, it's kept for consistency with other scripts.
    convert_yolo_to_videopose3d_format(args.yolo_json, args.video_width, args.video_height, args.output_dir)
