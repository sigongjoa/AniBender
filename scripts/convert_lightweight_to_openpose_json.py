import json
import os
import numpy as np
import argparse

# Lightweight 19 keypoints (from documentation)
# 0: nose, 1: neck, 2: right_shoulder, 3: right_elbow, 4: right_wrist,
# 5: left_shoulder, 6: left_elbow, 7: left_wrist, 8: right_hip, 9: right_knee,
# 10: right_ankle, 11: left_hip, 12: left_knee, 13: left_ankle, 14: right_eye,
# 15: left_eye, 16: right_ear, 17: left_ear, 18: background (ignored)

# OpenPose BODY_25 Keypoint Ordering:
# 0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
# 5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
# 10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
# 15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
# 20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel"

# Mapping from Lightweight 19 keypoints to OpenPose BODY_25 keypoints
# Value is Lightweight index, None if not directly available
# For missing keypoints, we'll use [0,0,0,0] (x,y,z,confidence)
LIGHTWEIGHT_TO_OPENPOSE_MAP = {
    0: 0,  # Nose
    1: 1,  # Neck
    2: 2,  # RShoulder
    3: 3,  # RElbow
    4: 4,  # RWrist
    5: 5,  # LShoulder
    6: 6,  # LElbow
    7: 7,  # LWrist
    8: 9,  # RHip (OpenPose RHip is 9, Lightweight RHip is 8)
    9: 10, # RKnee
    10: 11, # RAnkle
    11: 12, # LHip (OpenPose LHip is 12, Lightweight LHip is 11)
    12: 13, # LKnee
    13: 14, # LAnkle
    14: 15, # REye
    15: 16, # LEye
    16: 17, # REar
    17: 18, # LEar
    # OpenPose MidHip (8) is not directly in Lightweight, can be interpolated from hips
    # OpenPose foot keypoints (19-24) are not in Lightweight
}

# OpenPose BODY_25 keypoint names for reference
OPENPOSE_BODY_25_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe",
    "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
]

def convert_lightweight_to_openpose_json(input_json_path, output_dir):
    with open(input_json_path, 'r') as f:
        lightweight_data = json.load(f)

    openpose_output = {
        "version": 1.1,
        "people": []
    }

    for frame_data in lightweight_data:
        # Assuming single person for now, or taking the first person if multiple
        if not frame_data["keypoints"] or not frame_data["keypoints"][0]:
            # If no keypoints for this frame/person, append empty data
            openpose_output["people"].append({"pose_keypoints_3d": [0.0] * (25 * 4)})
            continue

        lightweight_keypoints = np.array(frame_data["keypoints"][0]) # Shape (19, 3) or (19, 4) if confidence is there
        
        # Ensure keypoints have confidence, default to 1.0 if not present
        if lightweight_keypoints.shape[1] == 3:
            confidences = np.ones((lightweight_keypoints.shape[0], 1))
            lightweight_keypoints = np.hstack((lightweight_keypoints, confidences))

        openpose_keypoints_3d = [0.0] * (25 * 4) # Initialize with zeros

        # Map existing Lightweight keypoints to OpenPose format
        for lw_idx, op_idx in LIGHTWEIGHT_TO_OPENPOSE_MAP.items():
            if lw_idx < lightweight_keypoints.shape[0]: # Ensure lightweight index is valid
                x, y, z, c = lightweight_keypoints[lw_idx]
                # OpenPose expects flat array: x0,y0,z0,c0, x1,y1,z1,c1, ...
                openpose_keypoints_3d[op_idx*4 + 0] = x
                openpose_keypoints_3d[op_idx*4 + 1] = y
                openpose_keypoints_3d[op_idx*4 + 2] = z
                openpose_keypoints_3d[op_idx*4 + 3] = c

        # Handle OpenPose MidHip (index 8) - interpolate from RHip (9) and LHip (12)
        # Lightweight RHip is 8, LHip is 11
        rh_lw_idx = 8
        lh_lw_idx = 11
        if rh_lw_idx < lightweight_keypoints.shape[0] and lh_lw_idx < lightweight_keypoints.shape[0]:
            rh_pos = lightweight_keypoints[rh_lw_idx, :3]
            lh_pos = lightweight_keypoints[lh_lw_idx, :3]
            mid_hip_pos = (rh_pos + lh_pos) / 2.0
            mid_hip_conf = min(lightweight_keypoints[rh_lw_idx, 3], lightweight_keypoints[lh_lw_idx, 3]) # Take min confidence
            
            op_midhip_idx = 8
            openpose_keypoints_3d[op_midhip_idx*4 + 0] = mid_hip_pos[0]
            openpose_keypoints_3d[op_midhip_idx*4 + 1] = mid_hip_pos[1]
            openpose_keypoints_3d[op_midhip_idx*4 + 2] = mid_hip_pos[2]
            openpose_keypoints_3d[op_midhip_idx*4 + 3] = mid_hip_conf

        openpose_output["people"].append({"pose_keypoints_3d": openpose_keypoints_3d})

    output_filename = os.path.join(output_dir, os.path.basename(input_json_path).replace('_lightweight_smoothed_3d_keypoints.json', '_openpose_3d_keypoints.json'))
    with open(output_filename, 'w') as f:
        json.dump(openpose_output, f, indent=4)
    
    print(f"Converted Lightweight JSON to OpenPose JSON and saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Lightweight 3D keypoints JSON to OpenPose 3D keypoints JSON.")
    parser.add_argument('--input_json_path', type=str, required=True, help="Path to the input Lightweight 3D keypoints JSON file.")
    parser.add_argument('--output_dir', type=str, default="/mnt/d/progress/ani_bender/output_data", help="Directory to save the output OpenPose JSON file.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    convert_lightweight_to_openpose_json(args.input_json_path, args.output_dir)