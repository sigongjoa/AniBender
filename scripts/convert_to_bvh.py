import json
import os
import numpy as np
import argparse
import mediapipe as mp # Added import for mediapipe

# Define the 19 keypoints from Lightweight Human Pose Estimation 3D Demo
# 0: nose, 1: neck, 2: right_shoulder, 3: right_elbow, 4: right_wrist,
# 5: left_shoulder, 6: left_elbow, 7: left_wrist, 8: right_hip, 9: right_knee,
# 10: right_ankle, 11: left_hip, 12: left_knee, 13: left_ankle, 14: right_eye,
# 15: left_eye, 16: right_ear, 17: left_ear, 18: background (ignored)

# Simplified BVH skeleton definition and mapping to Lightweight keypoints
# This is a conceptual mapping. Actual bone lengths and orientations
# would need to be calibrated for a specific avatar.
BVH_SKELETON = {
    "Hips": {"parent": None, "channels": ["Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"], "children": ["Spine", "LeftUpLeg", "RightUpLeg"]},
    "Spine": {"parent": "Hips", "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["Neck", "LeftShoulder", "RightShoulder"]},
    "Neck": {"parent": "Spine", "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["Head"]},
    "Head": {"parent": "Neck", "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": []},
    "LeftShoulder": {"parent": "Spine", "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftArm"]},
    "LeftArm": {"parent": "LeftShoulder", "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftForeArm"]},
    "LeftForeArm": {"parent": "LeftArm", "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftHand"]},
    "LeftHand": {"parent": "LeftForeArm", "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": []}, # Placeholder for wrist
    "RightShoulder": {"parent": "Spine", "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightArm"]},
    "RightArm": {"parent": "RightShoulder", "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightForeArm"]},
    "RightForeArm": {"parent": "RightArm", "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightHand"]},
    "RightHand": {"parent": "RightForeArm", "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": []}, # Placeholder for wrist
    "LeftUpLeg": {"parent": "Hips", "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftLeg"]},
    "LeftLeg": {"parent": "LeftUpLeg", "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["LeftFoot"]},
    "LeftFoot": {"parent": "LeftLeg", "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": []}, # Placeholder for ankle
    "RightUpLeg": {"parent": "Hips", "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightLeg"]},
    "RightLeg": {"parent": "RightUpLeg", "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": ["RightFoot"]},
    "RightFoot": {"parent": "RightLeg", "channels": ["Zrotation", "Xrotation", "Yrotation"], "children": []}, # Placeholder for ankle
}

# Mapping from BVH bone names to the 19-keypoint array indices from Lightweight Human Pose Estimation 3D Demo
KEYPOINT_MAP = {
    "Hips": [8, 11], # Midpoint of Right Hip (8) and Left Hip (11)
    "Spine": [1],  # Neck (1)
    "Neck": [1],      # Neck (1)
    "Head": [0],      # Nose (0)
    "LeftShoulder": [5], # Left Shoulder (5)
    "LeftArm": [5, 6], # Left Shoulder (5) to Left Elbow (6)
    "LeftForeArm": [6, 7], # Left Elbow (6) to Left Wrist (7)
    "LeftHand": [7], # Left Wrist (7)
    "RightShoulder": [2], # Right Shoulder (2)
    "RightArm": [2, 3], # Right Shoulder (2) to Right Elbow (3)
    "RightForeArm": [3, 4], # Right Elbow (3) to Right Wrist (4)
    "RightHand": [4], # Right Wrist (4)
    "LeftUpLeg": [11, 12], # Left Hip (11) to Left Knee (12)
    "LeftLeg": [12, 13], # Left Knee (12) to Left Ankle (13)
    "LeftFoot": [13], # Left Ankle (13)
    "RightUpLeg": [8, 9], # Right Hip (8) to Right Knee (9)
    "RightLeg": [9, 10], # Right Knee (9) to Right Ankle (10)
    "RightFoot": [10], # Right Ankle (10)
}

# Define the default orientation vector for each bone in a T-pose (Y-up, Z-forward)
# This is crucial for calculating rotations relative to a rest pose.
# These vectors point from parent joint to child joint in a conceptual T-pose.
REST_POSE_VECTORS = {
    "Hips": np.array([0, 0, 0]), # Root has no direction
    "Spine": np.array([0, 1, 0]), # Upwards along Y
    "Neck": np.array([0, 1, 0]), # Upwards along Y
    "Head": np.array([0, 1, 0]), # Upwards along Y
    "LeftShoulder": np.array([-1, 0, 0]), # Leftwards along X
    "LeftArm": np.array([-1, 0, 0]), # Leftwards along X
    "LeftForeArm": np.array([-1, 0, 0]), # Leftwards along X
    "LeftHand": np.array([-1, 0, 0]), # Leftwards along X
    "RightShoulder": np.array([1, 0, 0]), # Rightwards along X
    "RightArm": np.array([1, 0, 0]), # Rightwards along X
    "RightForeArm": np.array([1, 0, 0]), # Rightwards along X
    "RightHand": np.array([1, 0, 0]), # Rightwards along X
    "LeftUpLeg": np.array([0, -1, 0]), # Downwards along Y
    "LeftLeg": np.array([0, -1, 0]), # Downwards along Y
    "LeftFoot": np.array([0, -1, 0]), # Downwards along Y
    "RightUpLeg": np.array([0, -1, 0]), # Downwards along Y
    "RightLeg": np.array([0, -1, 0]), # Downwards along Y
    "RightFoot": np.array([0, -1, 0]), # Downwards along Y
}

# Function to calculate rotation matrix from one vector to another
def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2.
    Handles zero vectors by returning identity.
    """
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)

    if vec1_norm < 1e-6 or vec2_norm < 1e-6: # Handle near-zero vectors
        return np.eye(3)

    a = vec1 / vec1_norm
    b = vec2 / vec2_norm
    
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    if s < 1e-6: # Vectors are parallel or anti-parallel
        return np.eye(3) if c > 0 else -np.eye(3) # Same or opposite direction

    kmat = np.array([ [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0] ])
    
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

# Function to convert rotation matrix to Euler angles (ZXY order for BVH)
def rotation_matrix_to_euler_zxy(R):
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.degrees(z), np.degrees(x), np.degrees(y) # Z, X, Y

def convert_to_bvh(input_json_path, output_dir):
    with open(input_json_path, 'r') as f:
        data_3d = json.load(f)

    if not data_3d:
        print("Input JSON is empty. Cannot convert to BVH.")
        return

    # Iterate through each person detected in the first frame to get initial offsets
    # We will generate a BVH for each person found.
    num_people_in_first_frame = len(data_3d[0]["keypoints"])
    if num_people_in_first_frame == 0:
        print("No people detected in the first frame. Cannot generate BVH.")
        return

    for person_idx in range(num_people_in_first_frame):
        print(f"Generating BVH for Person {person_idx + 1}...")
        
        # Filter data for this specific person across all frames
        person_data_3d = []
        for frame_data in data_3d:
            if len(frame_data["keypoints"]) > person_idx:
                person_data_3d.append({
                    "keypoints": [frame_data["keypoints"][person_idx]] # Wrap in list for consistency
                })
            else:
                # If this person is not detected in a frame, add an empty keypoint list
                person_data_3d.append({
                    "keypoints": []
                })

        # Get the first frame's keypoints for this person to establish initial bone lengths/offsets
        # Find the first frame where this person is actually detected
        first_valid_frame_keypoints = None
        for frame_data_person in person_data_3d:
            if frame_data_person["keypoints"]:
                first_valid_frame_keypoints = np.array(frame_data_person["keypoints"][0])
                break
        
        if first_valid_frame_keypoints is None:
            print(f"Person {person_idx + 1} not detected in any frame. Skipping BVH generation for this person.")
            continue

        offsets = {}
        joint_positions_map = {} # To store calculated 3D positions for each "bone"
        
        # Calculate initial positions for BVH joints based on keypoints
        for bone_name, kp_indices in KEYPOINT_MAP.items():
            # Ensure keypoint indices are within bounds of first_valid_frame_keypoints
            valid_kp_indices = [idx for idx in kp_indices if idx < first_valid_frame_keypoints.shape[0]]
            
            if not valid_kp_indices: # If no valid keypoints for this bone
                joint_positions_map[bone_name] = np.array([0.0, 0.0, 0.0]) # Default to origin
                continue

            if len(valid_kp_indices) == 1:
                joint_positions_map[bone_name] = first_valid_frame_keypoints[valid_kp_indices[0], :3]
            elif len(valid_kp_indices) == 2:
                joint_positions_map[bone_name] = (first_valid_frame_keypoints[valid_kp_indices[0], :3] + first_valid_frame_keypoints[valid_kp_indices[1], :3]) / 2
            else: # Handle special cases like Spine, Hips, Neck, Head (using hardcoded indices from 19-keypoint array)
                if bone_name == "Head":
                    joint_positions_map[bone_name] = first_valid_frame_keypoints[0, :3] # Nose (index 0)
                elif bone_name == "Spine":
                    # Midpoint of Neck and Hips
                    neck_pos = first_valid_frame_keypoints[1, :3]
                    hips_mid_pos = (first_valid_frame_keypoints[8, :3] + first_valid_frame_keypoints[11, :3]) / 2
                    joint_positions_map[bone_name] = (neck_pos + hips_mid_pos) / 2
                elif bone_name == "Hips":
                    joint_positions_map[bone_name] = (first_valid_frame_keypoints[8, :3] + first_valid_frame_keypoints[11, :3]) / 2 # Mid-hip (indices 8, 11)
                elif bone_name == "Neck":
                    joint_positions_map[bone_name] = first_valid_frame_keypoints[1, :3] # Neck (index 1)

        # Calculate offsets for BVH HIERARCHY section
        # For MediaPipe world_landmarks, Hips is already at (0,0,0) relative to its own coordinate system
        # So, the root offset should be the absolute position of Hips in the first frame.
        # All other offsets are relative to their parents.
        for bone_name, bone_info in BVH_SKELETON.items():
            if bone_name == "Hips": # Root bone
                offsets[bone_name] = joint_positions_map[bone_name] # Absolute position for root
            else:
                parent_pos = joint_positions_map[bone_info["parent"]]
                current_pos = joint_positions_map[bone_name]
                offsets[bone_name] = current_pos - parent_pos

        # BVH HIERARCHY section
        bvh_hierarchy = []
        
        def write_bone_hierarchy(bone_name, indent_level):
            indent = "  " * indent_level
            bone_info = BVH_SKELETON[bone_name]
            
            if bone_info["parent"] is None: # Root
                bvh_hierarchy.append(f"ROOT {bone_name}")
            else:
                bvh_hierarchy.append(f"{indent}JOINT {bone_name}")

            bvh_hierarchy.append(f"{indent}{{")
            
            offset_vec = offsets[bone_name]
            bvh_hierarchy.append(f"{indent}  OFFSET {offset_vec[0]:.6f} {offset_vec[1]:.6f} {offset_vec[2]:.6f}")
            
            if bone_info["channels"]:
                channels_str = " ".join(bone_info['channels'])
                bvh_hierarchy.append(f"{indent}  CHANNELS {len(bone_info['channels'])} {channels_str}")
            
            for child_bone in bone_info["children"]:
                write_bone_hierarchy(child_bone, indent_level + 1)
            
            if not bone_info["children"]: # End Site for leaf joints
                bvh_hierarchy.append(f"{indent}  End Site")
                bvh_hierarchy.append(f"{indent}  {{")
                bvh_hierarchy.append(f"{indent}    OFFSET 0.000000 0.000000 0.000000")
                bvh_hierarchy.append(f"{indent}  }}")
                
            bvh_hierarchy.append(f"{indent}}}")

        write_bone_hierarchy("Hips", 0) # Start with the root bone

        # BVH MOTION section
        bvh_motion = ["MOTION"]
        bvh_motion.append(f"Frames: {len(person_data_3d)}")
        bvh_motion.append(f"Frame Time: 0.033333") # Assuming 30 FPS (1/30)

        # Process each frame for this person
        for frame_idx, frame_data_person in enumerate(person_data_3d):
            if not frame_data_person["keypoints"]: # If this person is not detected in this frame
                # Append default pose (T-pose) if person not detected
                motion_line = []
                for bone_name, bone_info in BVH_SKELETON.items():
                    if bone_info["parent"] is None: # Root
                        motion_line.extend([0.0, 0.0, 0.0]) # Default root position
                    motion_line.extend([0.0, 0.0, 0.0]) # Default rotations
                bvh_motion.append(" ".join(f"{val:.6f}" for val in motion_line))
                continue

            person_keypoints_current_frame = np.array(frame_data_person["keypoints"][0]) 
            
            current_frame_motion_data = []

            # Calculate current joint positions for this frame
            current_joint_positions_map = {}
            for bone_name, kp_indices in KEYPOINT_MAP.items():
                # Ensure keypoint indices are within bounds of person_keypoints_current_frame
                valid_kp_indices = [idx for idx in kp_indices if idx < person_keypoints_current_frame.shape[0]]
                
                if not valid_kp_indices: # If no valid keypoints for this bone
                    current_joint_positions_map[bone_name] = np.array([0.0, 0.0, 0.0]) # Default to origin
                    continue

                if len(valid_kp_indices) == 1:
                    current_joint_positions_map[bone_name] = person_keypoints_current_frame[valid_kp_indices[0], :3]
                elif len(valid_kp_indices) == 2:
                    current_joint_positions_map[bone_name] = (person_keypoints_current_frame[valid_kp_indices[0], :3] + person_keypoints_current_frame[valid_kp_indices[1], :3]) / 2
                else: # Handle special cases like Spine, Hips, Neck, Head (using hardcoded indices from 19-keypoint array)
                    if bone_name == "Head":
                        current_joint_positions_map[bone_name] = person_keypoints_current_frame[0, :3] # Nose (index 0)
                    elif bone_name == "Spine":
                        # Midpoint of Neck and Hips
                        neck_pos = person_keypoints_current_frame[1, :3]
                        hips_mid_pos = (person_keypoints_current_frame[8, :3] + person_keypoints_current_frame[11, :3]) / 2
                        current_joint_positions_map[bone_name] = (neck_pos + hips_mid_pos) / 2
                    elif bone_name == "Hips":
                        current_joint_positions_map[bone_name] = (person_keypoints_current_frame[8, :3] + person_keypoints_current_frame[11, :3]) / 2 # Mid-hip (indices 8, 11)
                    elif bone_name == "Neck":
                        current_joint_positions_map[bone_name] = person_keypoints_current_frame[1, :3] # Neck (index 1)

            # Calculate rotations for each bone
            for bone_name, bone_info in BVH_SKELETON.items():
                if bone_info["parent"] is None: # Root bone
                    # Root position is absolute
                    # MediaPipe world_landmarks are relative to Hips, so Hips position is its own absolute position
                    root_pos = current_joint_positions_map["Hips"]
                    current_frame_motion_data.extend(root_pos.tolist()) # Use actual Hips position
                    
                    # Root rotation (e.g., based on Hips to Spine direction)
                    if "Spine" in BVH_SKELETON["Hips"]["children"]:
                        current_bone_vec = current_joint_positions_map["Spine"] - root_pos
                        rest_pose_vec = REST_POSE_VECTORS["Spine"]
                        
                        R = rotation_matrix_from_vectors(rest_pose_vec, current_bone_vec)
                        rot_z, rot_x, rot_y = rotation_matrix_to_euler_zxy(R)
                        current_frame_motion_data.extend([rot_z, rot_x, rot_y]) # ZXY order
                    else:
                        current_frame_motion_data.extend([0.0, 0.0, 0.0]) # Default if no spine
                else:
                    # For child bones, calculate rotation relative to parent
                    parent_pos = current_joint_positions_map[bone_info["parent"]]
                    current_pos = current_joint_positions_map[bone_name]
                    
                    # Direction vector from parent to current bone
                    current_bone_vec = current_pos - parent_pos
                    
                    # Get rest pose vector for this bone
                    rest_pose_vec = REST_POSE_VECTORS[bone_name]
                    
                    R = rotation_matrix_from_vectors(rest_pose_vec, current_bone_vec)
                    rot_z, rot_x, rot_y = rotation_matrix_to_euler_zxy(R)
                    current_frame_motion_data.extend([rot_z, rot_x, rot_y]) # ZXY order

            # This line should be outside the bone iteration loop, once per frame
            bvh_motion.append(" ".join(f"{val:.6f}" for val in current_frame_motion_data))

        # Write to file for this person
        # Get the filename without extension
        filename_without_ext = os.path.splitext(os.path.basename(input_json_path))[0]

        # Remove known suffixes
        if filename_without_ext.endswith('_vibe_smoothed_3d_keypoints'):
            base_filename = filename_without_ext.replace('_vibe_smoothed_3d_keypoints', '')
        elif filename_without_ext.endswith('_videopose3d_smoothed_3d_keypoints'):
            base_filename = filename_without_ext.replace('_videopose3d_smoothed_3d_keypoints', '')
        elif filename_without_ext.endswith('_mediapipe_smoothed_3d_keypoints'):
            base_filename = filename_without_ext.replace('_mediapipe_smoothed_3d_keypoints', '')
        else:
            base_filename = filename_without_ext # Fallback if suffix not found
        output_filename = os.path.join(output_dir, f'{base_filename}_person{person_idx + 1}.bvh')
        with open(output_filename, 'w') as f:
            f.write("\n".join(bvh_hierarchy))
            f.write("\n")
            f.write("\n".join(bvh_motion))
        
        print(f"BVH file generated for Person {person_idx + 1} and saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert smoothed 3D keypoints to BVH format.")
    parser.add_argument('--input_json_path', type=str, required=True, help="Path to the input smoothed 3D keypoints JSON file.")
    parser.add_argument('--output_dir', type=str, default="/mnt/d/progress/ani_bender/output_data", help="Directory to save the output BVH file.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    convert_to_bvh(args.input_json_path, args.output_dir)