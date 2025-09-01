import glm
import json
import os
import numpy as np
import argparse
from bvhio.lib.Parser import writeBvh
from bvhio.lib.bvh import BvhContainer, BvhJoint
from SpatialTransform import Pose

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

def convert_json_to_bvh_bvhio(input_json_path, output_dir):
    with open(input_json_path, 'r') as f:
        full_data = json.load(f)
    
    if "people" not in full_data or not full_data["people"]:
        print("Input JSON does not contain 'people' data or it is empty. Cannot convert to BVH.")
        return

    # Extract the relevant keypoints data, assuming each entry in 'people' corresponds to a frame
    # and contains 'pose_keypoints_3d'
    data_3d = []
    for frame_data_raw in full_data["people"]:
        if "pose_keypoints_3d" in frame_data_raw:
            data_3d.append({"keypoints": frame_data_raw["pose_keypoints_3d"]})
        else:
            data_3d.append({"keypoints": []}) # Append empty if no keypoints for this frame

    if not data_3d:
        print("No valid keypoint data found after parsing 'people' array. Cannot convert to BVH.")
        return
    print(f"DEBUG: type(data_3d) = {type(data_3d)}")
    print(f"DEBUG: data_3d[0] = {data_3d[0]}")

    print(f"Generating BVH for Person 1 using bvhio...")
    
    person_data_3d = data_3d # Directly use data_3d as it's already per-frame for one person

    # Get the first frame's keypoints for this person to establish initial bone lengths/offsets
    first_valid_frame_keypoints = None
    for frame_data_person in person_data_3d:
        if frame_data_person["keypoints"] and len(frame_data_person["keypoints"]) > 0:
            first_valid_frame_keypoints = np.array(frame_data_person["keypoints"])
            break
    
    if first_valid_frame_keypoints is None:
        print(f"Person not detected in any frame. Skipping BVH generation for this person.")
        return

    # Dynamically calculate REST_POSE_VECTORS from first valid frame
    REST_POSE_VECTORS = {}
    for bone_name, kp_indices in KEYPOINT_MAP.items():
        if len(kp_indices) == 1:
            idx = kp_indices[0]
            REST_POSE_VECTORS[bone_name] = first_valid_frame_keypoints[idx*4 : idx*4+3]
        elif len(kp_indices) == 2:
            v1 = first_valid_frame_keypoints[kp_indices[0]*4 : kp_indices[0]*4+3]
            v2 = first_valid_frame_keypoints[kp_indices[1]*4 : kp_indices[1]*4+3]
            REST_POSE_VECTORS[bone_name] = (v1 + v2) / 2

    # Define the default orientation vector for each bone in a T-pose (Y-up, Z-forward)
    # This is crucial for calculating rotations relative to a rest pose.
    # These vectors point from parent joint to child joint in a conceptual T-pose.
    # Dynamically calculate REST_POSE_VECTORS from the first valid frame's keypoints
    REST_POSE_VECTORS = {}
    for bone_name, bone_info in BVH_SKELETON.items():
        if bone_name == "Hips":
            REST_POSE_VECTORS[bone_name] = np.array([0, 0, 0]) # Root has no direction
        else:
            parent_kp_indices = KEYPOINT_MAP[bone_info["parent"]]
            current_kp_indices = KEYPOINT_MAP[bone_name]

            # Calculate midpoint for parent and current keypoints if multiple indices are given
            parent_pos_rest = None
            if len(parent_kp_indices) == 1:
                parent_pos_rest = first_valid_frame_keypoints[parent_kp_indices[0]*4 : parent_kp_indices[0]*4+3]
            elif len(parent_kp_indices) == 2:
                parent_pos_rest = (first_valid_frame_keypoints[parent_kp_indices[0]*4 : parent_kp_indices[0]*4+3] + first_valid_frame_keypoints[parent_kp_indices[1]*4 : parent_kp_indices[1]*4+3]) / 2
            else: # Handle special cases for parent like Spine, Hips, Neck, Head
                if bone_info["parent"] == "Head":
                    parent_pos_rest = first_valid_frame_keypoints[0*4 : 0*4+3]
                elif bone_info["parent"] == "Spine":
                    neck_pos = first_valid_frame_keypoints[1*4 : 1*4+3]
                    hips_mid_pos = (first_valid_frame_keypoints[8*4 : 8*4+3] + first_valid_frame_keypoints[11*4 : 11*4+3]) / 2
                    parent_pos_rest = (neck_pos + hips_mid_pos) / 2
                elif bone_info["parent"] == "Hips":
                    parent_pos_rest = (first_valid_frame_keypoints[8*4 : 8*4+3] + first_valid_frame_keypoints[11*4 : 11*4+3]) / 2
                elif bone_info["parent"] == "Neck":
                    parent_pos_rest = first_valid_frame_keypoints[1*4 : 1*4+3]

            current_pos_rest = None
            if len(current_kp_indices) == 1:
                current_pos_rest = first_valid_frame_keypoints[current_kp_indices[0]*4 : current_kp_indices[0]*4+3]
            elif len(current_kp_indices) == 2:
                current_pos_rest = (first_valid_frame_keypoints[current_kp_indices[0]*4 : current_kp_indices[0]*4+3] + first_valid_frame_keypoints[current_kp_indices[1]*4 : current_kp_indices[1]*4+3]) / 2
            else: # Handle special cases for current like Spine, Hips, Neck, Head
                if bone_name == "Head":
                    current_pos_rest = first_valid_frame_keypoints[0*4 : 0*4+3]
                elif bone_name == "Spine":
                    neck_pos = first_valid_frame_keypoints[1*4 : 1*4+3]
                    hips_mid_pos = (first_valid_frame_keypoints[8*4 : 8*4+3] + first_valid_frame_keypoints[11*4 : 11*4+3]) / 2
                    current_pos_rest = (neck_pos + hips_mid_pos) / 2
                elif bone_name == "Hips":
                    current_pos_rest = (first_valid_frame_keypoints[8*4 : 8*4+3] + first_valid_frame_keypoints[11*4 : 11*4+3]) / 2
                elif bone_name == "Neck":
                    current_pos_rest = first_valid_frame_keypoints[1*4 : 1*4+3]

            if parent_pos_rest is not None and current_pos_rest is not None:
                REST_POSE_VECTORS[bone_name] = current_pos_rest - parent_pos_rest
            else:
                REST_POSE_VECTORS[bone_name] = np.array([0, 0, 0]) # Fallback for safety

    # Initialize BvhContainer
    bvh_container = BvhContainer()
    bvh_container.name = "Hips" # Root joint name
    bvh_container.frame_time = 1/30.0 # Assuming 30 FPS
    bvh_container.FrameTime = 1/30.0 # Explicitly set FrameTime for writing

    # Build Hierarchy (BvhJoint objects)
    offsets = {}
    joint_positions_map = {} # To store calculated 3D positions for each "bone"
    
    # Calculate initial positions for BVH joints based on keypoints
    for bone_name, kp_indices in KEYPOINT_MAP.items():
        valid_kp_indices = [idx for idx in kp_indices if idx < first_valid_frame_keypoints.shape[0]]
        
        if not valid_kp_indices: 
            joint_positions_map[bone_name] = np.array([0.0, 0.0, 0.0])
            continue

        if len(valid_kp_indices) == 1:
            joint_positions_map[bone_name] = first_valid_frame_keypoints[valid_kp_indices[0]*4 : valid_kp_indices[0]*4+3]
        elif len(valid_kp_indices) == 2:
            joint_positions_map[bone_name] = (first_valid_frame_keypoints[valid_kp_indices[0]*4 : valid_kp_indices[0]*4+3] + first_valid_frame_keypoints[valid_kp_indices[1]*4 : valid_kp_indices[1]*4+3]) / 2
        else: # Handle special cases like Spine, Hips, Neck, Head
            if bone_name == "Head":
                joint_positions_map[bone_name] = first_valid_frame_keypoints[0*4 : 0*4+3]
            elif bone_name == "Spine":
                neck_pos = first_valid_frame_keypoints[1*4 : 1*4+3]
                hips_mid_pos = (first_valid_frame_keypoints[8*4 : 8*4+3] + first_valid_frame_keypoints[11*4 : 11*4+3]) / 2
                joint_positions_map[bone_name] = (neck_pos + hips_mid_pos) / 2
            elif bone_name == "Hips":
                joint_positions_map[bone_name] = (first_valid_frame_keypoints[8*4 : 8*4+3] + first_valid_frame_keypoints[11*4 : 11*4+3]) / 2
            elif bone_name == "Neck":
                joint_positions_map[bone_name] = first_valid_frame_keypoints[1*4 : 1*4+3]

    for bone_name, bone_info in BVH_SKELETON.items():
        if bone_name == "Hips":
            offsets[bone_name] = joint_positions_map[bone_name]
        else:
            parent_pos = joint_positions_map[bone_info["parent"]]
            current_pos = joint_positions_map[bone_name]
            offsets[bone_name] = current_pos - parent_pos

    # Create BvhJoint objects and link them
    bvh_joints = {}
    def create_bvh_joint(joint_name, parent_bvh_joint=None):
        bone_info = BVH_SKELETON[joint_name]
        offset_vec = offsets[joint_name]
        channels = bone_info["channels"]
        
        bvh_joint = BvhJoint(joint_name, glm.vec3(offset_vec[0], offset_vec[1], offset_vec[2]))
        bvh_joint.Channels = channels # Assign channels after creation
        bvh_joints[joint_name] = bvh_joint

        if parent_bvh_joint:
            parent_bvh_joint.Children.append(bvh_joint)
        
        for child_bone in bone_info["children"]:
            create_bvh_joint(child_bone, bvh_joint)

    create_bvh_joint("Hips") # Start building hierarchy from Hips
    bvh_container.joints = [bvh_joints["Hips"]] # Assign the root joint
    bvh_container.Root = bvh_joints["Hips"] # Explicitly set the root

    # Add motion data (frames)
    # For each frame, calculate pose for each joint and append to joint.Keyframes
    for frame_data_person in person_data_3d:
        if not frame_data_person["keypoints"]:
            # If person not detected, append default pose (T-pose) to each joint's Keyframes
            for joint_name in bvh_joints:
                bvh_joints[joint_name].Keyframes.append(Pose(glm.vec3(0,0,0), glm.quat(1,0,0,0))) # Default Pose
        else:
            person_keypoints_current_frame = np.array(frame_data_person["keypoints"])
            
            current_joint_positions_map = {}
            for bone_name, kp_indices in KEYPOINT_MAP.items():
                valid_kp_indices = [idx for idx in kp_indices if idx < person_keypoints_current_frame.shape[0]]
                
                if not valid_kp_indices or len(person_keypoints_current_frame) == 0:
                    current_joint_positions_map[bone_name] = np.array([0.0, 0.0, 0.0])
                    continue

                if len(valid_kp_indices) == 1:
                    current_joint_positions_map[bone_name] = person_keypoints_current_frame[valid_kp_indices[0]*4 : valid_kp_indices[0]*4+3]
                elif len(valid_kp_indices) == 2:
                    current_joint_positions_map[bone_name] = (person_keypoints_current_frame[valid_kp_indices[0]*4 : valid_kp_indices[0]*4+3] + person_keypoints_current_frame[valid_kp_indices[1]*4 : valid_kp_indices[1]*4+3]) / 2
                else: # Handle special cases like Spine, Hips, Neck, Head
                    if bone_name == "Head":
                        current_joint_positions_map[bone_name] = person_keypoints_current_frame[0*4 : 0*4+3]
                    elif bone_name == "Spine":
                        neck_pos = person_keypoints_current_frame[1*4 : 1*4+3]
                        hips_mid_pos = (person_keypoints_current_frame[8*4 : 8*4+3] + person_keypoints_current_frame[11*4 : 11*4+3]) / 2
                        current_joint_positions_map[bone_name] = (neck_pos + hips_mid_pos) / 2
                    elif bone_name == "Hips":
                        current_joint_positions_map[bone_name] = (person_keypoints_current_frame[8*4 : 8*4+3] + person_keypoints_current_frame[11*4 : 11*4+3]) / 2
                    elif bone_name == "Neck":
                        current_joint_positions_map[bone_name] = person_keypoints_current_frame[1*4 : 1*4+3]

            # Now, for each joint, calculate its pose and append to its Keyframes
            for joint_name in bvh_joints:
                bone_info = BVH_SKELETON[joint_name]
                
                position = glm.vec3(0,0,0) # Default position
                rotation = glm.quat(1,0,0,0) # Default rotation (identity)

                if bone_info["parent"] is None: # Root bone (Hips)
                    root_pos = current_joint_positions_map["Hips"]
                    position = glm.vec3(root_pos[0], root_pos[1], root_pos[2])

                    # Root rotation (e.g., based on Hips to Spine direction)
                    if "Spine" in BVH_SKELETON["Hips"]["children"]:
                        current_bone_vec = current_joint_positions_map["Spine"] - root_pos
                        rest_pose_vec = REST_POSE_VECTORS["Spine"]
                        R = rotation_matrix_from_vectors(rest_pose_vec, current_bone_vec)
                        rotation = glm.quat_cast(glm.mat3(R)) # Convert rotation matrix to quaternion
                    
                else: # Child bones
                    parent_pos = current_joint_positions_map[bone_info["parent"]]
                    current_pos = current_joint_positions_map[joint_name]
                    
                    current_bone_vec = current_pos - parent_pos
                    rest_pose_vec = REST_POSE_VECTORS[joint_name]
                    
                    R = rotation_matrix_from_vectors(rest_pose_vec, current_bone_vec)
                    rotation = glm.quat_cast(glm.mat3(R)) # Convert rotation matrix to quaternion
                
                bvh_joints[joint_name].Keyframes.append(Pose(position, rotation))

    bvh_container.nframes = len(person_data_3d)
    bvh_container.FrameCount = len(person_data_3d) # Explicitly set FrameCount

    # Write to file for this person
    filename_without_ext = os.path.splitext(os.path.basename(input_json_path))[0]
    # Remove known suffixes to get a clean base name for the BVH file
    if filename_without_ext.endswith('_lightweight_smoothed_3d_keypoints'):
        base_filename = filename_without_ext.replace('_lightweight_smoothed_3d_keypoints', '')
    elif filename_without_ext.endswith('_vibe_smoothed_3d_keypoints'):
        base_filename = filename_without_ext.replace('_vibe_smoothed_3d_keypoints', '')
    elif filename_without_ext.endswith('_videopose3d_smoothed_3d_keypoints'):
        base_filename = filename_without_ext.replace('_videopose3d_smoothed_3d_keypoints', '')
    else:
        base_filename = filename_without_ext # Fallback if suffix not found

    output_filename = os.path.join(output_dir, f'{base_filename}.bvh') # Removed person_idx + 1
    writeBvh(output_filename, bvh_container)
    
    print(f"BVH file generated for Person 1 using bvhio and saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert smoothed 3D keypoints to BVH format using bvhio.")
    parser.add_argument('--input_json_path', type=str, required=True, help="Path to the input smoothed 3D keypoints JSON file.")
    parser.add_argument('--output_dir', type=str, default="/mnt/d/progress/ani_bender/output_data", help="Directory to save the output BVH file.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    convert_json_to_bvh_bvhio(args.input_json_path, args.output_dir)