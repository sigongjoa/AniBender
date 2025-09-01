import glm
import json
import os
import numpy as np
import argparse
from bvhio.lib.Parser import writeBvh
from bvhio.lib.bvh import BvhContainer, BvhJoint
from SpatialTransform import Pose

# --- SKELETON DEFINITION (GROUND TRUTH FROM VIDEOPOS3D SOURCE) ---

# This entire section is replicated from the VideoPose3D/common/h36m_dataset.py and skeleton.py
# to ensure the BVH skeleton is IDENTICAL to the one used by the model.

class Skeleton:
    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)
        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_children()

    def _compute_children(self):
        self._children = []
        for i in range(len(self._parents)):
            self._children.append([])
        for i, p in enumerate(self._parents):
            if p != -1:
                self._children[p].append(i)

    def parents(self):
        return self._parents

    def children(self):
        return self._children

    def remove_joints(self, joints_to_remove):
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]
                
        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)
        self._compute_children()
        return valid_joints

# Original 32-joint skeleton from Human3.6M
h36m_skeleton_raw = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12, 16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                             joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                             joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

# The joint names for the 32-joint skeleton
h36m_joint_names_raw = ['Hips', 'RHip', 'RKnee', 'RFoot', 'RFoot_dummy', 'RToe_dummy', 'LHip', 'LKnee', 'LFoot', 'LFoot_dummy', 'LToe_dummy', 'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist', 'RHand_dummy1', 'RHand_dummy2', 'RHand_dummy3', 'LHand_dummy1', 'LHand_dummy2', 'LHand_dummy3', 'RThumb_dummy', 'RThumb_dummy2', 'RThumb_dummy3', 'LThumb_dummy', 'LThumb_dummy2']

# Perform the same joint removal and rewiring as the model's source code
joints_to_remove = [4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]
h36m_skeleton_17 = h36m_skeleton_raw
valid_joints = h36m_skeleton_17.remove_joints(joints_to_remove)
h36m_skeleton_17._parents[11] = 8 # Rewire LShoulder
h36m_skeleton_17._parents[14] = 8 # Rewire RShoulder

# Get the final list of 17 joint names
h36m_joint_names_17 = [name for i, name in enumerate(h36m_joint_names_raw) if i not in joints_to_remove]

# Now, create the BVH_SKELETON and KEYPOINT_MAP based on this ground truth
BVH_SKELETON = {}
for i, parent_idx in enumerate(h36m_skeleton_17.parents()):
    joint_name = h36m_joint_names_17[i]
    parent_name = h36m_joint_names_17[parent_idx] if parent_idx != -1 else None
    children_names = [h36m_joint_names_17[child_idx] for child_idx in h36m_skeleton_17.children()[i]]
    
    channels = ["Zrotation", "Xrotation", "Yrotation"]
    if parent_name is None: # Root joint
        channels = ["Xposition", "Yposition", "Zposition"] + channels

    BVH_SKELETON[joint_name] = {
        "parent": parent_name,
        "channels": channels,
        "children": children_names
    }

# The KEYPOINT_MAP now simply maps the joint name to its index in the 17-point skeleton
KEYPOINT_MAP = {name: [i] for i, name in enumerate(h36m_joint_names_17)}

# --- UTILITY FUNCTIONS ---

def rotation_matrix_from_vectors(vec1, vec2):
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    if vec1_norm < 1e-6 or vec2_norm < 1e-6:
        return np.eye(3)
    a = vec1 / vec1_norm
    b = vec2 / vec2_norm
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-6:
        return np.eye(3) if c > 0 else -np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

def get_joint_positions(keypoints_array, keypoint_map):
    joint_positions = {}
    for bone_name, kp_indices in keypoint_map.items():
        idx = kp_indices[0]
        # The input from VideoPose3D is already (N, 3), not (N, 4)
        if idx < len(keypoints_array):
            joint_positions[bone_name] = keypoints_array[idx]
        else:
            joint_positions[bone_name] = np.array([0.0, 0.0, 0.0])
    return joint_positions

# --- MAIN CONVERSION LOGIC ---

def convert_json_to_bvh_bvhio(input_json_path, output_dir):
    with open(input_json_path, 'r') as f:
        data_from_videopose = json.load(f)

    # Adapt the VideoPose3D JSON structure
    person_data_3d = []
    for frame_data in data_from_videopose:
        if frame_data["keypoints"]:
            # Data is nested: list of frames -> list of people -> list of keypoints
            person_keypoints = frame_data["keypoints"][0] # Take the first person
            person_data_3d.append(np.array(person_keypoints)[:, :3]) # Use only x,y,z
        else:
            person_data_3d.append(np.array([]))

    if not person_data_3d:
        print("No valid keypoint data found. Aborting.")
        return

    print("Generating BVH based on VideoPose3D's exact skeleton specification...")
    
    first_valid_frame_keypoints = None
    for keypoints in person_data_3d:
        if keypoints.size > 0:
            first_valid_frame_keypoints = keypoints
            break
    
    if first_valid_frame_keypoints is None:
        print("No valid frames found. Aborting.")
        return

    # The data from VideoPose3D is already in a good coordinate system.
    rest_joint_positions = get_joint_positions(first_valid_frame_keypoints, KEYPOINT_MAP)

    offsets = {}
    for bone_name in h36m_joint_names_17:
        bone_info = BVH_SKELETON[bone_name]
        if bone_info["parent"] is None:
            offsets[bone_name] = np.array([0.0, 0.0, 0.0])
        else:
            parent_pos = rest_joint_positions[bone_info["parent"]]
            current_pos = rest_joint_positions[bone_name]
            offsets[bone_name] = current_pos - parent_pos

    ideal_rest_pose_vectors = {}
    for bone_name, offset_vec in offsets.items():
        bone_length = np.linalg.norm(offset_vec)
        if bone_length < 1e-6:
            ideal_rest_pose_vectors[bone_name] = np.array([0.0, 1.0, 0.0]) # Default to Y-up
            continue
        ideal_rest_pose_vectors[bone_name] = offset_vec / bone_length

    bvh_container = BvhContainer()
    bvh_container.FrameTime = 1/50.0 # VideoPose3D operates at 50fps
    bvh_joints = {}

    def create_bvh_joint(joint_name, parent_bvh_joint=None):
        bone_info = BVH_SKELETON[joint_name]
        offset_vec = offsets[joint_name]
        bvh_joint = BvhJoint(joint_name, glm.vec3(offset_vec[0], offset_vec[1], offset_vec[2]))
        bvh_joint.Channels = bone_info["channels"]
        bvh_joints[joint_name] = bvh_joint
        if parent_bvh_joint:
            parent_bvh_joint.Children.append(bvh_joint)
        for child_bone in bone_info["children"]:
            create_bvh_joint(child_bone, bvh_joint)

    create_bvh_joint(h36m_joint_names_17[0])
    bvh_container.Root = bvh_joints[h36m_joint_names_17[0]]

    for frame_keypoints in person_data_3d:
        if frame_keypoints.size == 0:
            for joint_name in bvh_joints:
                bvh_joints[joint_name].Keyframes.append(Pose(glm.vec3(0,0,0), glm.quat(1,0,0,0)))
            continue

        current_joint_positions = get_joint_positions(frame_keypoints, KEYPOINT_MAP)

        for joint_name, bvh_joint in bvh_joints.items():
            bone_info = BVH_SKELETON[joint_name]
            position = glm.vec3(0,0,0)
            rotation = glm.quat(1,0,0,0)

            if bone_info["parent"] is None:
                root_pos = current_joint_positions[joint_name]
                position = glm.vec3(root_pos[0], root_pos[1], root_pos[2])
                
                child_name = bone_info["children"][0]
                current_bone_vec = current_joint_positions[child_name] - root_pos
                rest_pose_vec = ideal_rest_pose_vectors[child_name]
                R = rotation_matrix_from_vectors(rest_pose_vec, current_bone_vec)
                rotation = glm.quat_cast(glm.mat3(R))
            else:
                parent_pos = current_joint_positions[bone_info["parent"]]
                current_pos = current_joint_positions[joint_name]
                current_bone_vec = current_pos - parent_pos
                rest_pose_vec = ideal_rest_pose_vectors[joint_name]
                R = rotation_matrix_from_vectors(rest_pose_vec, current_bone_vec)
                rotation = glm.quat_cast(glm.mat3(R))
            
            bvh_joint.Keyframes.append(Pose(position, rotation))

    bvh_container.FrameCount = len(person_data_3d)

    filename_without_ext = os.path.splitext(os.path.basename(input_json_path))[0]
    base_filename = filename_without_ext.replace('_videopose3d_smoothed_3d_keypoints', '')
    output_filename = os.path.join(output_dir, f'{base_filename}_videopose3d.bvh')
    writeBvh(output_filename, bvh_container)
    
    print(f"BVH file successfully generated and saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VideoPose3D JSON to BVH format.")
    parser.add_argument('--input_json_path', type=str, required=True, help="Path to the input 3D keypoints JSON file from VideoPose3D.")
    parser.add_argument('--output_dir', type=str, default="output_data", help="Directory to save the output BVH file.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    convert_json_to_bvh_bvhio(args.input_json_path, args.output_dir)