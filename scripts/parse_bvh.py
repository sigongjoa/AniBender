import json
import os
import numpy as np
import argparse

def parse_bvh(bvh_file_path, output_dir):
    """
    Parses a BVH file and extracts joint hierarchy and motion data.
    Outputs frame-by-frame joint positions.
    """
    hierarchy = {}
    motion_data = []
    frame_time = 0.0
    num_frames = 0

    current_joint = None
    joint_stack = []
    
    # BVH parsing state
    parsing_hierarchy = True
    
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()

    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx].strip()

        if parsing_hierarchy:
            if line.startswith("ROOT") or line.startswith("JOINT"):
                joint_type = line.split()[0]
                joint_name = line.split()[1]
                
                parent_name = None
                if joint_type == "JOINT":
                    if not joint_stack:
                        raise ValueError(f"BVH parsing error: JOINT {joint_name} found without a parent on stack.")
                    parent_name = joint_stack[-1]
                
                hierarchy[joint_name] = {"parent": parent_name, "offset": None, "channels": [], "children": [], "end_site": None}
                if parent_name:
                    hierarchy[parent_name]["children"].append(joint_name)
                
                current_joint = joint_name
                joint_stack.append(joint_name)
                
            elif line.startswith("OFFSET"):
                offset_values = list(map(float, line.split()[1:]))
                hierarchy[current_joint]["offset"] = np.array(offset_values)
            elif line.startswith("CHANNELS"):
                channels = line.split()[2:]
                hierarchy[current_joint]["channels"] = channels
            elif line.startswith("End Site"):
                line_idx += 1 # Skip '{'
                line_idx += 1
                offset_values = list(map(float, lines[line_idx].strip().split()[1:]))
                hierarchy[current_joint]["end_site"] = np.array(offset_values)
                line_idx += 1 # Skip '}'
            elif line == "}":
                if joint_stack:
                    joint_stack.pop()
                # If joint_stack is empty after pop, and we are still in hierarchy, it means we finished a root joint
            elif line == "MOTION":
                parsing_hierarchy = False
                # Read motion header
                line_idx += 1
                num_frames = int(lines[line_idx].strip().split()[1])
                line_idx += 1
                frame_time = float(lines[line_idx].strip().split()[2])
                
                # Read motion data
                for i in range(num_frames):
                    line_idx += 1
                    motion_values = list(map(float, lines[line_idx].strip().split()))
                    motion_data.append(motion_values)
                break # All motion data read, exit loop
        
        line_idx += 1

    # Reconstruct joint global positions for each frame
    all_frame_joint_positions = []

    for frame_motion_values in motion_data:
        current_joint_positions = {}
        channel_idx = 0

        # Recursively calculate global positions
        def calculate_global_position(joint_name, parent_global_pos=np.array([0.0, 0.0, 0.0]), parent_global_rot=np.eye(3)):
            nonlocal channel_idx
            joint_info = hierarchy[joint_name]
            
            local_offset = np.copy(joint_info["offset"]) # Use a copy to avoid modifying original offset
            local_rotation_matrix = np.eye(3)

            # Apply channels in order (ZXY Euler for BVH)
            rot_order = []
            for channel in joint_info["channels"]:
                value = frame_motion_values[channel_idx]
                channel_idx += 1
                
                if "position" in channel.lower(): # Root position channels
                    if joint_name == "Hips": # Only Hips has position channels
                        if channel == "Xposition": local_offset[0] = value
                        elif channel == "Yposition": local_offset[1] = value
                        elif channel == "Zposition": local_offset[2] = value
                elif "rotation" in channel.lower():
                    angle_rad = np.radians(value)
                    if channel == "Xrotation": rot_order.append(('X', angle_rad))
                    elif channel == "Yrotation": rot_order.append(('Y', angle_rad))
                    elif channel == "Zrotation": rot_order.append(('Z', angle_rad))
            
            # Apply rotations in specified order (ZXY for BVH)
            for axis, angle_rad in rot_order:
                if axis == 'X':
                    local_rotation_matrix = local_rotation_matrix @ np.array([
                        [1, 0, 0],
                        [0, np.cos(angle_rad), -np.sin(angle_rad)],
                        [0, np.sin(angle_rad), np.cos(angle_rad)]
                    ])
                elif axis == 'Y':
                    local_rotation_matrix = local_rotation_matrix @ np.array([
                        [np.cos(angle_rad), 0, np.sin(angle_rad)],
                        [0, 1, 0],
                        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
                    ])
                elif axis == 'Z':
                    local_rotation_matrix = local_rotation_matrix @ np.array([
                        [np.cos(angle_rad), -np.sin(angle_rad), 0],
                        [np.sin(angle_rad), np.cos(angle_rad), 0],
                        [0, 0, 1]
                    ])

            # Calculate global position and rotation
            global_position = parent_global_pos + parent_global_rot @ local_offset
            global_rotation_matrix = parent_global_rot @ local_rotation_matrix
            
            current_joint_positions[joint_name] = global_position.tolist()

            # Recurse for children
            for child_joint in joint_info["children"]:
                calculate_global_position(child_joint, global_position, global_rotation_matrix)
            
            # Add End Site position if exists
            if joint_info["end_site"] is not None:
                end_site_global_pos = global_position + global_rotation_matrix @ joint_info["end_site"]
                current_joint_positions[f"{joint_name}_EndSite"] = end_site_global_pos.tolist()

        # Start recursion from the root joint (assuming 'Hips' is always root)
        channel_idx = 0 # Reset channel index for each frame
        calculate_global_position("Hips") # Assuming Hips is the root
        all_frame_joint_positions.append(current_joint_positions)

    base_filename_with_person = os.path.basename(bvh_file_path).replace('.bvh', '')
    # Extract the part before '_person' and the '_personX' part
    parts = base_filename_with_person.split('_person')
    if len(parts) > 1:
        main_part = parts[0]
        person_part = '_person' + parts[1]
        # Remove the _lightweight_smoothed_3d_keypoints suffix from the main part
        if main_part.endswith('_lightweight_smoothed_3d_keypoints'):
            main_part = main_part.replace('_lightweight_smoothed_3d_keypoints', '')
        base_filename = main_part + person_part
    else:
        base_filename = base_filename_with_person # Fallback if _person not found

    output_filename = os.path.join(output_dir, f'{base_filename}_parsed_positions.json')
    with open(output_filename, 'w') as f:
        json.dump({"frames": all_frame_joint_positions, "hierarchy": {k: {key: val.tolist() if isinstance(val, np.ndarray) else val for key, val in v.items()} for k, v in hierarchy.items()}}, f, indent=4)
    
    print(f"BVH parsed and joint positions saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse BVH file and extract joint positions.")
    parser.add_argument('--bvh_path', type=str, required=True, help="Path to the input BVH file.")
    parser.add_argument('--output_dir', type=str, default="/mnt/d/progress/ani_bender/output_data", help="Directory to save the parsed JSON file.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    parse_bvh(args.bvh_path, args.output_dir)