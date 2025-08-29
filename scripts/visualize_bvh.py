import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import mediapipe as mp

mp_pose = mp.solutions.pose

# Define the connections for the simplified skeleton for visualization
# These connect the BVH bone names (or their corresponding keypoints)
# This needs to match the hierarchy defined in convert_to_bvh.py
SKELETON_CONNECTIONS = [
    ("Hips", "Spine"),
    ("Spine", "Neck"),
    ("Neck", "Head"),
    ("Spine", "LeftShoulder"),
    ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Spine", "RightShoulder"),
    ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"),
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"),
]

def visualize_bvh(parsed_json_path, output_frames_dir):
    with open(parsed_json_path, 'r') as f:
        data = json.load(f)
    
    frames_data = data["frames"]
    hierarchy = data["hierarchy"]

    if not frames_data:
        print("Parsed JSON contains no frame data. Cannot visualize.")
        return

    os.makedirs(output_frames_dir, exist_ok=True)

    # Determine plot limits dynamically
    all_coords = []
    for frame in frames_data:
        for joint_pos in frame.values():
            all_coords.append(joint_pos)
    all_coords = np.array(all_coords)

    if all_coords.size == 0:
        print("No coordinates found for plotting.")
        return

    min_x, max_x = all_coords[:, 0].min(), all_coords[:, 0].max()
    min_y, max_y = all_coords[:, 1].min(), all_coords[:, 1].max()
    min_z, max_z = all_coords[:, 2].min(), all_coords[:, 2].max()

    # Add more padding for a wider view
    padding_factor = 1.5 # Increased padding
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z

    max_dim_range = max(range_x, range_y, range_z)
    
    # Center the view and expand limits based on the largest dimension
    mid_x = (max_x + min_x) / 2
    mid_y = (max_y + min_y) / 2
    mid_z = (max_z + min_z) / 2

    plot_limits = [
        [mid_x - max_dim_range * padding_factor / 2, mid_x + max_dim_range * padding_factor / 2],
        [mid_y - max_dim_range * padding_factor / 2, mid_y + max_dim_range * padding_factor / 2],
        [mid_z - max_dim_range * padding_factor / 2, mid_z + max_dim_range * padding_factor / 2]
    ]

    print(f"Generating {len(frames_data)} visualization frames...")

    for i, frame_joint_positions in enumerate(frames_data):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set plot limits
        ax.set_xlim(plot_limits[0])
        ax.set_ylim(plot_limits[1])
        ax.set_zlim(plot_limits[2])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {i}')

        # Set initial camera angle for better view
        # elev: elevation angle (up/down), azim: azimuth angle (left/right)
        ax.view_init(elev=15, azim=-100) # Slightly higher and rotated for a better general view

        # Plot joints
        joint_coords = np.array(list(frame_joint_positions.values()))
        ax.scatter(joint_coords[:, 0], joint_coords[:, 1], joint_coords[:, 2], c='blue', marker='o')

        # Plot connections
        for connection in SKELETON_CONNECTIONS:
            joint1_name, joint2_name = connection
            if joint1_name in frame_joint_positions and joint2_name in frame_joint_positions:
                p1 = np.array(frame_joint_positions[joint1_name])
                p2 = np.array(frame_joint_positions[joint2_name])
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='red')
            # Handle EndSites if they are part of connections
            elif f"{joint1_name}_EndSite" in frame_joint_positions and joint2_name in frame_joint_positions:
                p1 = np.array(frame_joint_positions[f"{joint1_name}_EndSite"])
                p2 = np.array(frame_joint_positions[joint2_name])
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='red', linestyle='--')
            elif joint1_name in frame_joint_positions and f"{joint2_name}_EndSite" in frame_joint_positions:
                p1 = np.array(frame_joint_positions[joint1_name])
                p2 = np.array(frame_joint_positions[f"{joint2_name}_EndSite"])
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='red', linestyle='--')

        # Save frame
        frame_filename = os.path.join(output_frames_dir, f'frame_{i:05d}.png')
        plt.savefig(frame_filename)
        plt.close(fig)

    print(f"Visualization frames saved to {output_frames_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize BVH joint positions and save frames.")
    parser.add_argument('--parsed_json_path', type=str, required=True, help="Path to the parsed BVH JSON file.")
    parser.add_argument('--output_frames_dir', type=str, default="/mnt/d/progress/ani_bender/output_data/frames", help="Directory to save the output image frames.")

    args = parser.parse_args()

    os.makedirs(args.output_frames_dir, exist_ok=True)

    visualize_bvh(args.parsed_json_path, args.output_frames_dir)