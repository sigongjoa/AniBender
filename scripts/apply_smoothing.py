import json
import os
import numpy as np
import argparse
# from one_euro_filter import OneEuroFilter # Uncomment if you install one-euro-filter

def apply_smoothing(input_json_path, output_dir):
    """
    Applies temporal smoothing to 3D keypoint data.
    This is a placeholder for a more sophisticated method like SmoothNet.
    For demonstration, it currently does no actual smoothing.
    Handles multiple people per frame.
    """
    with open(input_json_path, 'r') as f:
        data_3d = json.load(f)

    smoothed_data_3d = []
    
    # Placeholder for actual smoothing logic
    # In a real implementation, you would iterate through keypoints over time
    # and apply a filter (e.g., OneEuroFilter, or a learned temporal model).
    # For now, it just copies the data.
    for frame_data in data_3d:
        # Iterate through each person in the frame
        smoothed_person_keypoints_list = []
        for person_keypoints_3d in frame_data["keypoints"]:
            # Apply smoothing to this person's keypoints
            # Currently, just copying the data
            smoothed_person_keypoints_list.append(person_keypoints_3d)
        
        smoothed_data_3d.append({
            "keypoints": smoothed_person_keypoints_list
        })

    output_filename = os.path.join(output_dir, os.path.basename(input_json_path).replace('_3d_keypoints.json', '_smoothed_3d_keypoints.json'))
    with open(output_filename, 'w') as f:
        json.dump(smoothed_data_3d, f, indent=4)
    
    print(f"3D keypoints smoothed (placeholder) and saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply temporal smoothing to 3D pose keypoints.")
    parser.add_argument('--input_json_path', type=str, required=True, help="Path to the input 3D keypoints JSON file.")
    parser.add_argument('--output_dir', type=str, default="/mnt/d/progress/ani_bender/output_data", help="Directory to save the output smoothed 3D keypoints JSON file.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    apply_smoothing(args.input_json_path, args.output_dir)