import json
import os
import numpy as np
import argparse
from smoothing_utils import moving_average_filter, savgol_filtering, OneEuroFilter

def apply_filter(frames, method="moving_average", window_size=5, window_length=11, polyorder=2):
    smoothed = []
    if method == "moving_average":
        # Ensure frames is a list of dictionaries, and keypoints are processed correctly
        keypoints_array = np.array([f["keypoints"] for f in frames])
        smoothed_keypoints_array = moving_average_filter(keypoints_array, window_size=window_size)
        
        for i, frame in enumerate(frames):
            smoothed.append({
                "frame_idx": frame["frame_idx"],
                "keypoints": smoothed_keypoints_array[i].tolist()
            })
    elif method == "savgol":
        keypoints_array = np.array([f["keypoints"] for f in frames])
        smoothed_keypoints_array = savgol_filtering(keypoints_array, window_length=window_length, polyorder=polyorder)
        
        for i, frame in enumerate(frames):
            smoothed.append({
                "frame_idx": frame["frame_idx"],
                "keypoints": smoothed_keypoints_array[i].tolist()
            })
    elif method == "one_euro":
        # OneEuroFilter needs to be applied per person and per coordinate
        # It's stateful, so a new instance per person/coordinate might be needed for true independence
        # For simplicity, applying it to the entire keypoints array for now.
        # This might not be the ideal way to use OneEuroFilter for multi-person/multi-joint data.
        # A more robust implementation would iterate through each person's keypoints over time.
        
        # Initialize OneEuroFilter for each joint/coordinate if needed, or process sequentially.
        # For now, let's assume a single filter instance processes the entire sequence of keypoints for all people.
        
        # Reshape to (num_frames, num_people * num_joints * num_coords) for filtering
        original_shape = np.array(frames[0]["keypoints"]).shape # (num_people, num_joints, num_coords)
        num_people = original_shape[0]
        num_joints = original_shape[1]
        num_coords = original_shape[2]

        # Flatten keypoints for filtering
        flattened_keypoints = np.array([np.array(f["keypoints"]).flatten() for f in frames])
        
        euro_filters = []
        for _ in range(flattened_keypoints.shape[1]): # For each flattened coordinate
            euro_filters.append(OneEuroFilter(freq=30, min_cutoff=1.0, beta=0.01))

        smoothed_flattened_keypoints = np.zeros_like(flattened_keypoints)
        for i in range(flattened_keypoints.shape[0]): # Iterate over frames
            for j in range(flattened_keypoints.shape[1]): # Iterate over flattened coordinates
                smoothed_flattened_keypoints[i, j] = euro_filters[j](flattened_keypoints[i, j])

        for i, frame in enumerate(frames):
            # Reshape back to original (num_people, num_joints, num_coords)
            reshaped_keypoints = smoothed_flattened_keypoints[i].reshape(num_people, num_joints, num_coords)
            smoothed.append({
                "frame_idx": frame["frame_idx"],
                "keypoints": reshaped_keypoints.tolist()
            })
    else:
        return frames  # no filtering
    return smoothed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply temporal smoothing to 3D pose keypoints.")
    parser.add_argument('--input_json_path', type=str, required=True, help="Path to the input 3D keypoints JSON file.")
    parser.add_argument('--output_dir', type=str, default="/mnt/d/progress/ani_bender/output_data", help="Directory to save the output smoothed 3D keypoints JSON file.")
    parser.add_argument('--method', type=str, default="moving_average",
                        choices=["moving_average", "savgol", "one_euro", "none"], help="Smoothing method to apply.")
    parser.add_argument('--window_size', type=int, default=5, help="Window size for moving_average filter.")
    parser.add_argument('--window_length', type=int, default=11, help="Window length for savgol filter.")
    parser.add_argument('--polyorder', type=int, default=2, help="Polynomial order for savgol filter.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the input JSON data
    with open(args.input_json_path, "r") as f:
        frames_data = json.load(f)

    # Apply the selected filter
    smoothed_frames = apply_filter(frames_data, method=args.method,
                                   window_size=args.window_size,
                                   window_length=args.window_length,
                                   polyorder=args.polyorder)

    # Determine output filename
    input_filename_base = os.path.basename(args.input_json_path).replace('_3d_keypoints.json', '')
    output_filename = os.path.join(args.output_dir, f'{input_filename_base}_smoothed_3d_keypoints.json')

    # Save the smoothed data
    with open(output_filename, 'w') as f:
        json.dump(smoothed_frames, f, indent=4)
    
    print(f"3D keypoints smoothed using {args.method} method and saved to {output_filename}")
