import joblib
import os
import numpy as np
import argparse
import json

def process_vibe_output(vibe_pkl_path, output_dir, video_filename_base):
    """
    Loads VIBE's output pickle file, extracts 3D joint positions, and saves them to a JSON file.
    """
    vibe_results = joblib.load(vibe_pkl_path)

    # VIBE output is a dictionary where keys are person_ids
    # Each person_id contains a dictionary with 'joints3d' (numpy array of shape (num_frames, num_joints, 3))

    # We need to convert this to a JSON format similar to what uplift_to_3d.py produces:
    # [
    #   { "frame_idx": 0, "keypoints": [ [ [x,y,z,visibility], ... ], [ [x,y,z,visibility], ... ] ] },
    #   { "frame_idx": 1, "keypoints": [ [ [x,y,z,visibility], ... ], [ [x,y,z,visibility], ... ] ] },
    #   ...
    # ]
    # Where each inner list corresponds to a person.

    # Determine the total number of frames from any person's data
    num_frames = 0
    if vibe_results:
        # Get the first person's data to determine number of frames
        first_person_id = list(vibe_results.keys())[0]
        num_frames = vibe_results[first_person_id]['joints3d'].shape[0]

    json_output_data = []

    for frame_idx in range(num_frames):
        current_frame_all_person_keypoints = []
        for person_id in vibe_results.keys():
            person_joints3d = vibe_results[person_id]['joints3d']
            
            if frame_idx < person_joints3d.shape[0]:
                # Extract 3D joints for the current frame and person
                # Shape is (num_joints, 3)
                frame_keypoints = person_joints3d[frame_idx]
                
                # Convert to list and add dummy visibility (1.0)
                person_keypoints_with_visibility = [[float(kp[0]), float(kp[1]), float(kp[2]), 1.0] for kp in frame_keypoints]
                current_frame_all_person_keypoints.append(person_keypoints_with_visibility)
            # else:
                # If person data doesn't exist for this frame (e.g., person left scene),
                # we can append an empty list or a zero array for that person.
                # For now, we'll just skip if data is not available, meaning fewer persons in some frames.
                # This needs to be handled carefully by downstream scripts if they expect consistent person count.

        json_output_data.append({
            "frame_idx": frame_idx,
            "keypoints": current_frame_all_person_keypoints
        })

    output_json_path = os.path.join(output_dir, f'{video_filename_base}_vibe_3d_keypoints.json')
    with open(output_json_path, 'w') as f:
        json.dump(json_output_data, f, indent=4)
    
    print(f"3D keypoints from VIBE output saved to {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process VIBE output pickle file and extract 3D keypoints.")
    parser.add_argument('--vibe_pkl_path', type=str, required=True, help="Path to the VIBE output pickle file.")
    parser.add_argument('--output_dir', type=str, default="/mnt/d/progress/ani_bender/output_data", help="Directory to save the output JSON file.")
    parser.add_argument('--video_filename_base', type=str, required=True, help="Base filename of the video for consistent output naming.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    process_vibe_output(args.vibe_pkl_path, args.output_dir, args.video_filename_base)
