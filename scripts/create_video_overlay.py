
import cv2
import numpy as np
import argparse
import os
from itertools import cycle

def create_video_overlay(video_path, keypoints_files, output_path):
    """
    Creates a video with 3D skeletons overlaid on the original video.
    """
    # Define the skeleton structure for H36M (17 joints)
    # This defines which joint is connected to which parent joint.
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

    # Load all keypoints data
    all_person_keypoints = [np.load(f) for f in keypoints_files]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Define colors for different skeletons
    colors = cycle([(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for person_idx, person_keypoints in enumerate(all_person_keypoints):
            if frame_idx < len(person_keypoints):
                kps = person_keypoints[frame_idx]
                color = next(colors)

                # --- Simple Orthographic Projection ---
                # We are in camera space, so we can roughly project by scaling and offsetting.
                # A proper projection would require camera intrinsics.
                # This is a simplified approach for visualization.
                proj_kps = (kps[:, :2] * np.array([width, height]) / 2) + np.array([width, height]) / 2
                proj_kps = proj_kps.astype(int)

                # Draw joints
                for i in range(kps.shape[0]):
                    cv2.circle(frame, tuple(proj_kps[i]), 5, color, -1)

                # Draw bones
                for i, parent in enumerate(parents):
                    if parent != -1:
                        cv2.line(frame, tuple(proj_kps[i]), tuple(proj_kps[parent]), color, 2)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print(f"Successfully created overlay video at {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--keypoints', nargs='+', required=True, help='List of .npy files for each person.')
    parser.add_argument('--output', type=str, required=True, help='Path for the output video file.')
    args = parser.parse_args()

    create_video_overlay(args.video, args.keypoints, args.output)
