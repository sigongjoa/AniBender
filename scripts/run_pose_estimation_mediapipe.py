
import cv2
import mediapipe as mp
import json
import os
import argparse

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def run_pose_estimation_mediapipe(video_path, output_dir, output_annotated_frames_dir=None):
    """
    Runs MediaPipe Pose on a video to extract 3D world keypoints and saves them to a JSON file.
    Also overlays the pose estimation on video frames and saves them.
    Focuses on single person tracking with MediaPipe's built-in smoothing.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_data = []
    frame_idx = 0

    print(f"Processing video with MediaPipe: {video_path}")

    if output_annotated_frames_dir:
        os.makedirs(output_annotated_frames_dir, exist_ok=True)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"End of video or failed to read frame at index {frame_idx}")
                break

            # print(f"\n--- Processing Frame {frame_idx} (MediaPipe) ---")
            
            # Convert the BGR image to RGB.
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            current_frame_keypoints = []
            if results.pose_landmarks and results.pose_world_landmarks: # Ensure world landmarks are also present
                # Draw the pose annotation on the image.
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # MediaPipe provides 33 pose landmarks. We need to map them to YOLO's 17 keypoints
                # This mapping is approximate and might need fine-tuning.
                # We will now extract 3D world landmarks (in meters) directly.
                
                # MediaPipe landmarks (subset for mapping):
                # NOSE (0), LEFT_EYE (2), RIGHT_EYE (5), LEFT_EAR (7), RIGHT_EAR (8)
                # LEFT_SHOULDER (11), RIGHT_SHOULDER (12)
                # LEFT_ELBOW (13), RIGHT_ELBOW (14)
                # LEFT_WRIST (15), RIGHT_WRIST (16)
                # LEFT_HIP (23), RIGHT_HIP (24)
                # LEFT_KNEE (25), RIGHT_KNEE (26)
                # LEFT_ANKLE (27), RIGHT_ANKLE (28)

                mp_to_yolo_map_indices = {
                    0: mp_pose.PoseLandmark.NOSE, # Nose
                    1: mp_pose.PoseLandmark.LEFT_EYE, # Left Eye
                    2: mp_pose.PoseLandmark.RIGHT_EYE, # Right Eye
                    3: mp_pose.PoseLandmark.LEFT_EAR, # Left Ear
                    4: mp_pose.PoseLandmark.RIGHT_EAR, # Right Ear
                    5: mp_pose.PoseLandmark.LEFT_SHOULDER, # Left Shoulder
                    6: mp_pose.PoseLandmark.RIGHT_SHOULDER, # Right Shoulder
                    7: mp_pose.PoseLandmark.LEFT_ELBOW, # Left Elbow
                    8: mp_pose.PoseLandmark.RIGHT_ELBOW, # Right Elbow
                    9: mp_pose.PoseLandmark.LEFT_WRIST, # Left Wrist
                    10: mp_pose.PoseLandmark.RIGHT_WRIST, # Right Wrist
                    11: mp_pose.PoseLandmark.LEFT_HIP, # Left Hip
                    12: mp_pose.PoseLandmark.RIGHT_HIP, # Right Hip
                    13: mp_pose.PoseLandmark.LEFT_KNEE, # Left Knee
                    14: mp_pose.PoseLandmark.RIGHT_KNEE, # Right Knee
                    15: mp_pose.PoseLandmark.LEFT_ANKLE, # Left Ankle
                    16: mp_pose.PoseLandmark.RIGHT_ANKLE, # Right Ankle
                }

                person_keypoints_list = []
                for i in range(17):
                    landmark = results.pose_world_landmarks.landmark[mp_to_yolo_map_indices[i]]
                    # MediaPipe provides x, y, z in meters, relative to hips center
                    # We also include visibility as confidence
                    person_keypoints_list.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
                current_frame_keypoints.append(person_keypoints_list)
                # print(f"    Detected person with {len(person_keypoints_list)} keypoints.")
            # else:
                # print(f"No pose landmarks detected in frame {frame_idx}.")

            # if not current_frame_keypoints:
                # print(f"No people detected in frame {frame_idx}.")
            # else:
                # print(f"Total people detected in frame {frame_idx}: {len(current_frame_keypoints)}")

            frame_data.append({
                "frame_idx": frame_idx,
                "keypoints": current_frame_keypoints
            })
            
            # Save annotated frame
            if output_annotated_frames_dir:
                frame_filename = os.path.join(output_annotated_frames_dir, f'frame_{frame_idx:05d}.png')
                cv2.imwrite(frame_filename, image)

            frame_idx += 1

    cap.release()

    output_filename = os.path.join(output_dir, os.path.basename(video_path).replace('.', '_') + '_mediapipe_3d_keypoints.json')
    with open(output_filename, 'w') as f:
        json.dump(frame_data, f, indent=4)
    
    print(f"\nMediaPipe 3D world keypoints extracted and saved to {output_filename}")
    print(f"Annotated frames saved to {output_annotated_frames_dir}")
    print(f"Total frames processed: {frame_idx}")
    print(f"Total frames with keypoints data: {sum(1 for f in frame_data if f['keypoints'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 3D world pose keypoints from a video using MediaPipe Pose.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--output_dir', type=str, default="/mnt/d/progress/ani_bender/output_data", help="Directory to save the output JSON file.")
    parser.add_argument('--output_annotated_frames_dir', type=str, default="/mnt/d/progress/ani_bender/output_data/annotated_frames_mediapipe", help="Directory to save annotated image frames.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_pose_estimation_mediapipe(args.video_path, args.output_dir, args.output_annotated_frames_dir)
