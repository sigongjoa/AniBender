
import cv2
from ultralytics import YOLO
import json
import os
import argparse

def run_pose_estimation(video_path, output_dir):
    """
    Runs YOLO-pose on a video to extract 2D keypoints and saves them to a JSON file.
    """
    model = YOLO('yolov8n-pose.pt') # You can choose other YOLO-pose models like yolov8s-pose.pt, yolov8m-pose.pt

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_data = []
    frame_idx = 0

    print(f"Processing video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video or failed to read frame at index {frame_idx}")
            break

        print(f"\n--- Processing Frame {frame_idx} ---")
        results = model(frame, verbose=False) # verbose=False to suppress extensive output
        print(f"Number of YOLO results for frame {frame_idx}: {len(results)}")

        persons_data = []
        for r in results:
            if r.keypoints is not None and r.boxes is not None:
                keypoints_tensor = r.keypoints.data
                boxes_tensor = r.boxes.data

                # Ensure the number of detected persons is consistent
                num_persons = min(len(keypoints_tensor), len(boxes_tensor))

                for i in range(num_persons):
                    person_keypoints = keypoints_tensor[i].tolist() # [17, 3]
                    
                    # Bbox is [x1, y1, x2, y2, conf, class]
                    # We need [x1, y1, x2, y2, conf]
                    person_bbox = boxes_tensor[i][:5].tolist()

                    persons_data.append({
                        "bbox": person_bbox,
                        "keypoints": person_keypoints
                    })
                    print(f"    Detected person {i+1} with {len(person_keypoints)} keypoints.")

        if not persons_data:
            print(f"No people detected in frame {frame_idx}.")
        else:
            print(f"Total people detected in frame {frame_idx}: {len(persons_data)}")

        frame_data.append({
            "frame_idx": frame_idx,
            "persons": persons_data
        })
        frame_idx += 1

    cap.release()

    output_filename = os.path.join(output_dir, os.path.basename(video_path).replace('.', '_') + '_2d_keypoints.json')
    with open(output_filename, 'w') as f:
        json.dump(frame_data, f, indent=4)
    
    print(f"\n2D keypoints extracted and saved to {output_filename}")
    print(f"Total frames processed: {frame_idx}")
    print(f"Total frames with keypoints data: {sum(1 for f in frame_data if f['persons'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 2D pose keypoints from a video using YOLO-pose.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--output_dir', type=str, default="/mnt/d/progress/ani_bender/output_data", help="Directory to save the output JSON file.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_pose_estimation(args.video_path, args.output_dir)
