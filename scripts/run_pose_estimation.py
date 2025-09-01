import cv2
from ultralytics import YOLO
import json
import os
import argparse
import numpy as np

def chunk_video(video_length, chunk_size, overlap_size):
    """
    Calculates start and end frames for video chunks with overlap.
    Returns a list of (start_frame, end_frame) tuples.
    """
    chunks = []
    if chunk_size <= 0 or overlap_size < 0:
        raise ValueError("chunk_size must be positive and overlap_size non-negative.")

    if chunk_size <= overlap_size:
        print("Warning: chunk_size is less than or equal to overlap_size. This may lead to unexpected behavior.")

    current_frame = 0
    while current_frame < video_length:
        end_frame = min(current_frame + chunk_size, video_length)
        chunks.append((current_frame, end_frame))
        if end_frame == video_length:
            break
        current_frame = end_frame - overlap_size
        # Ensure current_frame doesn't go negative if overlap_size is too large for the first chunk
        if current_frame < 0: current_frame = 0
    return chunks

def run_pose_estimation(video_path, output_dir, chunk_size, overlap_size):
    """
    Runs YOLO-pose on a video to extract 2D keypoints in chunks and saves them to JSON files.
    """
    model = YOLO('yolov8n-pose.pt') # You can choose other YOLO-pose models like yolov8s-pose.pt, yolov8m-pose.pt

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_filename_base = os.path.basename(video_path).replace('.', '_')

    print(f"Processing video: {video_path} (Total frames: {video_length}, FPS: {video_fps})")

    # Determine chunks
    chunks = chunk_video(video_length, chunk_size, overlap_size)
    print(f"Video will be processed in {len(chunks)} chunks.")

    for chunk_idx, (start_frame, end_frame) in enumerate(chunks):
        print(f"\n--- Processing Chunk {chunk_idx + 1}/{len(chunks)} (Frames {start_frame}-{end_frame-1}) ---")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_data_chunk = []
        current_frame_in_chunk = start_frame

        while current_frame_in_chunk < end_frame:
            ret, frame = cap.read()
            if not ret:
                print(f"End of video or failed to read frame at index {current_frame_in_chunk} in chunk.")
                break

            # print(f"--- Processing Frame {current_frame_in_chunk} ---") # Too verbose
            results = model(frame, verbose=False) # verbose=False to suppress extensive output

            persons_data = []
            for r in results:
                if r.keypoints is not None and r.boxes is not None:
                    keypoints_tensor = r.keypoints.data
                    boxes_tensor = r.boxes.data

                    num_persons = min(len(keypoints_tensor), len(boxes_tensor))

                    for i in range(num_persons):
                        person_keypoints = keypoints_tensor[i].tolist() # [17, 3]
                        person_bbox = boxes_tensor[i][:5].tolist()

                        persons_data.append({
                            "bbox": person_bbox,
                            "keypoints": person_keypoints
                        })
            
            frame_data_chunk.append({
                "frame_idx": current_frame_in_chunk,
                "persons": persons_data
            })
            current_frame_in_chunk += 1

        output_filename_chunk = os.path.join(output_dir, f'{video_filename_base}_chunk{chunk_idx}_2d_keypoints.json')
        with open(output_filename_chunk, 'w') as f:
            json.dump(frame_data_chunk, f, indent=4)
        print(f"2D keypoints for chunk {chunk_idx} saved to {output_filename_chunk}")

    cap.release()
    print(f"\n2D pose estimation completed for all chunks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 2D pose keypoints from a video using YOLO-pose.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--output_dir', type=str, default="/mnt/d/progress/ani_bender/output_data", help="Directory to save the output JSON file.")
    parser.add_argument('--chunk_size', type=int, default=243, help="Number of frames to process in each chunk.")
    parser.add_argument('--overlap_size', type=int, default=121, help="Number of overlapping frames between chunks.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_pose_estimation(args.video_path, args.output_dir, args.chunk_size, args.overlap_size)