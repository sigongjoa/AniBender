
import cv2
import os
import argparse

def create_video_from_frames(input_frames_dir, output_video_path, fps=30):
    """
    Creates a video from a sequence of image frames.
    """
    images = [img for img in os.listdir(input_frames_dir) if img.endswith(".png")]
    images.sort() # Ensure frames are in correct order

    if not images:
        print(f"No PNG images found in {input_frames_dir}")
        return

    # Read the first image to get dimensions
    first_image_path = os.path.join(input_frames_dir, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error: Could not read the first image {first_image_path}")
        return

    height, width, layers = frame.shape
    size = (width, height)

    # Define the codec and create VideoWriter object
    # Use XVID for broader compatibility, or MP4V for .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        return

    print(f"Creating video from {len(images)} frames...")
    for i, image_name in enumerate(images):
        img_path = os.path.join(input_frames_dir, image_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
        out.write(img)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(images)} frames.")

    out.release()
    print(f"Video successfully created at {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video from image frames.")
    parser.add_argument('--input_frames_dir', type=str, default="/mnt/d/progress/ani_bender/output_data/frames", help="Directory containing the image frames.")
    parser.add_argument('--output_video_path', type=str, default="/mnt/d/progress/ani_bender/output_data/bvh_animation.mp4", help="Path for the output video file.")
    parser.add_argument('--fps', type=int, default=30, help="Frames per second for the output video.")

    args = parser.parse_args()

    create_video_from_frames(args.input_frames_dir, args.output_video_path, args.fps)
