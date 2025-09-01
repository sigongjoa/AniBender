# AniBender: Video-to-Animation Pose Estimation Pipeline

## Project Overview

AniBender is a comprehensive pipeline designed to extract 3D human pose and motion data from videos and convert it into animatable BVH (Biovision Hierarchy) format. This project leverages state-of-the-art deep learning models for 2D and 3D pose estimation, providing flexible options for motion analysis and animation generation.

## Features

-   **Video Ingestion**: Download YouTube videos or process local video files.
-   **Multi-Model Support**: Utilizes different cutting-edge models for 2D and 3D pose estimation.
    -   **YOLOv8**: For robust 2D human detection and keypoint estimation.
    -   **VideoPose3D**: For lifting 2D keypoints to 3D space.
    -   **VIBE**: For direct 3D human pose and shape estimation from video.
-   **Motion Processing**: Includes steps for temporal smoothing of 3D keypoints.
-   **BVH Conversion**: Converts 3D pose data into standard BVH format, compatible with 3D animation software (e.g., Blender).
-   **Visualization**: Generates animated videos with 3D skeletons for easy review of the extracted motion.
-   **Multi-Person Handling**: Supports processing and generating BVH for multiple detected individuals in a video.

## Pipeline Architecture

This project currently provides **two distinct pipelines**, each focusing on a different core 3D pose estimation methodology:

1.  **YOLO + VideoPose3D Pipeline (`run_yolo_videopose3d_pipeline.py`)**:
    -   **Input**: Video file.
    -   **Process**: YOLOv8 performs 2D human pose detection. The detected 2D keypoints are then fed into VideoPose3D to predict 3D joint locations.
    -   **Output**: Smoothed 3D keypoints (JSON), BVH animation files, and visualization videos.

2.  **VIBE Pipeline (`run_vibe_pipeline.py`)**:
    -   **Input**: Video file.
    -   **Process**: VIBE directly estimates 3D human pose and shape parameters from the input video.
    -   **Output**: Smoothed 3D keypoints (JSON), BVH animation files, and visualization videos.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sigongjoa/AniBender.git
cd AniBender
```

### 2. Create and Activate Virtual Environment

It is highly recommended to use a Python virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages. Note that some dependencies are installed from Git repositories.

```bash
pip install -r models/VIBE/requirements.txt
pip install ultralytics # For YOLOv8
```

### 4. Prepare VIBE Data

VIBE requires additional model weights and data. This script downloads them. Ensure `gdown` and `unzip` are installed on your system (e.g., `sudo apt install gdown unzip` on Ubuntu/Debian).

```bash
bash models/VIBE/scripts/prepare_data.sh
```

## Usage

### 1. Download a Video (Optional)

You can download a YouTube video using the provided script:

```bash
venv/bin/python scripts/download_youtube_video.py --url "YOUR_YOUTUBE_URL" --output_dir input_videos
```

Or place your video file directly into the `input_videos/` directory.

### 2. Run a Pipeline

Choose one of the pipelines below to process your video.

#### a) Run YOLO + VideoPose3d Pipeline

To run the pipeline with a sample video:

```bash
/mnt/d/progress/ani_bender/venv/bin/python run_yolo_videopose3d_pipeline.py --video_path sample_video.mp4
```

To run with your own video:

```bash
/mnt/d/progress/ani_bender/venv/bin/python run_yolo_videopose3d_pipeline.py --video_path "input_videos/your_video.mp4" --output_base_dir output_data
```

#### b) Run VIBE Pipeline

```bash
venv/bin/python run_vibe_pipeline.py --video_path "input_videos/your_video.mp4" --output_base_dir output_data
```

Replace `"input_videos/your_video.mp4"` with the actual path to your video file.

## Output

Processed results, including 3D keypoint JSON files, BVH animation files, and visualization videos, will be saved in the `output_data/` directory.

Example output files for a video named `my_video.mp4`:

-   `output_data/my_video_mp4_2d_keypoints.json` (YOLO output)
-   `output_data/my_video_mp4_videopose3d_3d_keypoints.json` (VideoPose3D 3D output)
-   `output_data/my_video_mp4_vibe_3d_keypoints.json` (VIBE 3D output)
-   `output_data/my_video_mp4_person1.bvh` (BVH for first person)
-   `output_data/bvh_animation_videopose3d_person1_3d.mp4` (Visualization video for VideoPose3D pipeline)
-   `output_data/bvh_animation_vibe_person1_3d.mp4` (Visualization video for VIBE pipeline)

## Advanced Usage and Debugging

Beyond the main pipelines, several scripts can be used for debugging and manual processing.

### 1. Visualization Pipeline

To debug the output of the 3D pose estimation, a dedicated visualization pipeline can be run. This pipeline does not generate a BVH file but instead produces video overlays and data logs.

```bash
python3 run_full_pipeline.py --video "input_videos/your_video.mp4"
```

This will generate the following files in `output_data/`:
-   `video_2d_overlay.mp4`: The original video with the 2D skeleton drawn on top.
-   `video_3d_overlay.mp4`: The original video with the projected 3D skeleton drawn on top.
-   `2d_keypoints.txt`: A frame-by-frame log of the 2D keypoint coordinates.
-   `3d_keypoints.txt`: A frame-by-frame log of the 3D keypoint coordinates.

### 2. Manual BVH Conversion

If you have a JSON file containing 3D keypoints from the `VideoPose3D` pipeline, you can manually convert it to BVH using the corrected conversion script:

```bash
python3 venv/bin/python scripts/convert_json_to_bvh_bvhio.py --input_json_path "output_data/your_video_videopose3d_smoothed_3d_keypoints.json" --output_dir "output_data"
```

This script uses the ground-truth skeleton definition from the VideoPose3D model to ensure an accurate conversion.

## Future Work & Improvements

-   **Unified Pipeline**: Integrate both YOLO+VideoPose3D and VIBE into a single, more flexible pipeline with options to choose the desired 3D estimation backend.
-   **Improved Multi-Person Handling**: Enhance tracking and 3D pose estimation for complex multi-person scenarios.
-   **Blender Integration**: Streamline the retargeting process in Blender for easier character animation.
-   **Performance Optimization**: Further optimize the pipeline for faster processing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.