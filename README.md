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

#### a) Run YOLO + VideoPose3D Pipeline

```bash
venv/bin/python run_yolo_videopose3d_pipeline.py --video_path "input_videos/your_video.mp4" --output_base_dir output_data
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

## Future Work & Improvements

-   **Unified Pipeline**: Integrate both YOLO+VideoPose3D and VIBE into a single, more flexible pipeline with options to choose the desired 3D estimation backend.
-   **Improved Multi-Person Handling**: Enhance tracking and 3D pose estimation for complex multi-person scenarios.
-   **Blender Integration**: Streamline the retargeting process in Blender for easier character animation.
-   **Performance Optimization**: Further optimize the pipeline for faster processing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
