# 영상 기반 3D 아바타 애니메이션 시스템 구축 - 상세 진행 계획

## **현재까지의 진행 상황**

저희는 영상 기반 3D 아바타 애니메이션 시스템의 핵심 파이프라인을 구축했습니다.

*   **환경 설정:** Python 가상 환경 및 필요한 모든 라이브러리(`ultralytics`, `mediapipe`, `yt-dlp`, `opencv-python`, `numpy` 등) 설치를 완료했습니다.
*   **동영상 획득:** `download_youtube_video.py` 스크립트를 통해 유튜브 동영상을 `H.264` 코덱의 `mp4` 파일로 다운로드하는 기능을 구현했습니다.
*   **포즈 추정 백엔드:**
    *   **YOLO-pose 기반:** 다중 인물 2D 포즈 추정 및 BVH 애니메이션 생성 파이프라인을 구축했습니다. (현재는 MediaPipe 기반으로 전환 중)
    *   **MediaPipe Pose 기반:** MediaPipe의 내장 3D 포즈 추정 기능을 활용하여 3D 키포인트를 직접 추출하고, 이를 BVH 애니메이션으로 변환하는 파이프라인을 구축했습니다. MediaPipe의 스무딩 기능과 3D 데이터 덕분에 애니메이션 품질이 크게 개선될 것으로 기대됩니다.
*   **BVH 변환 및 시각화:**
    *   MediaPipe 3D 키포인트 데이터를 BVH 파일로 변환하는 로직을 개선했습니다.
    *   생성된 BVH 파일을 파싱하여 3D 골격 애니메이션을 이미지 프레임으로 시각화하고, 이를 동영상 파일로 만드는 기능을 구현했습니다.
    *   시각화 시 카메라 시점을 개선하여 애니메이션을 더 잘 관찰할 수 있도록 조정했습니다.
*   **자동화:** `run_full_pipeline.py` 스크립트와 `run_pipeline.bat` 파일을 통해 전체 파이프라인을 자동으로 실행할 수 있도록 설정했습니다.

## **사용자님께서 깨어나신 후 진행해야 할 작업**

가장 먼저, `run_pipeline.bat` 파일을 실행하여 파이프라인이 성공적으로 작동하는지 확인해야 합니다.

### **1. `run_pipeline.bat` 실행 및 결과 확인**

**실행 방법:**

1.  **동영상 파일 준비:** `D:\progress\ani_bender\input_videos\` 디렉토리에 사용할 동영상 파일(예: `videoplayback.mp4`)이 있는지 확인하십시오.
    *   **중요:** 동영상 파일 이름에 특수 문자(예: 이모지)가 포함되어 있지 않은지 확인하십시오. 만약 있다면, `boxing_highlights.mp4`와 같이 간단한 영어 이름으로 변경하는 것을 권장합니다.
2.  **명령 프롬프트(CMD) 열기:** Windows 검색창에 `cmd`를 입력하여 명령 프롬프트를 엽니다.
3.  **프로젝트 디렉토리로 이동:** 다음 명령어를 입력하여 `ani_bender` 프로젝트 루트 디렉토리로 이동합니다.
    ```bash
    cd /d D:\progress\ani_bender
    ```
4.  **파이프라인 실행:** 다음 명령어를 입력하여 `run_pipeline.bat` 파일을 실행합니다. **반드시 동영상 파일의 전체 경로를 큰따옴표로 묶어 인자로 전달해야 합니다.**
    ```bash
    run_pipeline.bat "D:\progress\ani_bender\input_videos\videoplayback.mp4"
    ```
    (만약 동영상 파일 이름이 `boxing_highlights.mp4`라면: `run_pipeline.bat "D:\progress\ani_bender\input_videos\boxing_highlights.mp4"`)

**예상 결과:**

*   명령 프롬프트 창에 파이프라인의 각 단계(MediaPipe 포즈 추정, 스무딩, BVH 변환, 파싱, 시각화, 동영상 생성)가 순서대로 실행되는 로그가 출력될 것입니다.
*   오류 메시지 없이 모든 단계가 성공적으로 완료되어야 합니다.
*   `D:\progress\ani_bender\output_data\` 디렉토리에 다음과 같은 파일들이 생성될 것입니다:
    *   `[동영상파일명]_mediapipe_3d_keypoints.json`
    *   `[동영상파일명]_mediapipe_smoothed_3d_keypoints.json`
    *   `[동영상파일명]_mediapipe_person1.bvh` (MediaPipe는 단일 인물 중심이므로 기본적으로 1개만 생성)
    *   `[동영상파일명]_mediapipe_person1_parsed_positions.json`
    *   `frames_mediapipe_person1_3d\` (시각화 이미지 프레임 디렉토리)
    *   `bvh_animation_mediapipe_person1_3d.mp4` (최종 애니메이션 동영상)

### **2. 생성된 동영상 확인 및 피드백**

`D:\progress\ani_bender\output_data\bvh_animation_mediapipe_person1_3d.mp4` 파일을 재생하여 애니메이션을 확인하십시오.

**피드백 요청:**

*   **3D 애니메이션 품질:**
    *   MediaPipe의 3D 데이터와 개선된 시각화 설정 덕분에 애니메이션이 이전보다 얼마나 더 자연스럽고 부드러워졌는지 평가해 주십시오.
    *   특히, '사람이 걷는 것'처럼 보이는지, 좌표축 문제가 해결되었는지 확인해 주십시오.
*   **다중 인물 처리 (현재 상태):**
    *   이 동영상에는 MediaPipe의 특성상 한 명의 사람만 나타날 것입니다.

## **다음 단계 (사용자 피드백 기반)**

사용자님의 피드백에 따라 다음 단계를 진행할 것입니다.

### **시나리오 A: MediaPipe 3D 애니메이션 품질이 만족스러운 경우**

*   **다중 인물 동시 시각화 구현:**
    *   **목표:** YOLO-pose를 사용하여 다중 인물 2D 감지를 수행하고, 각 인물에 대해 MediaPipe 3D 포즈 추정(또는 유사한 3D 격상)을 적용한 후, 모든 인물을 하나의 동영상에 동시에 시각화합니다.
    *   **세부 계획:**
        1.  **YOLO 기반 2D 키포인트 재활용:** `run_pose_estimation.py` (YOLO)를 실행하여 다중 인물 2D 키포인트 데이터를 생성합니다.
        2.  **개별 인물 3D 격상 및 BVH 변환:** YOLO가 감지한 각 인물(person1, person2, ...)에 대해 MediaPipe 3D 포즈 추정 모델을 적용하여 3D 키포인트를 얻고, 이를 BVH로 변환합니다. (이 부분은 `run_full_pipeline.py`를 수정하여 YOLO 2D 출력을 MediaPipe 3D 모델로 처리하도록 변경해야 할 수 있습니다.)
        3.  **다중 인물 동시 시각화 스크립트 개발:** `visualize_bvh.py`를 수정하여 여러 파싱된 BVH JSON 파일(각 인물별)을 동시에 로드하고, 동일한 3D 플롯에 여러 골격을 그려 하나의 동영상으로 만듭니다. (이때 각 인물의 초기 위치를 조정하여 겹치지 않게 하는 로직이 필요할 수 있습니다.)
        4.  **최종 동영상 생성:** `create_video_from_frames.py`를 사용하여 최종 동영상을 만듭니다.

### **시나리오 B: MediaPipe 3D 애니메이션 품질이 여전히 만족스럽지 않은 경우**

*   **고급 2D-to-3D 모델 통합 재논의:**
    *   **목표:** MediaPipe보다 더 높은 품질의 3D 포즈 추정 결과를 제공할 수 있는 다른 고급 모델(예: `VideoPose3D`, `VIBE` 등)의 통합을 고려합니다.
    *   **세부 계획:** 해당 모델의 설치, 설정, 기존 파이프라인과의 연동 방안을 조사하고 사용자님께 제안합니다. 이는 상당한 시간과 노력이 필요할 수 있습니다.
