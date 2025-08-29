@echo off
REM Change to the directory where this .bat file is located
cd /d "%~dp0"

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Check if a video path is provided as an argument
IF "%~1"=="" (
    echo.
    echo Error: Please provide the full path to the video file as an argument.
    echo Example: run_pipeline.bat "D:\path\to\your\video.mp4"
    echo.
    pause
    exit /b 1
)

REM The video path is the first argument passed to the bat file
set "VIDEO_PATH=%~1"

REM Run the full Python pipeline
python run_full_pipeline.py --video_path "%VIDEO_PATH%"

REM Deactivate the virtual environment (optional)
call venv\Scripts\deactivate.bat

echo.
echo Full pipeline execution finished.
pause