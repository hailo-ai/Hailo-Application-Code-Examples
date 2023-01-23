@echo off

set download_done=false
set hef_file=yolov5m.hef
set mp4_file=full_mov_slow.mp4

if not exist hefs (
    mkdir hefs
)
cd hefs
if exist %hef_file% (
    echo "Info: %hef_file% already exists in the hefs directory, skipping download"
) else (
    curl -o %hef_file% https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.6.0/%hef_file% 2>&1 | findstr /C:"AccessDenied"
    if %errorlevel% == 0 (
        echo "Error: Access denied while trying to download %hef_file%"
        exit /b
    ) else if %errorlevel% neq 0 (
        echo "Error: Failed to download %hef_file%"
        exit /b
    ) else (
        set download_done=true
    )
)

cd ..
if not exist videos (
    mkdir videos
)
cd videos
if exist %mp4_file% (
    echo "Info: %mp4_file% already exists in the videos directory, skipping download"
) else (
    curl -o %mp4_file% https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.6.0/%mp4_file% 2>&1 | findstr /C:"AccessDenied"
    if %errorlevel% == 0 (
        echo "Error: Access denied while trying to download %mp4_file%"
        exit /b
    ) else if %errorlevel% neq 0 (
        echo "Error: Failed to download %mp4_file%"
        exit /b
    ) else (
        set download_done=true
    )
)

if %download_done% == true (
    echo "Resources successfully downloaded"
) else (
    echo "All files already exist, no new files were downloaded"
)