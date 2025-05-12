@echo off

set download_done=false
set hef_file=yolov8s_nms_on_hailo.hef

if not exist hefs (
    mkdir hefs
)
cd hefs
if exist %hef_file% (
    echo "Info: %hef_file% already exists in the hefs directory, skipping download"
) else (
    curl -o %hef_file% https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8/%hef_file% 2>&1 | findstr /C:"AccessDenied"
    
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


if %download_done% == true (
    echo "Resources successfully downloaded"
) else (
    echo "All files already exist, no new files were downloaded"
)