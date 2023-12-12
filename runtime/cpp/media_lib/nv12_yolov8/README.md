This is example application of using the Media Library, it works only for yolov8, receives data in 4K, NV12 format.

To build for hailo-15:

1 Install one of the latest releases for Hailo15(1.1.1/1.1.2)

2 From the prebuild/sdk folder, run this script: (it will create a cross compile env)poky-glibc-x86_64-core-image-minimal-armv8a-hailo15-evb-security-camera-toolchain-4.0.2.sh 

3 Clone from the media library from github   (or 1.1.1 version)

4 Make sure you have open cv installed, version 4.5.5

5 Copy the following files to their place:
hailo-media-library/api/src/detection/yolov8.cpp
hailo-media-library/api/src/detection/yolov8_postprocess.cpp
hailo-media-library/api/include/media_library/common (all the common directory)
hailo-media-library/api/include/media_library/open_source (all the open_source directory)
hailo-media-library/api/examples/vision_preproc_example.cpp
hailo-media-library/api/include/media_library/yolov8.hpp
hailo-media-library/api/include/media_library/yolov8_postprocess.hpp
hailo-media-library/api/meson.build

6 Change in this file:/hailo-media-library/media_library/src/hailo_encoder/encoder.cpp:line 83 from “8” to “50”.

7 Change dir to  “hailo-media-library”

8 Enable your cross compile env:. /opt/poky/4.0.2/environment-setup-armv8a-poky-linux 

9 Run “meson build”

10 Run ninja -C build

11 Copy the compiled files to the Hailo device
scp build/api/vision_preproc_example {username}@{ip}:/home/root/apps/media_library/
scp build/api/libdetection.so.0 {username}@{ip}:/home/root/apps/media_library/
scp api/examples/preproc_config_example.json username}@{ip}:/usr/bin
scp api/examples/encoder_config_example.json {username}@{ip}:/usr/bin
scp build/media_library/libhailo_encoder.so.0 {username}@{ip}:/usr/lib/
scp build/media_library/libhailo_media_library_encoder.so.0 {username}@{ip}:/usr/lib/

12 Copy the hef file:scp yolov8n.hef {username}@{ip}:/home/root/apps/media_library/

13 Run view_media_lib_example_app.sh

14 ssh to your hailo device

15 cd apps/media_library

16 run the app: ./vision_preproc_example

