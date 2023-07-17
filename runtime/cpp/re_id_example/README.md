# Hailo RE-ID example code

The folowing is a sample code for using Hailo-8 chip to identify persons entering a room,
and then saving them in a database for further identification, in case a person comes back to the room.

Requirements for compiling and running the sample code:
* C++20
* Haio-8 attached to the system
* HailoRT version 4.10.0
* xtensor library (submodule)
* xtl, required by xtensor library (submodule)
* rapidjson (submodule)

Bfore you begin, please run the script `get_frames_and_hefs.sh` to create a directory called video_images, containing png images.

In order to compile the sample application, one should run `./build.sh`

After a successful compilation, one should run `./build/x86_64/vstream_re_id_example -hef=yolov5s_personface.hef -reid=repvgg_a0_person_reid_2048.hef -num=1`

