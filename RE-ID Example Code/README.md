# Hailo RE-ID example code

The folowwing is a sample code for using Hailo-8 chip to identify persons entering a room,
and the saving them in a database for further identification, in case a person comes back to the room.

Requirements for compiling and running the sample code:
C++ version ???
Haio-8 installed on the host, or used with M.2 PCIe board
HailoRT suite version 4.10.0 installed on the machine
xtensor library version ???
A directory called video_images, conaining png images in the form of imageX.png when X = numbers running from 1 and on, representing the video images to be processed

In order to compile the sample application, one should run "./build.sh"

After successful compilation, one should run "./build/x86_64/vstream_re_id_example.4.10.0 -hef=yolov5s_personface.hef -reid=repvgg_a0_person_reid_2048.hef -num=1"

