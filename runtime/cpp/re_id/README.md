# Hailo RE-ID example code

**Last HailoRT version checked - 4.16.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


The folowing is a sample code for using Hailo-8 chip to identify persons entering a room,
and then saving them in a database for further identification, in case a person comes back to the room.

Requirements for compiling and running the sample code:
* C++20
* Haio-8 attached to the system
* HailoRT version >= 4.15.0
* xtensor library (submodule)
* xtl, required by xtensor library (submodule)
* rapidjson (submodule)

Bfore you begin, please run the script `get_frames_and_hefs.sh` to create a directory called video_images, containing png images.

In order to compile the sample application, one should run `./build.sh`

After a successful compilation, one should run `./build/x86_64/vstream_re_id_example -hef=yolov5s_personface.hef -reid=repvgg_a0_person_reid_2048.hef -num=1`

