**Last TAPPAS version checked - 3.24.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


Multistream example
=============

| This example code is showcasing a use case of running multiple streams on the same pipeline.
| The example code is using the Hailo plugins to run a full detection pipeline on each stream.
| The main changes employed here are:
| The RTSP and video sources are added to a source bin which is then added to the pipeline.
| The source bin can be seeked back or move to NULL state to reset the stream. While the main pipeline is still running.
| New hailoroundrobin implementation that allows to run arbitration on multiple streams. with the option to skip a stream after a timeout.
| Adding CPP on top allows for better control and debugging of the pipeline.

Requirements
============
- TAPPAS environment
   - TAPPAS Docker (tested on TAPPAS 3.24.0)
   - Halio Suite Docker (tested on hailo_sw_suite_2023-04)
- Hailo device

Patching the hailoroundrobin plugin
===================================
- backup current hailoroundrobin plugin
   - cp  /local/workspace/tappas/core/hailo/plugins/muxer/gsthailoroundrobin.cpp /local/workspace/tappas/core/hailo/plugins/muxer/gsthailoroundrobin.cpp.backup
   - cp /local/workspace/tappas/core/hailo/plugins/muxer/gsthailoroundrobin.hpp /local/workspace/tappas/core/hailo/plugins/muxer/gsthailoroundrobin.hpp.backup
   - cp /local/workspace/tappas/core/hailo/plugins/meson.build /local/workspace/tappas/core/hailo/plugins/meson.build.bkp
- copy new files from patch directory
   - cp patch/gsthailoroundrobin.cpp /local/workspace/tappas/core/hailo/plugins/muxer/gsthailoroundrobin.cpp
   - cp patch/gsthailoroundrobin.hpp /local/workspace/tappas/core/hailo/plugins/muxer/gsthailoroundrobin.hpp
   - cp patch/meson.build /local/workspace/tappas/core/hailo/plugins/meson.build
- build hailo plugins by running this script
   - /local/workspace/tappas/scripts/gstreamer/install_hailo_gstreamer.sh 
- NOTE: this may cause cause problems withe other TAPPAS examples using this plugin.

Building
========
| Copy the example to the TAPPAS environment.
| Make sure you got the TAPPAS_WORKSPACE environment variable set to the TAPPAS workspace.
| Run the install script ./install.sh
| The install script will build the app and install it in the app directory.
| The install script is using meson to build the app.
| To compile with debug symbols run ./install.sh debug
| Note that this script is compiling only the code in the app directory. 
| To debug code from TAPPAS you should compile TAPPAS with debug symbols (see TAPPAS documentation). 
| The meson script is using /opt/hailo/tappas/pkgconfig/hailo_tappas.pc to find the TAPPAS libraries and so files.

Configuring
===========
| set youe RTSP source in multistream_app.cpp 
| const std::string RTSP_SRC_0 = ....
| const std::string RTSP_SRC_1 = ....
| the rtsp_user and rtsp_pass can be passed as argument to src_bin consturctor. Or is all are the same can be set as default in SrcBin.hpp constructor definition.
| currently set to const std::string& rtsp_user = "root", const std::string& rtsp_pass = "hailo"

Running
=======
| Run ./multistream_app -n 2  --rtsp-src --gst-debug=*debug:4
| The application code and the src_bin code got their own debug categories. app_debug and src_bin_debug.
| You can set them from cli by using --gst-debug=app_debug:4,src_bin_debug:4
| You enable fps and timestaps debug by using the --fps-probe and --pts-probe optins.
