**Last TAPPAS version checked - 3.25.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


====================================================
 License Plate Recognition (LPR) on multiple streams
====================================================

Overview
========

This GStreamer pipeline demonstrates License Plate Recognition (LPR) on multiple streams. This pipeline can work with VAAPI acceleration or CPU only. For optimal performance, it is recommended to use VAAPI acceleration.

Prerequisites
=============

- Hailo Suite installed on the host machine 
- Tested with Hailo July Suite (version 2023-07.1)
- Hailo-8 device connected via PCIe.
- If VAAPI is required, install it on the host machine.

Preparations
============

1. This application requires TAPPAS / HailoSuite installed with Ubuntu 22 (+ VAAPI optional). To use VAAPI, ensure you are able to run the VAAPI sanity pipeline. Refer to the details in ``tappas/apps/h8/gstreamer/x86_hw_accelerated/README.rst``. Note that the app uses auto-pluggers to select the best hardware acceleration available. To disable VAAPI (if installed), set the environment variable ``LIBVA_DRIVER_NAME=fakedriver``.
2. Copy this directory to your TAPPAS / HailoSuite docker.
3. Execute the install script using the command below:
   
   .. code-block:: bash

       $ ./install.sh
       # If you encounter errors, clean the build directory and run the install script again with the following:
       $ meson --wipe build.release
   
   You should now find a ``./multistream_lpr`` executable in the ``multistream_lpr`` library.

Instructions
============

1. Download the video files from S3. This will download 8 videos and create links to duplicate them.
2. Navigate to the resources/videos directory and run the script below to get video resources:

   .. code-block:: bash

       $ cd resources/videos 
       $ ./get_video_resources.sh

3. Download the required HEF files from S3. Navigate to the resources directory and run the script below to get HEF resources:

   .. code-block:: bash

       $ cd resources
       $ ./get_hef_resources.sh

4. Make sure you got the $TAPPAS_WORKSPACE environment variable set to the root of your TAPPAS / HailoSuite docker. For example: export TAPPAS_WORKSPACE=/home/...../hailo_sw_suite/artifacts/tappas
5. To run the demo, execute the command in the parent library. The default setting runs with 4 streams and no display:

   .. code-block:: bash

       $ ./multistream_lpr
   
   Use the help option to see all available options:

   .. code-block:: bash

       $ ./multistream_lpr -h
   
   To run with 8 streams and display, use:

   .. code-block:: bash

       $ ./multistream_lpr --num-of-inputs 8 --enable-display
   
   **Note**: Running with the display option reduces FPS. Each stream is 1080p and handling it requires a lot of resources. To get the best performance, run without the display option.
   
Enjoy ;)
