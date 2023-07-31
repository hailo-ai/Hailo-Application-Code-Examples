Simple app
===========

| This example is a bare bones app that shows how to use Gstreamer with C++ on top.
| It uses the TAPPAS environment to build the app and for the required Gstreamer plugins.

Features
========
- Instantiate a Gstreamer pipeline with Hailo plugins running full detection pipeline.
- Attaching a callback to the pipeline bus to receive messages from the pipeline.
- Example code for selecting input source (camera, video file, URI).
   - For camera input example /dev/video0
   - For video file input example file:///home/user/video.mp4 (URI file prefix file://)
   - For online URI example try http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4"
- Example code for getting Hailo statistics from the pipeline bus messages.


Requirements
============
- TAPPAS environment
   - TAPPAS Docker (tested on TAPPAS 3.24.0)
   - Halio Suite Docker (tested on hailo_sw_suite_2023-04)
- Hailo device

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

Running
=======
| Run ./example_app to start the app.
| Run ./example_app -h to see the available options.
| online URI example: ./example_app --sync-pipeline -i http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4
| local file example: ./example_app --sync-pipeline -i file:///local/workspace/tappas/apps/h8/gstreamer/resources/mp4/detection0.mp4