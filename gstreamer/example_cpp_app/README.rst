Example app
=============

| This example is a bit more complicated than example 1 and showcase more features you can implement with Gstreamer with C++ on top.
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
- Example code for adding a custom overlay to the video stream.
- Data aggregation and statistics calculation to be added as overlay on the video stream.
   - Includes, FPS, Hailo stats, and Host stats (memory and CPU usage).
   - Note the the memory usage represents the maximum resident set size (RSS) used by the process since the start of the process.
   - See getrusage() documentation for more information.
- Attach callback to element to get FPS on the element.
- Attach callback to element to print detections (--probe-example)
- To dump the pipeline to a dot file use (--dump-dot-files)
   - The dot files will be saved in the current directory.
   - There are two dot files, one before setting the pipeline to play and one after play state is reached.
   - To see the full pipeline after caps negotiation and auto-plugging use the dot file after play state is reached.
   - To view the dot files use the dot command from graphviz. You can display to screen or create a png file.
      - dot -Tpng -o pipeline_playing.png pipeline_playing.dot
      - dot -T x11 pipeline_playing.dot
      

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
