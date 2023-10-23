
Python Detection Pipeline
=========================

Overview
========
    This is a python implementation of TAPPAS detection pipeline using Yolov5m.
    For more information see `the TAPPAS documentation. <https://github.com/hailo-ai/tappas/tree/master/apps/h8/gstreamer/general/detection#detection-pipeline>`_
    
    It uses resources from the TAPPAS installation and should be pointed to the TAPPAS installation directory.
    You should set the TAPPAS_WORKSPACE environment variable to yout TAPPAS installation directory.
    It uses Hailo Python bindings and should be run from Hailo virtualenv. (or TAPPAS docker)
    It is built to work with the Raspberry Pi intgration of TAPPAS and Hailo.

Requirements
============
- TAPPAS environment
   - TAPPAS Docker (tested on TAPPAS 3.25.0 and 3.26.0)
   - Halio Suite Docker (tested on hailo_sw_suite_2023-07, and hailo_sw_suite_2023-10)
   - Note that if you are using TAPPAS 3.25.0 or hailo_sw_suite_2023-07 you should remove the remark on the 'thresholds_str=""' line in the detection.py file. see instructions in the file.
- Hailo device


App structure
=============
The application includes 2 files.

detection.py
------------
The first is the main file that runs the pipeline - ``detection.py``.

You should not edit this file unless you know what you are doing.

callback_template.py
--------------------
The second file is the user's callback file - ``callback_template.py``.

This file is called by the hailopython plugin and should be edited by the user.

The "run" function will be called for every frame that is processed by the pipeline.

From this function you can access the frame data and the detection results.

See the comments in the file for more details.

Running the application
=======================
To run the pipeline, you should run the following command:

.. code-block:: sh

    python3 detection.py

You can see additional options by running:

.. code-block:: sh

   python3 detection.py --help
   usage: detection.py [-h] [--input INPUT] [--python-module PYTHON_MODULE] [--show-fps] [--disable-sync] [--dump-dot]

    Detection App

    options:
    -h, --help            show this help message and exit
    --input INPUT, -i INPUT
                            Input source. Can be a file, USB or MIPI camera
    --python-module PYTHON_MODULE, -py PYTHON_MODULE
                            Python module with your callback function
    --show-fps, -f        Print FPS on sink
    --disable-sync        Disables display sink sync, will run as fast possible.
    --dump-dot            Dump the pipeline graph to a dot file pipeline.dot

The application will run until you press Ctrl+C or the video file ends.

