
Python Detection Pipeline
=========================

Overview
--------
This is a python implementation of TAPPAS detection pipeline.

It uses resources from the TAPPAS installation and should be pointed to the TAPPAS installation directory.

You should set the TAPPAS_WORKSPACE environment variable to yout TAPPAS installation directory.

It uses Hailo Python bindings and should be run from Hailo virtualenv. (or TAPPAS docker)

App structure
-------------
The application includes 2 files.

detection.py
++++++++++++
The first is the main file that runs the pipeline - ``detection.py``.

You should not edit this file unless you know what you are doing.

callback_template.py
++++++++++++++++++++
The second file is the user's callback file - ``callback_template.py``.

This file is called by the hailopython plugin and should be edited by the user.

The "run" function will be called for every frame that is processed by the pipeline.

From this function you can access the frame data and the detection results.

See the comments in the file for more details.

Running the application
-----------------------
To run the pipeline, you should run the following command:

.. code-block:: sh

    python3 detection.py

You can see additional options by running:

.. code-block:: sh

   python3 detection.py --help

This application supports input from file, USB camera and MIPI camera.

The application will run until you press Ctrl+C or the video file ends.

