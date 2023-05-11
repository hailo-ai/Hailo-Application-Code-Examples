
TDA4VM Optimaized Pose Estimation Pipeline
==========================================

Overview
--------

``tda4_pose_estimation.sh`` demonstrates human pose estimation on a video device source.
 This is done by running a ``single-stream pose estimation pipeline`` on top of GStreamer using the Hailo-8 device and the TDA4VM DSP.

 This demo was tested using the following versions: 
 - TI 
	TDA4VM Yocto Dunfell, version ti-processor-sdk-linux-j7-evm-08_04_00_06
 - Hailo 
	Hailort and hailo_pci drivers version 4.10 
	Tappas version 3.21.0
 Model - centerpose_regnetx_1.6gf_fpn.
 Camera - Logitech BRIO 4K. 

Requirements
-------

 - TDA4VM 
 - Hailo device
 - TAPPAS environment

Options
-------

.. code-block:: sh

   ./tda4_pose_estimation.sh [--input FILL-ME]

    echo "TDA4 Pose Estimation pipeline usage:"

* 
  ``--input`` is an optional flag, Set the video source - a video device path (default is /dev/video0).

* 
  ``--show-fps``  is an optional flag that enables printing FPS on screen.

* ``--print-gst-launch`` is a flag that prints the ready gst-launch command without running it"

Run
---
 Copy the bash script "tda4_pose_estimation.sh" under Tappas pose_estimation directory (e.g. $TAPPAS_WORKSPACE/apps/gstreamer/general/pose_estimation)
.. code-block:: sh
   cp tda4_pose_estimation.sh $TAPPAS_WORKSPACE/apps/gstreamer/general/pose_estimation
   cd $TAPPAS_WORKSPACE/apps/gstreamer/general/pose_estimation
   ./tda4_pose_estimation.sh






