**Last TAPPAS version checked - 3.21.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


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
 - Hailo-8 device (Hailort and hailo_pci drivers V4.10)
 - TAPPAS environment (V3.21.0)

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






