Object Detection with Tracking using ByteTracker and Supervision in GSTreamer
==============================================================================

This script performs object detection with tracking on a video using Hailo.
It annotates detected objects with bounding boxes and labels and tracks them across frames in the video.

Requirements
------------

- hailo_platform==4.17.0
- hailo_tappas 3.28.0 
- supervision

Supported Models
----------------

This example is an enhancement to the standard detection pipeline from TAPPAS and supports the same
options

Usage
-----
0. Get into TAPPAS docker:
    Assuming that you have TAPPAS 3.28.0 installed, enter the container:
    ```shell script
    ./run_tappas_docker.sh --resume
        
    cd apps/h8/gstreamer/general/
    ```

1. Clone the repository:
    ```shell script
    git clone <https://github.com/hailo-ai/Hailo-Application-Code-Examples.git>
        
    cd example-folder-path
    ```

2. Install dependencies:
    ```shell script
    pip install -r requirements.txt
    ```

4. copy the files into the detection folder

5. Run the script:
    ```shell script
    ./detection.sh
    ```

**Output**

![Output example](./tracker.gif?raw=true)

Additional Notes
----------------

- The example was only tested with ``TAPPAS 3.28.0``

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.

