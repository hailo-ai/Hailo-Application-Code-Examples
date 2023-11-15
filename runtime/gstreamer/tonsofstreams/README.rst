**Last TAPPAS version checked - 3.24.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


This example runs multiple streams on multiple Hailo devices.

It is built to work with 4 Hailo devices.
To download required HEF and media files run ./install.sh script.
Note that the mp4 files used here are not regular mp4 files. They are modified to be used as streaming media files.
This is done to allow to run them in loop. see scripts/gstreamer/gstreamer_video_converter.sh for details

Requirements:
- TAPPAS environment (tested on TAPPAS 3.24.0)
- 4 Hailo devices

To run the example run ./tonsofstreams.sh
