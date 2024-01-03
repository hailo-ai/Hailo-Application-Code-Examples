**Last TAPPAS version checked - 3.25.0**

**Disclaimer:** <br />
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.<br />
Please note that this example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.


Multistream example with stream ID
=============

| This example is based on the TAPPAS multistream example.
| Follow #STREAM_ID# remarks in the code to see the changes.
| Adding the stream ID is done using the libstream_id_tool
| Adding ID is done using the plugin like so:
| hailofilter so-path=$STREAM_ID_SO config-path=SRC_NAME
| Reading the ID is done using the libstream_id_tool like so:
| hailofilter so-path=$STREAM_ID_SO function-name=print_stream_id
| You can see its code under tappas/core/hailo/libs/tools/set_stream_id.cpp
| The roi->get_stream_id() and roi->set_stream_id() are used to get and set the stream ID.
| These are off-course available also as API you can use in your code.
| You can run the example directly running ./multi_stream_detection.sh
| Note that this script is using the same dependencies as the TAPPAS multistream example.

Requirements
============
- TAPPAS environment
   - TAPPAS Docker (tested on TAPPAS 3.25.0)
   - Halio Suite Docker (tested on hailo_sw_suite_2023-07)
- Hailo device

