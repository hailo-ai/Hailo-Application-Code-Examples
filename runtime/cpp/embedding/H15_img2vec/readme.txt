We want a plugin (in .so format), that will write embedding vectors to text file, one line per input image.
 To be used as a gst plug-in (!hailofilter), together with a simple embedding (feature extractor) HEF given to !hailo-net

1. Ripped off yolov8_cross_compilation example
2. Modified CmakeLists.txt
3. Modified the filter() in postprocessing.cpp
4. Removed everything else.

To build, use cross-compile env 
(e.g. docker, inside Hailo repo-docker.int.hailo.ai/hailo_repo_init_cpu:hrt_dfc) 
and run ``bash build.sh /opt/poky/4.0.2``, transfer the resultant .so from 

