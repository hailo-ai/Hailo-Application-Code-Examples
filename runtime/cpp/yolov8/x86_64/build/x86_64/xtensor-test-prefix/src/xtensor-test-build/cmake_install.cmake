# Install script for directory: /home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/external")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/xtensor" TYPE FILE FILES
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xaccessible.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xaccumulator.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xadapt.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xarray.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xassign.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xaxis_iterator.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xaxis_slice_iterator.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xblockwise_reducer.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xblockwise_reducer_functors.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xbroadcast.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xbuffer_adaptor.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xbuilder.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xchunked_array.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xchunked_assign.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xchunked_view.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xcomplex.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xcontainer.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xcsv.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xdynamic_view.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xeval.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xexception.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xexpression.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xexpression_holder.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xexpression_traits.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xfixed.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xfunction.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xfunctor_view.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xgenerator.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xhistogram.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xindex_view.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xinfo.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xio.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xiterable.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xiterator.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xjson.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xlayout.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xmanipulation.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xmasked_view.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xmath.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xmime.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xmultiindex_iterator.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xnoalias.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xnorm.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xnpy.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xoffset_view.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xoperation.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xoptional.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xoptional_assembly.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xoptional_assembly_base.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xoptional_assembly_storage.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xpad.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xrandom.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xreducer.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xrepeat.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xscalar.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xsemantic.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xset_operation.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xshape.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xslice.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xsort.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xstorage.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xstrided_view.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xstrided_view_base.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xstrides.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xtensor.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xtensor_config.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xtensor_forward.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xtensor_simd.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xutils.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xvectorize.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xview.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test/include/xtensor/xview_utils.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/xtensor" TYPE FILE FILES
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test-build/xtensorConfig.cmake"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test-build/xtensorConfigVersion.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/xtensor/xtensorTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/xtensor/xtensorTargets.cmake"
         "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test-build/CMakeFiles/Export/share/cmake/xtensor/xtensorTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/xtensor/xtensorTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/xtensor/xtensorTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/xtensor" TYPE FILE FILES "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test-build/CMakeFiles/Export/share/cmake/xtensor/xtensorTargets.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pkgconfig" TYPE FILE FILES "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test-build/xtensor.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test-build/xtensor.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtensor-test-prefix/src/xtensor-test-build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
