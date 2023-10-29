# Install script for directory: /home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/xtl" TYPE FILE FILES
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xany.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xbasic_fixed_string.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xbase64.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xclosure.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xcompare.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xcomplex.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xcomplex_sequence.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xspan.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xspan_impl.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xdynamic_bitset.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xfunctional.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xhalf_float.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xhalf_float_impl.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xhash.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xhierarchy_generator.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xiterator_base.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xjson.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xmasked_value_meta.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xmasked_value.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xmeta_utils.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xmultimethods.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xoptional_meta.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xoptional.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xoptional_sequence.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xplatform.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xproxy_wrapper.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xsequence.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xsystem.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xtl_config.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xtype_traits.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xvariant.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xvariant_impl.hpp"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test/include/xtl/xvisitor.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/xtl" TYPE FILE FILES
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test-build/xtlConfig.cmake"
    "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test-build/xtlConfigVersion.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/xtl/xtlTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/xtl/xtlTargets.cmake"
         "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test-build/CMakeFiles/Export/share/cmake/xtl/xtlTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/xtl/xtlTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/xtl/xtlTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/xtl" TYPE FILE FILES "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test-build/CMakeFiles/Export/share/cmake/xtl/xtlTargets.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pkgconfig" TYPE FILE FILES "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test-build/xtl.pc")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/omerw/Hailo/repo/Hailo-Application-Code-Examples-1/runtime/cpp/yolov8/x86_64/build/x86_64/xtl-test-prefix/src/xtl-test-build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
