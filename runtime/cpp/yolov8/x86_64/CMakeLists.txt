cmake_minimum_required(VERSION 3.11)
project(vstream_yolov8_example_cpp)

set(COMPILE_OPTIONS -Wall -Wextra -Wconversion -O3 -DNDEBUG -Wno-ignored-qualifiers -Wno-conversion -Wno-unused-but-set-parameter -Wno-extra -Wno-reorder -Wno-unused-local-typedefs)

set(BASE_DIR /path/to/yolov8/example/folder)

find_package(Threads)
find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES
./*.cpp
)

include(ExternalProject)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

ExternalProject_Add(xtl-test
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
)

ExternalProject_Add(xtensor-test
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -Dxtl_DIR=${BASE_DIR}/build/x86_64/external/share/cmake/xtl/
)

include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

link_libraries(stdc++fs)
add_executable(${PROJECT_NAME} ${SOURCES})
add_dependencies(${PROJECT_NAME} xtl-test xtensor-test)
include_directories(${OpenCV_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} Threads::Threads HailoRT::libhailort)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
