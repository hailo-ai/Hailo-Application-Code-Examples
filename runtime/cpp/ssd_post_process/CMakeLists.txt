cmake_minimum_required(VERSION 3.11)
project(vstream_ssd_example_cpp)

set(CMAKE_CXX_STANDARD 20)
set(COMPILE_OPTIONS -Werror -Wall -Wextra -Wconversion -O3 -Wno-reorder)

find_package(Threads)
find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES
    ./*.cpp
)
message(STATUS "Found SOURCES: " ${SOURCES})

add_executable(${PROJECT_NAME} ${SOURCES})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} HailoRT::libhailort Threads::Threads)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
