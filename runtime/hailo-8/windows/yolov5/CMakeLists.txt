cmake_minimum_required(VERSION 3.11)
project(cpp_yolov5_win_standalone_example)

set(CMAKE_CXX_STANDARD 20)

set(COMPILE_OPTIONS -Wall)

# / d5045 = Spectre mitigation warning
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd5045") 

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Threads)
find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})

message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES ./*.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} HailoRT::libhailort)
target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})