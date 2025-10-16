cmake_minimum_required(VERSION 3.10.0)
project(classifier)

set(COMPILE_OPTIONS -Wall -std=gnu++2a -Werror -O3)

find_package(HailoRT REQUIRED)
find_package(Threads)
find_package(OpenCV REQUIRED)

message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} HailoRT::libhailort Threads::Threads)
