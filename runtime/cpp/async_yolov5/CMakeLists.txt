cmake_minimum_required(VERSION 3.0.0)
project(async_infer VERSION 0.1.0)

find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(HailoRT 4.14.0 EXACT REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} multi_async.cpp yolo_post.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PRIVATE HailoRT::libhailort Threads::Threads ${OpenCV_LIBS})

if(WIN32)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /DWIN32_LEAN_AND_MEAN
        /DNOMINMAX                  # NOMINMAX is required in order to play nice with std::min/std::max (otherwise Windows.h defines it's own)
        /wd4201 /wd4251
    )
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)