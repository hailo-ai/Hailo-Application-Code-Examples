cmake_minimum_required(VERSION 3.0.0)

project(infer)

find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(HailoRT 4.13.0 EXACT REQUIRED)

add_executable(infer infer.c)
target_link_libraries(infer PRIVATE HailoRT::libhailort Threads::Threads)

if(WIN32)
    target_compile_options(infer PRIVATE
        /DWIN32_LEAN_AND_MEAN
        /DNOMINMAX                  # NOMINMAX is required in order to play nice with std::min/std::max (otherwise Windows.h defines it's own)
        /wd4201 /wd4251
    )
endif()

set_target_properties(infer PROPERTIES CXX_STANDARD 14)