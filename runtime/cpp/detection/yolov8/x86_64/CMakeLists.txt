cmake_minimum_required(VERSION 3.20)
project(vstream_yolov8_example_cpp)

set(COMPILE_OPTIONS -Wall -Wextra -Wconversion -ffast-math -O3 -DNDEBUG -Wno-ignored-qualifiers -Wno-conversion -Wno-unused-but-set-parameter -Wno-extra -Wno-reorder -Wno-unused-local-typedefs)

set(BASE_DIR /path/to/yolov8/example/folder)

find_package(Threads)
find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES
./*.cpp
)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    include(FetchContent)

    FetchContent_Declare(xtl
        GIT_REPOSITORY https://github.com/xtensor-stack/xtl
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
        OVERRIDE_FIND_PACKAGE
    )

    FetchContent_Declare(xtensor
        GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -Dxtl_DIR=${BASE_DIR}/build/x86_64/external/share/cmake/xtl/
        OVERRIDE_FIND_PACKAGE
    )

    FetchContent_Declare(xsimd
        GIT_REPOSITORY https://github.com/xtensor-stack/xsimd
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -Dxtl_DIR=${BASE_DIR}/build/x86_64/external/share/cmake/xtl/
        OVERRIDE_FIND_PACKAGE
    )

    FetchContent_MakeAvailable(xtl xtensor xsimd)
else()
    include(ExternalProject)

    set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

    ExternalProject_Add(xtl
        GIT_REPOSITORY https://github.com/xtensor-stack/xtl
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
    )

    ExternalProject_Add(xtensor
        GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -Dxtl_DIR=${BASE_DIR}/build/x86_64/external/share/cmake/xtl/
    )

    ExternalProject_Add(xsimd
        GIT_REPOSITORY https://github.com/xtensor-stack/xsimd
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -Dxtl_DIR=${BASE_DIR}/build/x86_64/external/share/cmake/xtl/
    )

    include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
    link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)
endif()


link_libraries(stdc++fs)
add_executable(${PROJECT_NAME} ${SOURCES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} Threads::Threads HailoRT::libhailort xtl xtensor xsimd)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})