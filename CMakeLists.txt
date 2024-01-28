cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(xtrt LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -g -Wall -Werror -shared -fPIC")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# CUDA
find_package(CUDA REQUIRED)

# TensorRT
set(TensorRT_INCLUDE_DIRS "/usr/include/aarch64-linux-gnu" CACHE STRING "TensorRT headers path")
set(TensorRT_LIBRARIES "/usr/lib/aarch64-linux-gnu" CACHE STRING "TensorRT libs path")

# GStreamer
set(GSTREAMER_INCLUDE_DIRS "/usr/include/gstreamer-1.0" CACHE STRING "GStreamer include path")
set(GLIB_INCLUDE_DIRS "/usr/include/glib-2.0" CACHE STRING "GLib include path")
set(GLIBCONFIG_INCLUDE_DIRS "/usr/lib/aarch64-linux-gnu/glib-2.0/include" CACHE STRING "GLib config include path")

# DeepStream
set(DEEPSTREAM "/opt/nvidia/deepstream/deepstream" CACHE STRING "DeepStream root path")
set(DS_LIBRARIES ${DEEPSTREAM}/lib)
set(DS_INCLUDE_DIRS ${DEEPSTREAM}/sources/includes)


add_executable(deepstream_face
    deepstream_face.cpp
    modules/interrupt.cpp
    modules/perf.cpp
)
add_executable(deepstream_pose
    deepstream_pose.cpp
    modules/interrupt.cpp
    modules/perf.cpp
)

target_include_directories(deepstream_face PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${DS_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${GLIBCONFIG_INCLUDE_DIRS}
    modules
)
target_include_directories(deepstream_pose PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${DS_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${GLIBCONFIG_INCLUDE_DIRS}
    modules
)

