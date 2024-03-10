cmake_minimum_required(VERSION 3.11)
project(ds_tracker VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# find_package(TensorRT REQUIRED)

#find gstream-1.0 gstreamer-rtsp-server-1.0, add include dirs and link libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-rtsp-server-1.0)

# CUDA
find_package(CUDA REQUIRED)

# TensorRT
set(TensorRT_INCLUDE_DIRS "/usr/include/aarch64-linux-gnu" CACHE STRING "TensorRT headers path")
set(TensorRT_LIBRARIES "/usr/lib/aarch64-linux-gnu" CACHE STRING "TensorRT libs path")

# DeepStream
set(DEEPSTREAM "/opt/nvidia/deepstream/deepstream" CACHE STRING "DeepStream root path")
set(DS_LIBRARIES ${DEEPSTREAM}/lib)
set(DS_INCLUDE_DIRS ${DEEPSTREAM}/sources/includes)

# NVDS_VERSION:=6.1
# 
# LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
# APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/
# APP_INSTALL_ROOT?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)
# set variables
set(LIB_INSTALL_DIR /opt/nvidia/deepstream/deepstream/lib/)
set(APP_INSTALL_DIR /opt/nvidia/deepstream/deepstream/bin/)
set(APP_INSTALL_ROOT /opt/nvidia/deepstream/deepstream)

link_directories(
    ${LIB_INSTALL_DIR}
    /usr/local/cuda/lib64
)

add_library(task SHARED
    task/border_cross.cpp
    task/gather.cpp
)

add_executable(ds_detect ds_detect.cpp)
target_include_directories(ds_detect PRIVATE
    /usr/local/cuda/include
    ${GSTREAMER_INCLUDE_DIRS}
    ${APP_INSTALL_ROOT}/sources/includes
)
target_link_libraries(ds_detect PRIVATE
    ${GSTREAMER_LIBRARIES} cudart nvdsgst_meta nvds_meta nvds_yml_parser task
)
