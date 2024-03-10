cmake_minimum_required(VERSION 3.11)
project(ds_tracker VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

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

#find gstream-1.0 gstreamer-rtsp-server-1.0, add include dirs and link libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-rtsp-server-1.0)

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
# add library to build task
# add_library(task SHARED
#     task/border_cross.cpp
#     task/gather.cpp
# )

add_executable(ds_track ds_track.cpp)
target_include_directories(ds_track PRIVATE
    /usr/local/cuda/include
    ${GSTREAMER_INCLUDE_DIRS}
    ${APP_INSTALL_ROOT}/sources/includes
)
target_link_libraries(ds_track PRIVATE
    ${GSTREAMER_LIBRARIES} cudart nvdsgst_meta nvds_meta nvds_yml_parser task
)

add_executable(ds_app ds_app.cpp)
target_include_directories(ds_app PRIVATE
    /usr/local/cuda/include
    ${GSTREAMER_INCLUDE_DIRS}
    ${APP_INSTALL_ROOT}/sources/includes
)
target_link_libraries(ds_app PRIVATE
    ${GSTREAMER_LIBRARIES} cudart nvdsgst_meta nvds_meta nvds_yml_parser task
)

add_executable(ds_app_multi ds_app_multi.cpp)
target_include_directories(ds_app_multi PRIVATE
    /usr/local/cuda/include
    ${GSTREAMER_INCLUDE_DIRS}
    ${APP_INSTALL_ROOT}/sources/includes
)
target_link_libraries(ds_app_multi PRIVATE
    ${GSTREAMER_LIBRARIES} cudart nvdsgst_meta nvds_meta nvds_yml_parser task
)
