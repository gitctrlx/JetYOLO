cmake_minimum_required(VERSION 3.11)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(ds_tracker VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)
# TensorRT
# set(TensorRT_INCLUDE_DIRS "/usr/include/aarch64-linux-gnu" CACHE STRING "TensorRT headers path")
# set(TensorRT_LIBRARIES "/usr/lib/aarch64-linux-gnu" CACHE STRING "TensorRT libs path")

#find gstream-1.0 gstreamer-rtsp-server-1.0, add include dirs and link libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-rtsp-server-1.0)

# DeepStream
set(DEEPSTREAM "/opt/nvidia/deepstream/deepstream" CACHE STRING "DeepStream root path")
set(DS_LIBRARIES ${DEEPSTREAM}/lib)
set(DS_INCLUDE_DIRS ${DEEPSTREAM}/sources/includes)


# Add executable
add_executable(ds_tracker
    deepstream_track.cpp
)
add_executable(ds_tracker_app
    deepstream_track_app.cpp
    task/border_cross.cpp
    task/gather.cpp

)

add_executable(ds_tracker_app_multi
    deepstream_track_app_multi.cpp
    task/border_cross.cpp
    task/gather.cpp
)


# Set library paths for DeepStream and CUDA
link_directories(${DS_LIBRARIES} ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Include directories
target_include_directories(ds_tracker PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${DS_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)
target_include_directories(ds_tracker_app PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${DS_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    task
)
target_include_directories(ds_tracker_app_multi PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${DS_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    task
)

# Link libraries
target_link_libraries(ds_tracker PRIVATE
    ${GSTREAMER_LIBRARIES}
    ${CUDA_LIBRARIES} cudart
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
    ${DS_LIBRARIES}/libnvdsgst_meta.so
    ${DS_LIBRARIES}/libnvds_meta.so
    ${DS_LIBRARIES}/libnvds_yml_parser.so
)
target_link_libraries(ds_tracker_app PRIVATE
    ${GSTREAMER_LIBRARIES}
    ${CUDA_LIBRARIES} cudart
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
    ${DS_LIBRARIES}/libnvdsgst_meta.so
    ${DS_LIBRARIES}/libnvds_meta.so
    ${DS_LIBRARIES}/libnvds_yml_parser.so
)
target_link_libraries(ds_tracker_app_multi PRIVATE
    ${GSTREAMER_LIBRARIES}
    ${CUDA_LIBRARIES} cudart
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
    ${DS_LIBRARIES}/libnvdsgst_meta.so
    ${DS_LIBRARIES}/libnvds_meta.so
    ${DS_LIBRARIES}/libnvds_yml_parser.so
)


# Ensure the linker can find the CUDA libraries
target_link_directories(ds_tracker PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
target_link_directories(ds_tracker_app PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
target_link_directories(ds_tracker_app_multi PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/lib64)