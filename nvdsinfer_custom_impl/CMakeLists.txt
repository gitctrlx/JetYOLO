cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(xtrt_nvdsinfer_custom_impl LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -g -Wall -Werror -shared -fPIC")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# CUDA
find_package(CUDA REQUIRED)

# TensorRT
# set(TensorRT_INCLUDE_DIRS "/usr/include/aarch64-linux-gnu" CACHE STRING "TensorRT headers path")
# set(TensorRT_LIBRARIES "/usr/lib/aarch64-linux-gnu" CACHE STRING "TensorRT libs path")
find_package(TensorRT REQUIRED)

# DeepStream
set(DEEPSTREAM "/opt/nvidia/deepstream/deepstream" CACHE STRING "DeepStream root path")
set(DS_LIBRARIES ${DEEPSTREAM}/lib)
set(DS_INCLUDE_DIRS ${DEEPSTREAM}/sources/includes)

# Create individual libraries for each plugin
add_library(nvdsparsebbox_mmyolo SHARED
        nvdsparsebbox_mmyolo.cpp
        )

add_library(nvdsinfer_custom_impl_Yolo SHARED
        nvdsparsebbox_Yolo.cpp
        yoloForward_nc.cu
        yoloPlugins.cpp
        )

add_library(nvdsinfer_custom_impl_Yolo_face SHARED
        nvdsparseface_Yolo.cpp 
        )
        
add_library(nvdsinfer_custom_impl_Yolo_pose SHARED 
        nvdsparsepose_Yolo.cpp 
        )

# Link libraries
target_link_libraries(nvdsparsebbox_mmyolo PRIVATE ${CUDA_LIBRARIES} nvinfer nvinfer_plugin)
target_link_libraries(nvdsinfer_custom_impl_Yolo PRIVATE ${CUDA_LIBRARIES} nvinfer nvinfer_plugin)
target_link_libraries(nvdsinfer_custom_impl_Yolo_face PRIVATE ${CUDA_LIBRARIES} nvinfer nvinfer_plugin)
target_link_libraries(nvdsinfer_custom_impl_Yolo_pose PRIVATE ${CUDA_LIBRARIES} nvinfer nvinfer_plugin)

# Set include directories for each library
target_include_directories(nvdsparsebbox_mmyolo PRIVATE ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${DS_INCLUDE_DIRS})
target_include_directories(nvdsinfer_custom_impl_Yolo PRIVATE ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${DS_INCLUDE_DIRS})
target_include_directories(nvdsinfer_custom_impl_Yolo_face PRIVATE ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${DS_INCLUDE_DIRS})
target_include_directories(nvdsinfer_custom_impl_Yolo_pose PRIVATE ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${DS_INCLUDE_DIRS})
