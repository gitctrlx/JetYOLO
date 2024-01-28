cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(xtrt_nvdsinfer_custom_impl LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -g -Wall -Werror -shared -fPIC")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# CUDA
find_package(CUDA REQUIRED)

# TensorRT
set(TensorRT_INCLUDE_DIRS "/usr/include/aarch64-linux-gnu" CACHE STRING "TensorRT headers path")
set(TensorRT_LIBRARIES "/usr/lib/aarch64-linux-gnu" CACHE STRING "TensorRT libs path")

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
        utils.cpp
        )

add_library(nvdsinfer_custom_impl_Yolo_face SHARED
        nvdsparseface_Yolo.cpp 
        utils.cpp)
        
add_library(nvdsinfer_custom_impl_Yolo_pose SHARED 
        nvdsparsepose_Yolo.cpp 
        utils.cpp)

# Link libraries
target_link_libraries(nvdsparsebbox_mmyolo PRIVATE nvinfer nvinfer_plugin)
target_link_libraries(nvdsinfer_custom_impl_Yolo PRIVATE nvinfer nvinfer_plugin)
target_link_libraries(nvdsinfer_custom_impl_Yolo_face PRIVATE nvinfer nvinfer_plugin)
target_link_libraries(nvdsinfer_custom_impl_Yolo_pose PRIVATE nvinfer nvinfer_plugin)

# Set include directories for each library
target_include_directories(nvdsparsebbox_mmyolo PRIVATE ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${DS_INCLUDE_DIRS})
target_include_directories(nvdsinfer_custom_impl_Yolo PRIVATE ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${DS_INCLUDE_DIRS})
target_include_directories(nvdsinfer_custom_impl_Yolo_face PRIVATE ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${DS_INCLUDE_DIRS})
target_include_directories(nvdsinfer_custom_impl_Yolo_pose PRIVATE ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${DS_INCLUDE_DIRS})

# Set compiler flags
target_compile_options(nvdsinfer_custom_impl_Yolo PRIVATE -fPIC)