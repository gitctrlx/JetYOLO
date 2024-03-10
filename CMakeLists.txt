cmake_minimum_required(VERSION 3.18)

project(JetYOLO)

# xtrt
option(BUILD_XTRT "Build xtrt" ON)

# nvdsinfer_custom_impl
option(BUILD_NVDSINFER_CUSTOM_IMPL "Build nvdsinfer_custom_impl" ON)

# tools
option(BUILD_TOOLS_POLYGON_DRAW "Build Polygon_Draw" ON)

# apps
option(BUILD_APPS_DS_YOLO_DETECT "Build ds_yolo_detect" OFF)
option(BUILD_APPS_DS_YOLO_FACE "Build ds_yolo_face" OFF)
option(BUILD_APPS_DS_YOLO_POSE "Build ds_yolo_pose" OFF)
option(BUILD_APPS_DS_YOLO_TRACKER "Build ds_yolo_tracker" OFF)


if(BUILD_XTRT)
    add_subdirectory(xtrt)
endif()

if(BUILD_NVDSINFER_CUSTOM_IMPL)
    add_subdirectory(nvdsinfer_custom_impl)
endif()

if(BUILD_TOOLS_POLYGON_DRAW)
    add_subdirectory(tools/polygon_draw)
endif()

if(BUILD_APPS_DS_YOLO_DETECT)
    add_subdirectory(apps/ds_yolo_detect)
endif()

if(BUILD_APPS_DS_YOLO_FACE)
    add_subdirectory(apps/ds_yolo_face)
endif()

if(BUILD_APPS_DS_YOLO_POSE)
    add_subdirectory(apps/ds_yolo_pose)
endif()

if(BUILD_APPS_DS_YOLO_TRACKER)
    add_subdirectory(apps/ds_yolo_tracker)
endif()
