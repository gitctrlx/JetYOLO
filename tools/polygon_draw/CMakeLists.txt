cmake_minimum_required(VERSION 3.18)

project(polygon_draw)

set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)

if (OpenCV_FOUND)
    message(STATUS "OpenCV found")
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
    message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "[E] OpenCV not found")
endif()


include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

add_executable(polygon_draw polygon_draw.cpp)
