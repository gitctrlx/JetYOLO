# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/lib/python3.9/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.9/dist-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/build

# Include any dependencies generated for this target.
include CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/flags.make

CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/nvdsparsebbox_Yolo.cpp.o: CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/flags.make
CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/nvdsparsebbox_Yolo.cpp.o: /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/nvdsparsebbox_Yolo.cpp
CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/nvdsparsebbox_Yolo.cpp.o: CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/nvdsparsebbox_Yolo.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/nvdsparsebbox_Yolo.cpp.o -MF CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/nvdsparsebbox_Yolo.cpp.o.d -o CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/nvdsparsebbox_Yolo.cpp.o -c /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/nvdsparsebbox_Yolo.cpp

CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/nvdsparsebbox_Yolo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/nvdsparsebbox_Yolo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/nvdsparsebbox_Yolo.cpp > CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/nvdsparsebbox_Yolo.cpp.i

CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/nvdsparsebbox_Yolo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/nvdsparsebbox_Yolo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/nvdsparsebbox_Yolo.cpp -o CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/nvdsparsebbox_Yolo.cpp.s

CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/yoloPlugins.cpp.o: CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/flags.make
CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/yoloPlugins.cpp.o: /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/yoloPlugins.cpp
CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/yoloPlugins.cpp.o: CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/yoloPlugins.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/yoloPlugins.cpp.o -MF CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/yoloPlugins.cpp.o.d -o CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/yoloPlugins.cpp.o -c /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/yoloPlugins.cpp

CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/yoloPlugins.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/yoloPlugins.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/yoloPlugins.cpp > CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/yoloPlugins.cpp.i

CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/yoloPlugins.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/yoloPlugins.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/yoloPlugins.cpp -o CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/yoloPlugins.cpp.s

CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/utils.cpp.o: CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/flags.make
CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/utils.cpp.o: /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/utils.cpp
CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/utils.cpp.o: CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/utils.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/utils.cpp.o -MF CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/utils.cpp.o.d -o CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/utils.cpp.o -c /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/utils.cpp

CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/utils.cpp > CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/utils.cpp.i

CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/utils.cpp -o CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/utils.cpp.s

# Object files for target nvdsinfer_custom_impl_Yolo
nvdsinfer_custom_impl_Yolo_OBJECTS = \
"CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/nvdsparsebbox_Yolo.cpp.o" \
"CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/yoloPlugins.cpp.o" \
"CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/utils.cpp.o"

# External object files for target nvdsinfer_custom_impl_Yolo
nvdsinfer_custom_impl_Yolo_EXTERNAL_OBJECTS =

libnvdsinfer_custom_impl_Yolo.so: CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/nvdsparsebbox_Yolo.cpp.o
libnvdsinfer_custom_impl_Yolo.so: CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/yoloPlugins.cpp.o
libnvdsinfer_custom_impl_Yolo.so: CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/utils.cpp.o
libnvdsinfer_custom_impl_Yolo.so: CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/build.make
libnvdsinfer_custom_impl_Yolo.so: CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library libnvdsinfer_custom_impl_Yolo.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/build: libnvdsinfer_custom_impl_Yolo.so
.PHONY : CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/build

CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/clean

CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/depend:
	cd /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/build /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/build /home/nvidia/code/deepstream-mmyolo/nvdsinfer_custom_impl/build/CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/nvdsinfer_custom_impl_Yolo.dir/depend

