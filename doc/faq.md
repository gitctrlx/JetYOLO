# FQA

### Q: CMake compilation stage did not find `gstreamer-rtsp-server-1.0`

```sh
CMake Error at /usr/share/cmake-3.28/Modules/FindPkgConfig.cmake:619 (message):
  The following required packages were not found:

   - gstreamer-rtsp-server-1.0

Call Stack (most recent call first):
  /usr/share/cmake-3.28/Modules/FindPkgConfig.cmake:841 (_pkg_check_modules_internal)
  apps/ds_yolo_detect/CMakeLists.txt:24 (pkg_check_modules)


-- Configuring incomplete, errors occurred!
```

A:

Run the following command to install:

```sh
apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio libgstrtspserver-1.0-dev
```

> Official installation website: https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c

### Q: CMake compilation stage did not find `nlohmann/json.hpp`

```sh
In file included from /JetYOLO/xtrt/tools/eval/main.cpp:1:
/JetYOLO/xtrt/tools/eval/yolo_infer.h:18:10: fatal error: nlohmann/json.hpp: No such file or directory
   18 | #include <nlohmann/json.hpp>
      |          ^~~~~~~~~~~~~~~~~~~
compilation terminated.
make[2]: *** [xtrt/tools/eval/CMakeFiles/yolo_eval.dir/build.make:76: xtrt/tools/eval/CMakeFiles/yolo_eval.dir/main.cpp.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:361: xtrt/tools/eval/CMakeFiles/yolo_eval.dir/all] Error 2
make: *** [Makefile:91: all] Error 2
```

A:

```sh
sudo apt-get update
sudo apt-get install nlohmann-json3-dev
```