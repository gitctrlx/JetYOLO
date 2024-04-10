# DeepStream YOLO Detection

**Quick Start**

- You can quickly launch a DeepStream application using deepStream-app:

Before running the code below, please make sure that you have built the engine file using `xtrt`, meaning you have completed the section `3. Building the Engine.`

```sh
deepstream-app -c deepstream_app_config.txt
```

> **Note:**  If you wish to start directly from this step, please ensure that you have completed the following preparations:
>
> First, you need to modify the `deepstream_app_config.txt` configuration file by updating the engine file path to reflect your actual engine file path. Given that the engine is built within xtrt, you will find the engine file within the `xtrt/engine` directory. In addition to this, it is crucial to verify that the path to your plugin has been properly compiled. By default, the plugin code resides in the `nvdsinfer_custom_impl` folder, while the compiled plugin `.so` files can be found in the `build/nvdsinfer_custom_impl` directory.

- Alternatively, you can run the following code to view an example of the detection inference:


```bash
./build/apps/ds_yolo_detect/ds_detect file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
```


> **Note:**
>
> The command to run is: `./build/apps/ds_yolo_tracker/ds_tracker_app [Your video file path or RTSP stream URL]`
>
> Display Contents:
>
> - The top left corner shows the current frame's pedestrian and vehicle count.
> - Detected individuals and vehicles within the frame will be marked with bounding boxes.

This example is based on the `app/ds_yolo_detect` directory, showcasing its processing pipeline as illustrated below:

![](./assets/ds_detect_pipe.png)

Upon running the application, you can view the output stream on players like [VLC](https://www.videolan.org/vlc/) by entering: `rtsp://[IP address of the device running the application]:8554/ds-test`. This allows you to see:

> **Note：**The streamed video output can be viewed on any device within the same local network.

![image-20240328113411809](../../assets/image-20240328113411809.png)