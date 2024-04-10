# DeepStream YOLO APP: Boundary Detection

### Personnel/Vehicle Boundary Detection

#### Single-Stream Inference Application:

This feature enables real-time tracking and boundary detection for individuals and vehicles using a single video stream. The application utilizes DeepStream for efficient processing.

This example is based on the `app/ds_yolo_tracker` directory, showcasing its processing pipeline as illustrated below:

![](../../assets/ds_track_pipe.png)

To view an inference example, execute the following command:

```sh
./build/apps/ds_yolo_tracker/ds_tracker_app file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
```

> **Usage:** 
>
> - `./build/apps/ds_yolo_tracker/ds_tracker_app [Your video file path or RTSP stream URL]`
>
> **Display Features:**
>
> - The top-left corner shows the total count of pedestrians and vehicles that have passed.
> - At the center is a boundary detection box; vehicles crossing this area are highlighted with a red bounding box.

Upon running the application, you can view the output stream on players like [VLC](https://www.videolan.org/vlc/) by entering: `rtsp://[IP address of the device running the application]:8554/ds-test`. This allows you to see:

> **Note：**The streamed video output can be viewed on any device within the same local network.

![image-20240326181100263](../../assets/image-20240326181100263.png)

#### Multi-Stream Application:

This application extends the capabilities of the single-stream inference application to support simultaneous processing and analysis of multiple video streams. It enables efficient monitoring and boundary detection for individuals and vehicles across several feeds, leveraging NVIDIA DeepStream for optimized performance.

This example is based on the `app/ds_yolo_tracker` directory, showcasing its processing pipeline as illustrated below:

![](../../assets/ds_track_app_multi_pipe.png)

To run the application with multiple video feeds, use the following command syntax:

```sh
./build/apps/ds_yolo_tracker/ds_tracker_app_multi file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4  file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
```

> **Usage:** 
>
> - `./build/apps/ds_yolo_tracker/ds_tracker_app_multi [Video file path or RTSP stream URL 1] [Video file path or RTSP stream URL 2] [...]`
> - **note:** After compilation, the current program only supports input from two stream addresses. If you wish to facilitate input from more streams, you will need to modify the corresponding code. For details, please refer to the [detailed documentation](doc).
>
> **Display Features:** The application provides a unified display that incorporates elements from all the processed streams.
>
> - **Overall Counts:** The top-left corner of each video feed display shows the total count of pedestrians and vehicles that have passed within that specific stream.
> - **Boundary Detection Box:** A boundary detection box is presented at the center of each video feed. Vehicles crossing this predefined area in any of the streams are immediately highlighted with a red bounding box to signify a boundary violation.

Upon running the application, you can view the output stream on players like [VLC](https://www.videolan.org/vlc/) by entering: `rtsp://[IP address of the device running the application]:8554/ds-test`. This allows you to see:

> **Note：**The streamed video output can be viewed on any device within the same local network.

![image-20240329161014875](../../assets/image-20240329161014875.png)