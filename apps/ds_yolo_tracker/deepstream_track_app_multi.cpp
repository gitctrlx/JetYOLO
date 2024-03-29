#include "deepstream_track.h"
#include "utils.h"

/**
 * @brief Sets the polygon for a given source ID in the deepstream track application.
 * 
 * This function reads the polygon points from a file and assigns them to the specified source ID.
 * If the polygon for the source ID is already set, this function does nothing.
 * 
 * @param source_id The ID of the source for which to set the polygon.
 * @param frame_meta A pointer to the NvDsFrameMeta structure containing the frame metadata.
 */
void set_polygon_for_source(guint source_id, NvDsFrameMeta *frame_meta) {
    if (g_source_info[source_id].g_polygon.empty()) {
        guint width = frame_meta->pipeline_width;
        guint height = frame_meta->pipeline_height;
        std::string filePath = "./configs/ds_yolo_tracker_config/polygon_" + std::to_string(source_id) + ".txt";
        readPoints(filePath, g_source_info[source_id].g_polygon, width, height);
        g_print("Polygon for source %d loaded. Frame height = %d, width = %d\n", source_id, height, width);
    }
}

/**
 * @brief Sets the properties of the NvDsObjectMeta based on the given polygon.
 * 
 * This function sets the color and background properties of the detection box in the NvDsObjectMeta
 * based on whether the center point of the detection box is inside the given polygon.
 * If the center point is inside the polygon, the color of the detection box is set to red and the
 * background color is set to red with transparency 0.2. Otherwise, the color of the detection box
 * is set to green.
 * 
 * @param obj_meta The NvDsObjectMeta object to set the properties for.
 * @param polygon The polygon to check if the center point is inside.
 */
static void set_object_meta_properties(NvDsObjectMeta *obj_meta, const std::vector<Point>& polygon) {
    // Obtain the center point of the detection box
    Point center = {
        obj_meta->rect_params.left + obj_meta->rect_params.width / 2,
        obj_meta->rect_params.top + obj_meta->rect_params.height / 2
    };

    // If the center point is within the polygon.
    bool insidePolygon = isInside(polygon, center);

    if (insidePolygon) {
        // Change the color of the detection box to red
        obj_meta->rect_params.border_color.red = 1.0;
        obj_meta->rect_params.border_color.green = 0.0;
        obj_meta->rect_params.border_color.blue = 0.0;
        obj_meta->rect_params.border_color.alpha = 1.0;
        // Set the background color of the detection box to red and the transparency to 0.2
        obj_meta->rect_params.bg_color.red = 1.0;
        obj_meta->rect_params.bg_color.green = 0.0;
        obj_meta->rect_params.bg_color.blue = 0.0;
        obj_meta->rect_params.bg_color.alpha = 0.2;
        obj_meta->rect_params.has_bg_color = 1;
    } else {
        // Change the color of the detection box to green
        obj_meta->rect_params.border_color.red = 0.0;
        obj_meta->rect_params.border_color.green = 1.0;
        obj_meta->rect_params.border_color.blue = 0.0;
        obj_meta->rect_params.border_color.alpha = 1.0;
    }
}


/**
 * @brief Updates the display metadata for a given source ID.
 * 
 * This function updates the display metadata with the person count, vehicle count, and polygon information for the given source ID.
 * It prepares the text to be displayed on the screen and sets the font, color, and background parameters for the text.
 * It also draws the polygon on the screen with a specified line width and color.
 * 
 * @param display_meta A pointer to the NvDsDisplayMeta structure representing the display metadata.
 * @param source_id The ID of the source for which the display metadata is being updated.
 */
void update_display_meta(NvDsDisplayMeta *display_meta, guint source_id) {
    // Ensure there is a polygon and counts to display for the given source_id
    assert(source_id < (sizeof(g_source_info) / sizeof(g_source_info[0])) && "Source ID is out of bounds");

    auto& source_info = g_source_info[source_id];
    const auto& polygon = source_info.g_polygon;

    // Preparing text for display
    char display_text[MAX_DISPLAY_LEN];
    snprintf(display_text, MAX_DISPLAY_LEN, "Person Count = %zu, Vehicle Count = %zu", source_info.g_person_ids.size(), source_info.g_vehicle_ids.size());

    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    txt_params->display_text = g_strdup(display_text);
    txt_params->x_offset = 10;
    txt_params->y_offset = 12;
    txt_params->font_params.font_name = "Serif";
    txt_params->font_params.font_size = 20;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;
    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;
    display_meta->num_labels = 1;

    // Drawing the polygon
    display_meta->num_lines = polygon.size();
    for (size_t i = 0; i < polygon.size(); ++i) {
        size_t next_i = (i + 1) % polygon.size();
        NvOSD_LineParams &line_params = display_meta->line_params[i];
        line_params.x1 = polygon[i].x;
        line_params.y1 = polygon[i].y;
        line_params.x2 = polygon[next_i].x;
        line_params.y2 = polygon[next_i].y;
        line_params.line_width = 2;
        line_params.line_color.red = 1.0;
        line_params.line_color.green = 0.0;
        line_params.line_color.blue = 1.0; // Magenta for visibility
        line_params.line_color.alpha = 1.0;
    }
}

/**
 * @brief Callback function for the pad probe on the sink pad of the OSD element.
 * 
 * This function is called when a buffer is received on the sink pad of the OSD element.
 * It processes the buffer and updates the display metadata for each frame in the buffer.
 * 
 * @param pad The sink pad on which the probe is installed.
 * @param info Information about the probe.
 * @param u_data User data passed to the probe.
 * @return The return value of the probe function.
 */
static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    GstBuffer *buf = GST_BUFFER(info->data);
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        guint source_id = frame_meta->source_id;

        assert(source_id < (sizeof(g_source_info) / sizeof(g_source_info[0])) && "Source ID is out of bounds");

        set_polygon_for_source(source_id, frame_meta);

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
            set_object_meta_properties(obj_meta, g_source_info[source_id].g_polygon);

            // Vehicle and person classification and tracking
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
                if (std::find(g_source_info[source_id].g_vehicle_ids.begin(), g_source_info[source_id].g_vehicle_ids.end(), obj_meta->object_id) == g_source_info[source_id].g_vehicle_ids.end()) {
                    g_source_info[source_id].g_vehicle_ids.push_back(obj_meta->object_id);
                }
            } else if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                if (std::find(g_source_info[source_id].g_person_ids.begin(), g_source_info[source_id].g_person_ids.end(), obj_meta->object_id) == g_source_info[source_id].g_person_ids.end()) {
                    g_source_info[source_id].g_person_ids.push_back(obj_meta->object_id);
                }
            }
        }
        NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        update_display_meta(display_meta, source_id);
        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }

    frame_number++;
    update_frame_counter();
    return GST_PAD_PROBE_OK;
}


/**
 * @brief A boolean data type in GLib, representing true or false.
 *
 * @return TRUE if the operation is successful, FALSE otherwise.
 */
static gboolean set_tracker_properties(GstElement *nvtracker)
{
    GError *error = NULL;
    g_autoptr(GKeyFile) key_file = g_key_file_new();
    if (!g_key_file_load_from_file(key_file, TRACKER_CONFIG_FILE, G_KEY_FILE_NONE, &error)) {
        g_printerr("Failed to load config file: %s\n", error->message);
        return FALSE;
    }

    g_auto(GStrv) keys = g_key_file_get_keys(key_file, CONFIG_GROUP_TRACKER, NULL, &error);
    if (error != NULL) {
        g_printerr("Failed to get keys: %s\n", error->message);
        return FALSE;
    }

    for (gchar **key = keys; *key; ++key) {
        if (!g_strcmp0(*key, CONFIG_GROUP_TRACKER_WIDTH) || !g_strcmp0(*key, CONFIG_GROUP_TRACKER_HEIGHT) ||
            !g_strcmp0(*key, CONFIG_GPU_ID) || !g_strcmp0(*key, CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS)) {
            gint value = g_key_file_get_integer(key_file, CONFIG_GROUP_TRACKER, *key, &error);
            if (error != NULL) {
                g_printerr("Failed to get integer for key '%s': %s\n", *key, error->message);
                return FALSE;
            }
            g_object_set(G_OBJECT(nvtracker), *key, value, NULL);
        } else if (!g_strcmp0(*key, CONFIG_GROUP_TRACKER_LL_CONFIG_FILE) || !g_strcmp0(*key, CONFIG_GROUP_TRACKER_LL_LIB_FILE)) {
            g_autofree gchar *file_path = get_absolute_file_path(TRACKER_CONFIG_FILE, g_key_file_get_string(key_file, CONFIG_GROUP_TRACKER, *key, &error));
            if (error != NULL) {
                g_printerr("Failed to get file path for key '%s': %s\n", *key, error->message);
                return FALSE;
            }
            g_object_set(G_OBJECT(nvtracker), *key, file_path, NULL);
        }
        else {
            g_printerr("Unknown key '%s' for group [%s]", *key, CONFIG_GROUP_TRACKER);
        }
    }

    return TRUE;
}

/**
 * @brief Callback function for handling new pads in the decodebin element.
 *
 * This function is called when a new pad is added to the decodebin element. It checks the media type of the pad and proceeds only if it is for video. If the media type indicates an NVIDIA decoder plugin, it links the decoder src pad to the source bin ghost pad. Otherwise, it prints an error message.
 *
 * @param decodebin The decodebin element.
 * @param decoder_src_pad The decoder src pad.
 * @param data The user data (source bin element).
 */
static void cb_newpad(GstElement *decodebin, GstPad *decoder_src_pad, gpointer data) {
    GstElement *source_bin = (GstElement *)data;
    GstCaps *caps = gst_pad_get_current_caps(decoder_src_pad);

    if (!caps) {
        g_printerr("Failed to get caps\n");
        return;
    }

    const gchar *media_type = gst_structure_get_name(gst_caps_get_structure(caps, 0));
    g_print("Media type: %s\n", media_type);

    // Proceed only if the pad is for video
    if (g_str_has_prefix(media_type, "video/")) {
        GstCapsFeatures *features = gst_caps_get_features(caps, 0);

        // Check for NVMM memory features indicating an NVIDIA decoder plugin
        if (gst_caps_features_contains(features, "memory:NVMM")) {
            GstPad *bin_ghost_pad = gst_element_get_static_pad(source_bin, "src");

            if (!gst_ghost_pad_set_target(GST_GHOST_PAD(bin_ghost_pad), decoder_src_pad)) {
                g_printerr("Failed to link decoder src pad to source bin ghost pad\n");
            } else {
                g_print("Successfully linked the decoder src pad to source bin ghost pad\n");
            }

            gst_object_unref(bin_ghost_pad);
        } else {
            g_printerr("Error: Decodebin did not pick NVIDIA decoder plugin.\n");
        }
    }

    gst_caps_unref(caps);
}

/**
 * @brief Callback function for the "child-added" signal of a GstChildProxy.
 *        This function is called when a child element is added to the GstChildProxy.
 *        It checks if the added element is a decodebin and connects the "child-added"
 *        signal of the decodebin to itself recursively.
 *
 * @param child_proxy The GstChildProxy object that emitted the signal.
 * @param object The GObject representing the added child element.
 * @param name The name of the added child element.
 * @param user_data User data passed to the signal handler.
 */
static void decodebin_child_added(GstChildProxy *child_proxy, GObject *object, gchar *name, gpointer user_data) {
    g_print("Decodebin child added: %s\n", name);

    // Directly check if the element is a decodebin instead of using strstr.
    // This is more straightforward and avoids unnecessary string scanning
    // since we are looking for a match at the start of the string.
    if (g_str_has_prefix(name, "decodebin")) {
        g_signal_connect(object, "child-added", G_CALLBACK(decodebin_child_added), user_data);
        g_print("Connected to 'child-added' signal for %s\n", name);
    }
}

/**
 * @brief Creates a source bin with a GstElement for reading from a URI.
 * 
 * This function creates a source GstBin to abstract the content of the bin from the rest of the pipeline.
 * It creates a GstElement of type "uridecodebin" to read from the specified URI.
 * The function also connects to the "pad-added" signal of the decodebin and adds the URI decode bin to the source bin.
 * Finally, it creates a ghost pad for the source bin and returns the source bin.
 * 
 * @param index The index of the source bin.
 * @param uri The URI to read from.
 * @return A GstElement representing the source bin, or NULL if the ghost pad creation fails.
 */
GstElement *create_source_bin(guint index, const gchar *uri)
{
    g_print("Creating source bin\n");

    // Create a source GstBin to abstract this bin's content from the rest of the pipeline
    gchar bin_name[16];
    g_snprintf(bin_name, sizeof(bin_name), "source-bin-%02d", index);
    g_print("%s\n", bin_name);
    GstElement *nbin = gst_bin_new(bin_name);

    // Source element for reading from the URI
    GstElement *uri_decode_bin = gst_element_factory_make("uridecodebin", "uri-decode-bin");

    // Set the input URI to the source element
    g_object_set(G_OBJECT(uri_decode_bin), "uri", uri, NULL);

    // Connect to the "pad-added" signal of the decodebin
    g_signal_connect(uri_decode_bin, "pad-added", G_CALLBACK(cb_newpad), nbin);
    g_signal_connect(uri_decode_bin, "child-added", G_CALLBACK(decodebin_child_added), nbin);

    // Add the URI decode bin to the source bin
    gst_bin_add(GST_BIN(nbin), uri_decode_bin);

    // Create a ghost pad for the source bin
    GstPad *bin_pad = gst_ghost_pad_new_no_target("src", GST_PAD_SRC);
    if (!bin_pad)
    {
        g_printerr("Failed to add ghost pad in source bin\n");
        return NULL;
    }

    gst_element_add_pad(nbin, bin_pad);
    return nbin;
}

int main(int argc, char *argv[])
{
    GMainLoop   *loop              = NULL;
    GstElement  *pipeline          = NULL, 
                *source            = NULL, 
                *source_1          = NULL, 
                *streammux         = NULL, 
                *pgie              = NULL,
                *nvtracker         = NULL, 
                *nvvidconv         = NULL,
                *nvosd             = NULL, 
                *nvvidconv_postosd = NULL, 
                *caps              = NULL, 
                *encoder           = NULL, 
                *rtppay            = NULL, 
                *sink              = NULL;

    GstBus      *bus               = NULL;
    guint        bus_watch_id      = 0;
    GstPad      *osd_sink_pad      = NULL;
    GstCaps     *caps_filter       = NULL;

    int current_device = -1;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

    /* Check input arguments */
    if (argc < 3)
    {
        g_printerr("OR: %s <H264 filename>\n", argv[0]);
        return -1;
    }

    /* Standard GStreamer initialization */
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    /**
     * @brief Initializes the GStreamer elements required for the pipeline.
     *
     * This code block is responsible for creating the various GStreamer elements that make up
     * the streaming pipeline. Each element has a specific role in the processing and handling
     * of video streams, from input to output.
     */

    pipeline          = gst_pipeline_new("ds-tracker-pipeline");                           // Create pipeline
    
    source            = create_source_bin(0, argv[1]);                                     // Create a source'bin element to read video streams from a file
    source_1          = create_source_bin(1, argv[2]); 

    streammux         = gst_element_factory_make("nvstreammux", "stream-muxer");           // Create a stream multiplexer to merge multiple streams into one stream and package multiple frames into a batch
    pgie              = gst_element_factory_make("nvinfer", "primary-nvinference-engine"); // Create PGIE elements for executing inference
    nvtracker         = gst_element_factory_make("nvtracker", "tracker");                  // Create a tracker element to track recognized objects
    nvvidconv         = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");   // Create nvvidconv elements to convert NV12 to RGBA
    nvosd             = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");         // Create an nvosd element to draw on the converted RGBA buffer
    nvvidconv_postosd = gst_element_factory_make("nvvideoconvert", "convertor_postosd");   // Create the nvvidconv_postosd element to convert NV12 to RGBA
    caps              = gst_element_factory_make("capsfilter", "filter");                  // Create caps elements to set video format

    GstElement *tiler = gst_element_factory_make("nvmultistreamtiler", "tiler");           // 创建tiler元素， 用于将多个视频流拼接为一个视频流
    guint tiler_rows  = 2;
    guint tiler_columns = 1;

    printf("Tiler rows: %d, columns: %d \n", tiler_rows, tiler_columns);
    /* we set the tiler properties here */
    g_object_set(G_OBJECT(tiler), "rows", tiler_rows, "columns", tiler_columns,
                 "width", 800, "height", 1000, NULL);

    // Determine the correct encoder and RTP payload packer based on the specified codec.
    const gchar *encoder_element, *rtppay_element;
    if (g_strcmp0(codec, "H264") == 0) {
        encoder_element = "nvv4l2h264enc";
        rtppay_element  = "rtph264pay";
        printf("Creating H264 Encoder and rtppay\n");
    } else if (g_strcmp0(codec, "H265") == 0) {
        encoder_element = "nvv4l2h265enc";
        rtppay_element  = "rtph265pay";
        printf("Creating H265 Encoder and rtppay\n");
    } else {
        g_printerr("Unsupported codec: %s. Exiting.\n", codec);
        return -1; 
    }

    // Create the encoder, RTP payload packer, and UDP sink based on the selected codec.
    encoder = gst_element_factory_make(encoder_element, "encoder");
    rtppay  = gst_element_factory_make(rtppay_element, "rtppay");
    sink    = gst_element_factory_make("udpsink", "udpsink");

    // Verify that all elements were created successfully.
    if (!source || !source_1 || !pgie || !nvtracker || !nvvidconv || !nvosd || !nvvidconv_postosd ||
        !caps || !encoder || !rtppay || !sink) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    /**
     * @brief Configures parameters for various elements in the GStreamer pipeline.
     *
     * This section sets up essential properties for the elements within the pipeline to ensure
     * correct operation and optimization of the video processing and streaming workflow.
     */
    g_object_set(G_OBJECT(streammux), 
        "batch-size", 1,
        "width", MUXER_OUTPUT_WIDTH, 
        "height", MUXER_OUTPUT_HEIGHT,
        "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, 
        NULL);
    g_object_set(G_OBJECT(pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);
    set_tracker_properties(nvtracker);
    g_object_set(G_OBJECT(caps), "caps", gst_caps_from_string("video/x-raw(memory:NVMM), format=I420"), NULL);
    gst_caps_unref(caps_filter); 
    g_object_set(G_OBJECT(encoder), "bitrate", bitrate, NULL);
    if (is_aarch64()) {
        g_object_set(G_OBJECT(encoder),
            "preset-level", 1, 
            "insert-sps-pps", 1, 
            NULL);
    }
    g_object_set(G_OBJECT(sink), 
        "host", "224.224.255.255", 
        "port", updsink_port_num, 
        "async", FALSE, 
        "sync", 1, 
        NULL);
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus); // Release bus objects to avoid memory leaks


    /**
     * @brief Adds a series of elements to the GStreamer pipeline.
     */
    gst_bin_add_many(GST_BIN(pipeline),
                     source, source_1, streammux, pgie, nvtracker, tiler,
                     nvvidconv, nvosd, nvvidconv_postosd, caps, encoder, rtppay, sink, NULL);

    /**
     * @brief Links the source element to the stream multiplexer's sink pad.
     */
    GstPad *sinkpad, *srcpad;

    gchar pad_name_src[16] = "src";
    GstElement *source_bin[2] = {source, source_1};

    for (int i = 0; i < 2; i++) {
        gchar pad_name_sink[16];
        snprintf(pad_name_sink, 15, "sink_%u", i);
        sinkpad = gst_element_get_request_pad(streammux, pad_name_sink);
        srcpad = gst_element_get_static_pad(source_bin[i], pad_name_src);

        if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
            g_printerr("Failed to link decoder to stream muxer. Exiting.\n");
            return -1;
        }
        gst_object_unref(sinkpad);
        gst_object_unref(srcpad);
    }
 
    if (!gst_element_link_many(streammux, pgie, nvtracker, tiler,
                               nvvidconv, nvosd, nvvidconv_postosd, caps, encoder, rtppay, sink, NULL)) {
        g_printerr("Elements could not be linked. Exiting.\n");
        return -1;
    }

    osd_sink_pad = gst_element_get_static_pad(nvtracker, "src");
    if (!osd_sink_pad) g_print("Unable to get sink pad\n");
    else gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_sink_pad_buffer_probe, NULL, NULL); // 添加探针
    g_timeout_add(5000, perf_print_callback, &g_perf_data);                                                // 添加定时器，用于打印性能数据
    gst_object_unref(osd_sink_pad);

    /**
     * @brief Initializes and configures the RTSP server for streaming video.
     */
    GstRTSPServer       *server;
    GstRTSPMountPoints  *mounts;
    GstRTSPMediaFactory *factory;

    server = gst_rtsp_server_new();
    g_object_set(G_OBJECT(server), "service", g_strdup_printf("%d", rtsp_port_num), NULL);
    gst_rtsp_server_attach(server, NULL);
    mounts = gst_rtsp_server_get_mount_points(server);

    factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(factory, g_strdup_printf("( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )", updsink_port_num, codec));
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    gst_rtsp_mount_points_add_factory(mounts, rtsp_path, factory);

    g_object_unref(mounts);

    printf("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d%s ***\n\n", rtsp_port_num, rtsp_path);

    /**
     * @brief Starts the GStreamer pipeline, enters the main loop to process streaming data, and performs cleanup upon exit.
     */
    /* Set the pipeline to "playing" state */
    g_print("Using file: %s\n", argv[1]);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    /* Iterate */
    g_print("Running...\n");
    g_main_loop_run(loop);

    /* Out of the main loop, clean up nicely */
    g_print("Returned, stopping playback\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    g_print("Deleting pipeline\n");
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    return 0;
}