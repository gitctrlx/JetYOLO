#include "deepstream_track.h"
#include "utils.h"

/**
 * @brief Sets the properties of the NvDsObjectMeta based on the given polygon.
 * 
 * This function sets the color and background properties of the detection box in the NvDsObjectMeta structure
 * based on whether the center point of the detection box is inside the given polygon.
 * If the center point is inside the polygon, the color of the detection box is set to red and the background color
 * is set to red with a transparency of 0.2. Otherwise, the color of the detection box is set to green.
 * 
 * @param obj_meta A pointer to the NvDsObjectMeta structure.
 * @param polygon A vector of Point objects representing the polygon.
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
 * @brief Updates the display metadata with text and polygon information.
 * 
 * This function adds text to the display metadata, including the person count and vehicle count.
 * It also draws a polygon on the display metadata based on the given points.
 * 
 * @param display_meta The display metadata to be updated.
 * @param polygon The vector of points representing the polygon to be drawn.
 * @param person_count Pointer to the variable storing the person count.
 * @param vehicle_count Pointer to the variable storing the vehicle count.
 */
static void update_display_meta(NvDsDisplayMeta *display_meta, const std::vector<Point>& polygon, guint *person_count, guint *vehicle_count) {
    /**
     * @brief Add Text.
     */
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    char display_text[MAX_DISPLAY_LEN];
    snprintf(display_text, MAX_DISPLAY_LEN, "Person Count = %u, Vehicle Count = %u", g_person_ids.size(), g_vehicle_ids.size());

    txt_params->display_text = g_strdup(display_text);
    // Set the position of text
    txt_params->x_offset = 10;
    txt_params->y_offset = 12;
    // typeface
    txt_params->font_params.font_name = "Serif";
    txt_params->font_params.font_size = 20;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;
    // background color
    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;
    display_meta->num_labels = 1;

    /**
     * @brief draw a polygon
     */
    display_meta->num_lines = polygon.size();
    for (guint i = 0; i < polygon.size(); ++i) {
        NvOSD_LineParams *line_params = &display_meta->line_params[i];
        line_params->x1 = polygon[i].x;
        line_params->y1 = polygon[i].y;
        line_params->x2 = polygon[(i + 1) % polygon.size()].x;
        line_params->y2 = polygon[(i + 1) % polygon.size()].y;
        line_params->line_width = 2;
        line_params->line_color.red = 1.0;
        line_params->line_color.green = 0.0;
        line_params->line_color.blue = 1.0; // Magenta for visibility
        line_params->line_color.alpha = 1.0;
    }
}

/**
 * @brief The return value of the pad probe function.
 * 
 * This enumeration represents the possible return values of the GstPadProbe function.
 * - GST_PAD_PROBE_OK: The probe function completed successfully.
 * - GST_PAD_PROBE_DROP: The probe function requests to drop the buffer.
 * - GST_PAD_PROBE_REMOVE: The probe function requests to remove the probe.
 * - GST_PAD_PROBE_PASS: The probe function requests to pass the buffer to the next probe or the pad.
 * 
 * @see GstPadProbeReturn
 */
static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    GstBuffer *buf = GST_BUFFER(info->data);
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    guint vehicle_count = 0;
    guint person_count = 0;

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
            // Initialize or update polygon from a file if needed
        if (g_polygon.empty()) {
            guint width = frame_meta->source_frame_width;
            guint height = frame_meta->source_frame_height;
            readPoints(POLYGON_FILE_PATH, g_polygon, width, height);
        }

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);

            // Vehicle and person classification
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
                if (std::find(g_vehicle_ids.begin(), g_vehicle_ids.end(), obj_meta->object_id) == g_vehicle_ids.end()) {
                    g_vehicle_ids.push_back(obj_meta->object_id);
                }
                vehicle_count++;
            } else if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                if (std::find(g_person_ids.begin(), g_person_ids.end(), obj_meta->object_id) == g_person_ids.end()) {
                    g_person_ids.push_back(obj_meta->object_id);
                }
                person_count++;
            }

            // Adjust object meta properties based on its position relative to the polygon
            set_object_meta_properties(obj_meta, g_polygon);
        }

        // After processing all objects, update display metadata for the current frame
        NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        update_display_meta(display_meta, g_polygon, &person_count, &vehicle_count);
        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }

    // Increment the global frame number and optionally update any global state
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
    if (argc != 2)
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
    streammux         = gst_element_factory_make("nvstreammux", "stream-muxer");           // Create a stream multiplexer to merge multiple streams into one stream and package multiple frames into a batch
    pgie              = gst_element_factory_make("nvinfer", "primary-nvinference-engine"); // Create PGIE elements for executing inference
    nvtracker         = gst_element_factory_make("nvtracker", "tracker");                  // Create a tracker element to track recognized objects
    nvvidconv         = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");   // Create nvvidconv elements to convert NV12 to RGBA
    nvosd             = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");         // Create an nvosd element to draw on the converted RGBA buffer
    nvvidconv_postosd = gst_element_factory_make("nvvideoconvert", "convertor_postosd");   // Create the nvvidconv_postosd element to convert NV12 to RGBA
    caps              = gst_element_factory_make("capsfilter", "filter");                  // Create caps elements to set video format
    
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
    if (!source || !pgie || !nvtracker || !nvvidconv || !nvosd || !nvvidconv_postosd ||
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

    // 1. Configure the stream multiplexer (streammux) settings
    // Sets the batch size, resolution (width and height), and timeout for batched push operations.
    // These settings are crucial for handling stream inputs, especially when dealing with multiple streams or high-resolution video.
    g_object_set(G_OBJECT(streammux), "batch-size", 1, NULL);
    g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height", MUXER_OUTPUT_HEIGHT, "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    // 2. Configure the Primary GIE (pgie) settings
    // Specifies the configuration file for the GIE, which contains model and inference settings.
    g_object_set(G_OBJECT(pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);

    // 3. (Commented out) Set tracker element parameters
    // The tracker's properties, if utilized, would be set here to track objects across frames.
    set_tracker_properties(nvtracker);

    // 4. Define the video format for the caps filter
    // The caps (capabilities) filter specifies the expected video format, essential for ensuring compatibility between pipeline elements.
    caps_filter = gst_caps_from_string("video/x-raw(memory:NVMM), format=I420");
    g_object_set(G_OBJECT(caps), "caps", caps_filter, NULL);
    gst_caps_unref(caps_filter); // Release the caps filter reference after setting

    // 5. Set encoder bitrate and presets
    // Adjusts the video encoder's bitrate for output video quality and compression.
    // Preset levels are adjusted for performance, with specific settings for AArch64 architecture to optimize for hardware.
    g_object_set(G_OBJECT(encoder), "bitrate", bitrate, NULL);
    if (is_aarch64()) {
        g_object_set(G_OBJECT(encoder), "preset-level", 1, NULL);
        g_object_set(G_OBJECT(encoder), "insert-sps-pps", 1, NULL);
    }

    // 6. Configure the UDP sink (udpsink) parameters
    // Sets network parameters for the udpsink, including multicast address, port, and synchronization settings.
    // These are critical for ensuring the video stream is correctly sent over the network.
    g_object_set(G_OBJECT(sink), "host", "224.224.255.255", NULL);
    g_object_set(G_OBJECT(sink), "port", updsink_port_num, NULL);
    g_object_set(G_OBJECT(sink), "async", FALSE, NULL);
    g_object_set(G_OBJECT(sink), "sync", 1, NULL);

    // Add a message handler to the pipeline's bus
    // This allows for handling GStreamer messages, which can include errors, state changes, or custom application messages.
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus); // Release the bus reference after adding the watch

    /**
     * @brief Adds a series of elements to the GStreamer pipeline.
     */
    gst_bin_add_many(GST_BIN(pipeline),
                     source, streammux, pgie, nvtracker,
                     nvvidconv, nvosd, nvvidconv_postosd, caps, encoder, rtppay, sink, NULL);

    /**
     * @brief Links the source element to the stream multiplexer's sink pad.
     */
    GstPad *sinkpad, *srcpad;

    // Get the sink pad of the streammux element
    sinkpad = gst_element_get_request_pad(streammux, pad_name_sink);
    // Get the src pad of the source'bin element
    srcpad = gst_element_get_static_pad(source, pad_name_src);

    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link decoder to stream muxer. Exiting.\n");
        return -1;
    }
    // Release Sinkpad and SRCPAD
    gst_object_unref(sinkpad);
    gst_object_unref(srcpad);

    /**
     * @brief Links the GStreamer elements into a pipeline and sets up monitoring.
     */
    if (!gst_element_link_many(streammux, pgie, nvtracker,
                               nvvidconv, nvosd, nvvidconv_postosd, caps, encoder, rtppay, sink, NULL))
    {
        g_printerr("Elements could not be linked. Exiting.\n");
        return -1;
    }

    // Add probes to obtain metadata
    osd_sink_pad = gst_element_get_static_pad(nvtracker, "src"); // Get the sink pad of the nvosd element
    if (!osd_sink_pad) g_print("Unable to get sink pad\n");
    else
        // Parameters: pad, probe type, probe callback function, callback function parameters, callback function parameter release function
        gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_sink_pad_buffer_probe, NULL, NULL); // Add probe
    g_timeout_add(5000, perf_print_callback, &g_perf_data);                                                // Add a timer for printing performance data
    gst_object_unref(osd_sink_pad);

    /**
     * @brief Initializes and configures the RTSP server for streaming video.
     */
    GstRTSPServer *server;
    GstRTSPMountPoints *mounts;
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
    gst_element_set_state(pipeline, GST_STATE_NULL);  // Set pipeline status to NULL
    g_print("Deleting pipeline\n");
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);
    return 0;
}
