#include "deepstream_track.h"
#include "utils.h"


/**
 * @brief Sets the properties of the NvDsObjectMeta object.
 * 
 * This function sets the border color, background color, display text, and font size
 * of the given NvDsObjectMeta object.
 * 
 * @param obj_meta The NvDsObjectMeta object to set the properties for.
 * @param border_color An array of three floats representing the RGB values of the border color.
 * @param bg_color An array of four floats representing the RGBA values of the background color.
 * @param label The label of the object.
 * @param track_id The ID of the object's track.
 * @param confidence The confidence score of the object.
 */
static void set_object_meta_properties(NvDsObjectMeta *obj_meta, float border_color[3], float bg_color[4], const char* label, int track_id, float confidence) {
    // Ensure that display_text has been allocated or has sufficient space
    if (obj_meta->text_params.display_text == NULL) {
        obj_meta->text_params.display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
    }

    // Set border and background colors
    obj_meta->rect_params.border_color.red = border_color[0];
    obj_meta->rect_params.border_color.green = border_color[1];
    obj_meta->rect_params.border_color.blue = border_color[2];
    obj_meta->rect_params.has_bg_color = 1;
    obj_meta->rect_params.bg_color.red = bg_color[0];
    obj_meta->rect_params.bg_color.green = bg_color[1];
    obj_meta->rect_params.bg_color.blue = bg_color[2];
    obj_meta->rect_params.bg_color.alpha = bg_color[3];

    // Format and set the displayed text to "label: confidence, track_id"
    snprintf(obj_meta->text_params.display_text, MAX_DISPLAY_LEN, "%s: %.2f, %d", label, confidence, track_id);
    obj_meta->text_params.font_params.font_size = 20; 
}



/**
 * @brief Updates the display metadata with the person count and vehicle count.
 *
 * This function initializes the first text_params in the display_meta and sets the display text
 * to show the person count and vehicle count. It also sets the font parameters and background color
 * for the text.
 *
 * @param display_meta A pointer to the NvDsDisplayMeta structure representing the display metadata.
 * @param person_count The number of detected persons.
 * @param vehicle_count The number of detected vehicles.
 */
static void update_display_meta(NvDsDisplayMeta *display_meta, guint person_count, guint vehicle_count) {
    // Initialize the first text_params in the display_meta
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;

    // Allocate or ensure memory for display_text
    if (txt_params->display_text == NULL) {
        txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
    }

    // Set the display text
    snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person Count = %d Vehicle Count = %d", person_count, vehicle_count);
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
}

/**
 * @brief The callback function for the pad probe on the sink pad of the OSD element.
 *
 * This function is called when a buffer is received on the sink pad of the OSD element.
 * It processes the buffer and updates the object metadata properties and display metadata.
 *
 * @param pad The sink pad on which the probe is installed.
 * @param info The probe information containing the buffer data.
 * @param u_data The user data passed to the probe function.
 * @return The return value indicating the status of the probe function.
 */
static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                                                   gpointer u_data) {
    GstBuffer *buf = GST_BUFFER(info->data);
    guint vehicle_count = 0, person_count = 0;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);

            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
                float border_color[] = {0.0, 0.0, 1.0};
                float bg_color[] = {0.0, 0.0, 1.0, 0.2};
                // Call function, pass category, confidence, and tracking ID
                set_object_meta_properties(obj_meta, border_color, bg_color, obj_meta->obj_label, obj_meta->object_id, obj_meta->confidence);
                vehicle_count++;
            } else if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                float border_color[] = {1.0, 0.0, 1.0};
                float bg_color[] = {1.0, 0.0, 1.0, 0.2};
                // ditto
                set_object_meta_properties(obj_meta, border_color, bg_color, obj_meta->obj_label, obj_meta->object_id, obj_meta->confidence);
                person_count++;
            }
        }

        NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        update_display_meta(display_meta, person_count, vehicle_count);
        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }

    // Example logging, consider adjusting based on actual needs
    // g_print("Frame Processed: Person Count = %d, Vehicle Count = %d\n", person_count, vehicle_count);
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

    int current_device             = -1;
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

    sinkpad = gst_element_get_request_pad(streammux, pad_name_sink);
    srcpad = gst_element_get_static_pad(source, pad_name_src);

    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK)
    {
        g_printerr("Failed to link decoder to stream muxer. Exiting.\n");
        return -1;
    }
    gst_object_unref(sinkpad);
    gst_object_unref(srcpad);

    
    if (!gst_element_link_many(streammux, pgie, nvtracker,
                               nvvidconv, nvosd, nvvidconv_postosd, caps, encoder, rtppay, sink, NULL)) {
        g_printerr("Elements could not be linked. Exiting.\n");
        return -1;
    }

    osd_sink_pad = gst_element_get_static_pad(nvosd, "sink"); 
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