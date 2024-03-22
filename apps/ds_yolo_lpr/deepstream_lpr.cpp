#include "deepstream_lpr.h"
#include "utils.h"


static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                                                   gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *)info->data;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  guint plate_count = 0;
  guint label_i = 0;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_class = NULL;
  NvDsMetaList *l_label = NULL;
  NvDsClassifierMeta *class_meta = NULL;
  NvDsLabelInfo *label_info = NULL;
  NvDsDisplayMeta *display_meta = NULL;
  guint plate_str_len = 0;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf); // 获取批处理元数据
  // 遍历批处理元数据，得到每一帧的元数据
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    // 获取每一帧的元数据
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    int offset = 0;
    // 遍历每一帧的元数据，得到每一个检测到的物体的元数据
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next)
    {
      // 获取每一个检测到的物体的元数据
      obj_meta = (NvDsObjectMeta *)(l_obj->data);
      // 设置文字内容
      obj_meta->text_params.display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);

     
      // 车牌检测
      if (obj_meta->unique_component_id == 1)
      {
        // 检测框边框
        obj_meta->rect_params.border_color.red = 1.0;
        obj_meta->rect_params.border_color.green = 0.0;
        obj_meta->rect_params.border_color.blue = 1.0;
        obj_meta->rect_params.border_width = 1;
        // 背景颜色
        obj_meta->rect_params.has_bg_color = 1;
        obj_meta->rect_params.bg_color.red = 1.0;
        obj_meta->rect_params.bg_color.green = 0.0;
        obj_meta->rect_params.bg_color.blue = 1.0;
        obj_meta->rect_params.bg_color.alpha = 0.1;

        // 设置文字背景透明
        obj_meta->text_params.set_bg_clr = 0;
      }

      // 车牌识别结果（分类结果）
      for (l_class = obj_meta->classifier_meta_list; l_class != NULL; l_class = l_class->next)
      {
        class_meta = (NvDsClassifierMeta *)(l_class->data);

        if (class_meta->unique_component_id == 3) // unique_component_id = 3 表示来自于车牌识别模型
        {
          // 遍历每一个分类结果，label_info_list是一个链表，每一个节点是一个NvDsLabelInfo结构体：https://docs.nvidia.com/metropolis/deepstream/sdk-api/struct__NvDsLabelInfo.html
          for (label_i = 0, l_label = class_meta->label_info_list; label_i < class_meta->num_labels && l_label; label_i++, l_label = l_label->next)
          {
            label_info = (NvDsLabelInfo *)(l_label->data);
            // 获取车牌字符长度
            plate_str_len = strlen(label_info->result_label);
            // 如果是车牌识别结果，且置信度大于0.5，且字符长度大于9
            if (label_info->label_id == 0 && label_info->result_class_id == 1 && label_info->result_prob > 0.2 && plate_str_len >= 9)
            {
              // 设置文字格式
              obj_meta->text_params.set_bg_clr = 1;
              obj_meta->text_params.text_bg_clr.red = 0.0;
              obj_meta->text_params.text_bg_clr.green = 0.0;
              obj_meta->text_params.text_bg_clr.blue = 0.0;
              obj_meta->text_params.text_bg_clr.alpha = 0.8;

              obj_meta->text_params.font_params.font_color.red = 1.0;
              obj_meta->text_params.font_params.font_color.green = 1.0;
              obj_meta->text_params.font_params.font_color.blue = .0;
              obj_meta->text_params.font_params.font_size = 15;
              // 设置文字位置
              obj_meta->text_params.x_offset = obj_meta->rect_params.left;
              obj_meta->text_params.y_offset = obj_meta->rect_params.top - 30;

              //  设置显示的文字
              // printf("Plate License %s, probability: %0.1f %% , len: %d\n", label_info->result_label, label_info->result_prob * 100, plate_str_len);
              snprintf(obj_meta->text_params.display_text, MAX_DISPLAY_LEN, "%s  %0.1f%%", label_info->result_label, label_info->result_prob * 100);

              plate_count++;
            }
          }
        }
      }
    }
    // 获取显示元数据，用于在屏幕上绘制多边形
    display_meta = nvds_acquire_display_meta_from_pool(batch_meta);

    // 添加文字
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
    offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Numberplate = %d ", plate_count);
    // offset += snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN, "车牌 = %d ", plate_count);

    // 设置文字的位置
    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    // 字体
    txt_params->font_params.font_name = "Serif";
    txt_params->font_params.font_size = 20;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    // 背景颜色
    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;

    // 添加显示
    nvds_add_display_meta_to_frame(frame_meta, display_meta);
  }

#if 0
    g_print ("Frame Number = %d Number of objects = %d "
            "Vehicle Count = %d Person Count = %d\n",
            frame_number, num_rects, vehicle_count, person_count);
#endif
  frame_number++;
  update_frame_counter();
  return GST_PAD_PROBE_OK;
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

// main函数
int main(int argc, char *argv[])
{
    GMainLoop   *loop              = NULL;
    GstElement  *pipeline          = NULL, 
                *source            = NULL, 
                *streammux         = NULL, 
                *pgie              = NULL,
                *secondary_classifier, 
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
    secondary_classifier = gst_element_factory_make("nvinfer", "secondary-infer-engine1"); 
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

    /**
     * @brief Configures parameters for various elements in the GStreamer pipeline.
     *
     * This section sets up essential properties for the elements within the pipeline to ensure
     * correct operation and optimization of the video processing and streaming workflow.
     */
    g_object_set(G_OBJECT(streammux), "batch-size", 1, NULL);
    g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
                MUXER_OUTPUT_HEIGHT,
                "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    g_object_set(G_OBJECT(pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);

    g_object_set(G_OBJECT(secondary_classifier), "config-file-path", THRID_CONFIG_FILE, "unique-id", 3, NULL);

    caps_filter = gst_caps_from_string("video/x-raw(memory:NVMM), format=I420");
    g_object_set(G_OBJECT(caps), "caps", caps_filter, NULL);
    gst_caps_unref(caps_filter);

    g_object_set(G_OBJECT(encoder), "bitrate", bitrate, NULL);
    if (is_aarch64()){
      g_object_set(G_OBJECT(encoder), "preset-level", 1, NULL);
      g_object_set(G_OBJECT(encoder), "insert-sps-pps", 1, NULL);
    }

    g_object_set(G_OBJECT(sink), "host", "224.224.255.255", NULL);
    g_object_set(G_OBJECT(sink), "port", updsink_port_num, NULL);
    g_object_set(G_OBJECT(sink), "async", FALSE, NULL);
    g_object_set(G_OBJECT(sink), "sync", 1, NULL);

    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    GstElement *queue1 = NULL, *queue2 = NULL;

    queue1 = gst_element_factory_make ("queue", "queue1");
    queue2 = gst_element_factory_make ("queue", "queue2");

    /**
     * @brief Adds a series of elements to the GStreamer pipeline.
     */
    gst_bin_add_many(GST_BIN(pipeline), queue1, queue2,
                   source, streammux, pgie,  secondary_classifier,
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

    
    if (!gst_element_link_many(streammux, pgie,  queue1, secondary_classifier, queue2,
                             nvvidconv, nvosd, nvvidconv_postosd, caps, encoder, rtppay, sink, NULL))
    {
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
