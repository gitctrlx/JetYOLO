#ifndef __DEEPSTREAM_H__
#define __DEEPSTREAM_H__

#include <gst/gst.h>
#include <glib.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include "nvds_yml_parser.h"

#include "gstnvdsmeta.h"
#include "nvds_analytics_meta.h"
#include <gst/rtsp-server/rtsp-server.h>
#include "task/border_cross.h"
#include "task/gather.h"
#include <sstream>
#include <fstream>
#include <cmath>

#include "utils.h"

guint        bitrate           = 5000000;      // bit rate
gchar       *codec             = "H264";       // Set encoding format
guint        updsink_port_num  = 5400;         // Set port number
guint        rtsp_port_num     = 8554;         // Set RTSP port number
gchar       *rtsp_path         = "/ds-test";   // Set RTSP path

SOURCE_INFO  g_source_info[2]; // Initialize to two original inputs

gchar        pad_name_sink[16] = "sink_0";
gchar        pad_name_src[16]  = "src";

// gie config
#define PGIE_CONFIG_FILE "./configs/ds_yolo_tracker_config/pgie_config.txt"
#define MAX_DISPLAY_LEN 64

// gie-tracker config
#define TRACKER_CONFIG_FILE "./configs/ds_yolo_tracker_config/tracker_config.txt"
#define MAX_TRACKING_ID_LEN 16

#define PGIE_CLASS_ID_VEHICLE 2
#define PGIE_CLASS_ID_PERSON 0

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

/* APP config */
#define POLYGON_FILE_PATH "./configs/ds_yolo_tracker_config/polygon_1.txt"

/* Tracker config parsing */
#define CONFIG_GROUP_TRACKER "tracker"
#define CONFIG_GROUP_TRACKER_WIDTH "tracker-width"
#define CONFIG_GROUP_TRACKER_HEIGHT "tracker-height"
#define CONFIG_GROUP_TRACKER_LL_CONFIG_FILE "ll-config-file"
#define CONFIG_GROUP_TRACKER_LL_LIB_FILE "ll-lib-file"
#define CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS "enable-batch-process"
#define CONFIG_GPU_ID "gpu-id"

#endif