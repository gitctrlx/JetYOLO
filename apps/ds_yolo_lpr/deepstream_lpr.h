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

#include "utils.h"

guint        bitrate           = 5000000;      // bit rate
gchar       *codec             = "H264";       // Set encoding format
guint        updsink_port_num  = 5400;         // Set port number
guint        rtsp_port_num     = 8554;         // Set RTSP port number
gchar       *rtsp_path         = "/ds-test";   // Set RTSP path

gchar        pad_name_sink[16] = "sink_0";
gchar        pad_name_src[16]  = "src";

// gie config
#define PGIE_CONFIG_FILE "./configs/ds_yolo_lpr_config/pgie_config.txt"
#define THRID_CONFIG_FILE "./configs/ds_yolo_lpr_config/lpr_config_sgie_ch.yml"
#define MAX_DISPLAY_LEN 64

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

#endif