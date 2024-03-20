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

// gie config
#define PGIE_CONFIG_FILE "./configs/ds_yolo_tracker_config/pgie_config.txt"
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

gint frame_number = 0;

#endif