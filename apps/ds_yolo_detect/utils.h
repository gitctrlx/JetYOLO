#ifndef UTILS_H
#define UTILS_H

#include <gst/gst.h>

/* Tracker config parsing */

#define CHECK_ERROR(error)                                               \
  if (error)                                                             \
  {                                                                      \
    g_printerr("Error while parsing config file: %s\n", error->message); \
    goto done;                                                           \
  }

bool is_aarch64()
{
#if defined(__aarch64__)
  return true;
#else
  return false;
#endif
}

typedef struct
{
  guint64 n_frames;
  guint64 last_fps_update_time;
  gdouble fps;
} PERF_DATA;
PERF_DATA g_perf_data = {0, 0, 0.0};

/**
 * @brief A boolean data type in GLib.
 *
 * gboolean is a data type in GLib that represents a boolean value. It can have two possible values: TRUE or FALSE.
 * In this context, gboolean is used as the return type of the perf_print_callback function.
 *
 * @param user_data A pointer to user-defined data that is passed to the perf_print_callback function.
 * @return TRUE if the callback should continue to be called, FALSE otherwise.
 */
gboolean perf_print_callback(gpointer user_data)
{
    PERF_DATA *perf_data = (PERF_DATA *)user_data;
    guint64 current_time = g_get_monotonic_time();
    guint64 time_elapsed = current_time - perf_data->last_fps_update_time;

    if (time_elapsed > 0)
    {
        perf_data->fps = 1000000.0 * perf_data->n_frames / time_elapsed;
        g_print("FPS: %0.2f\n", perf_data->fps);
        perf_data->n_frames = 0;
        perf_data->last_fps_update_time = current_time;
    }

    return G_SOURCE_CONTINUE;
}

void update_frame_counter()
{
  g_perf_data.n_frames++;
}

/**
 * @brief Represents a character string in GLib.
 *
 * The `gchar` type is a typedef for `char` in GLib. It is used to represent a character string.
 * It is commonly used in GLib-based libraries and applications.
 */
static gchar *get_absolute_file_path(gchar *cfg_file_path, gchar *file_path) {
    // Directly return if file_path is already an absolute path
    if (file_path && file_path[0] == '/') {
        return g_strdup(file_path); // Ensure ownership transfer by duplicating the string
    }

    // Get the absolute path of the configuration file
    gchar *abs_cfg_path = static_cast<gchar*>(g_malloc0(PATH_MAX + 1));
    if (!realpath(cfg_file_path, abs_cfg_path)) {
        g_free(file_path);
        g_free(abs_cfg_path); // Ensure to free allocated memory to avoid leaks
        return NULL;
    }

    // If file_path is NULL, return the directory of cfg_file_path
    if (!file_path) {
        gchar *last_slash = g_strrstr(abs_cfg_path, "/");
        if (last_slash != NULL) {
            *(last_slash + 1) = '\0'; // Null-terminate the string at the last slash
        }
        return abs_cfg_path; // Transfer ownership of abs_cfg_path directly
    }

    // Concatenate to form the absolute file path
    gchar *abs_file_path = g_strconcat(abs_cfg_path, file_path, NULL);
    g_free(abs_cfg_path); // Free the temporary absolute path buffer
    g_free(file_path); // The caller expects file_path to be freed or returned

    return abs_file_path;
}


/**
 * @brief A boolean data type in GLib.
 *
 * gboolean is a data type in GLib that represents a boolean value. It can have two possible values: TRUE or FALSE.
 * In this context, gboolean is used as the return type of the bus_call function.
 *
 * @param bus The GstBus object.
 * @param msg The GstMessage object.
 * @param data A pointer to user data.
 * @return TRUE to keep the callback active.
 */
static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    GMainLoop *loop = (GMainLoop *)data;

    // Handle different message types with a switch case
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop);  // Exit the main loop on EOS
            break;
        case GST_MESSAGE_ERROR: {
            gchar *debug = NULL;
            GError *error = NULL;

            gst_message_parse_error(msg, &error, &debug);  // Parse error message
            g_printerr("ERROR from element %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
            if (debug) {
                g_printerr("Error details: %s\n", debug);
                g_free(debug);  // Free debug information only if it exists
            }
            g_error_free(error);  // Always free the error
            g_main_loop_quit(loop);  // Exit the main loop on error
            break;
        }
        default:
            // No action required for other message types
            break;
    }
    return TRUE;  // Keep the callback active
}


#endif // UTILS_H
