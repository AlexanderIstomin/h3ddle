#include "H3Native.h"
#include "h3_gpu.h"
#include "h3_host.h"

#include <limits.h>
#include <math.h>

h3_params h3ddle_h3_default_params(void) {
    h3_params params = H3_PARAMS_DEFAULT;
    return params;
}

int h3ddle_h3_prepare_metal(const char *shader_source_path, char *error,
                            size_t error_size) {
    return h3_gpu_prepare(
        shader_source_path && *shader_source_path ? shader_source_path
                                                  : "h3_shaders.metal",
        error, error_size);
}

int h3ddle_h3_frames_for_seconds(double seconds) {
    if (!isfinite(seconds) || seconds <= 0.0) return H3_DEFAULT_FRAMES;
    double requested = ceil(seconds * H3_FPS);
    if (requested > INT_MAX) requested = INT_MAX;
    return h3_align_frame_count((int)requested);
}

const char *h3ddle_h3_version(void) {
    return H3_VERSION;
}

const char *h3ddle_h3_device_name(const h3_device_info *device) {
    return device ? device->name : "";
}

const char *h3ddle_h3_device_architecture(const h3_device_info *device) {
    return device ? device->architecture : "";
}

const char *h3ddle_h3_model_layout_name(const h3_model_info *model) {
    if (!model) return "unknown";
    switch (model->layout) {
        case H3_MODEL_LAYOUT_RELEASED_DIRECTORY: return "releasedDirectory";
        case H3_MODEL_LAYOUT_OPTIMIZED_INT8_SINGLE_FILE:
            return "optimizedINT8SingleFile";
        case H3_MODEL_LAYOUT_UNKNOWN: return "unknown";
    }
    return "unknown";
}

int h3ddle_h3_model_supports_generation(const h3_model_info *model) {
    return model ? model->generation_supported : 0;
}
