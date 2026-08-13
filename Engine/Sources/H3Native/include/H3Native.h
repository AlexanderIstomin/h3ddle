#ifndef H3DDLE_H3_NATIVE_H
#define H3DDLE_H3_NATIVE_H

#include "../../../Vendor/h3.c/h3.h"

#ifdef __cplusplus
extern "C" {
#endif

h3_params h3ddle_h3_default_params(void);
int h3ddle_h3_frames_for_seconds(double seconds);
int h3ddle_h3_prepare_metal(const char *shader_source_path, char *error,
                            size_t error_size);
const char *h3ddle_h3_version(void);
const char *h3ddle_h3_device_name(const h3_device_info *device);
const char *h3ddle_h3_device_architecture(const h3_device_info *device);
const char *h3ddle_h3_model_layout_name(const h3_model_info *model);
int h3ddle_h3_model_supports_generation(const h3_model_info *model);

#ifdef __cplusplus
}
#endif

#endif
