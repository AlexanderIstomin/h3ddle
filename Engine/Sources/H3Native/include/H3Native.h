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

/* Stable Audio 3 sound effects. The package is loaded and released per
 * call: at 1.7 GB it is cheap to reload, and holding it resident would
 * compete with H3 for the memory that actually constrains this app.
 *
 * `on_step` may be NULL; it fires once per denoising pass. Writes a
 * 16-bit stereo WAV at 44.1 kHz to `output_path`. */
typedef void (*h3ddle_sa3_step)(int completed, int total, void *opaque);
/* Releases any cached sound-effect package. Called when the video model is
 * about to load, so the two never occupy memory at once. */
void h3ddle_sa3_release(void);
int h3ddle_sa3_generate(const char *package_directory, const char *prompt,
                        double seconds, int steps, unsigned long long seed,
                        const char *output_path, h3ddle_sa3_step on_step,
                        void *opaque, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
