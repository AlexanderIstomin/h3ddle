#include "H3Native.h"
#include "h3_gpu.h"
#include "h3_host.h"
#include "sa3_generate.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* Writes 16-bit stereo PCM. The samples arrive channel-major and unclamped;
 * clipping rather than normalising matches the reference, which would
 * otherwise quietly change the level of every generation. */
static int sa3_write_wav(const char *path, const float *audio, int samples) {
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    unsigned int rate = 44100, data_bytes = (unsigned int)samples * 4u;
    unsigned int header[] = {36u + data_bytes, 16u, rate, rate * 4u, data_bytes};
    unsigned short fields[] = {1, 2, 16, 4};
    int ok = fwrite("RIFF", 1, 4, file) == 4;
    ok = ok && fwrite(&header[0], 4, 1, file) == 1;
    ok = ok && fwrite("WAVEfmt ", 1, 8, file) == 8;
    ok = ok && fwrite(&header[1], 4, 1, file) == 1;
    ok = ok && fwrite(&fields[0], 2, 1, file) == 1;   /* PCM */
    ok = ok && fwrite(&fields[1], 2, 1, file) == 1;   /* stereo */
    ok = ok && fwrite(&header[2], 4, 1, file) == 1;
    ok = ok && fwrite(&header[3], 4, 1, file) == 1;
    ok = ok && fwrite(&fields[3], 2, 1, file) == 1;   /* block align */
    ok = ok && fwrite(&fields[2], 2, 1, file) == 1;   /* bits */
    ok = ok && fwrite("data", 1, 4, file) == 4;
    ok = ok && fwrite(&header[4], 4, 1, file) == 1;
    for (int index = 0; ok && index < samples; index++)
        for (int channel = 0; channel < 2; channel++) {
            float value = audio[(size_t)channel * samples + index];
            if (value > 1.0f) value = 1.0f;
            if (value < -1.0f) value = -1.0f;
            short pcm = (short)(value * 32767.0f);
            ok = fwrite(&pcm, 2, 1, file) == 1;
        }
    fclose(file);
    return ok;
}

/* The package is kept between generations so a run of effects pays the
 * 1.8-second load once instead of every time. It is dropped the moment the
 * video model wants memory: at tens of gigabytes that model decides what
 * fits, and 1.7 GB of sound effects is not worth crowding it. */
static sa3 *sa3_cached;
static char sa3_cached_directory[1024];
static h3_gpu *sa3_cached_gpu;

void h3ddle_sa3_release(void) {
    if (!sa3_cached) return;
    sa3_free(sa3_cached);
    h3_gpu_free(sa3_cached_gpu);
    sa3_cached = NULL;
    sa3_cached_gpu = NULL;
    sa3_cached_directory[0] = '\0';
}

int h3ddle_sa3_generate(const char *package_directory, const char *prompt,
                        double seconds, int steps, unsigned long long seed,
                        const char *output_path, h3ddle_sa3_step on_step,
                        void *opaque, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!package_directory || !prompt || !output_path) {
        if (error && error_size)
            snprintf(error, error_size, "a package, prompt and destination are "
                     "required");
        return 0;
    }
    h3_gpu *gpu = sa3_cached_gpu;
    sa3 *model = sa3_cached;
    if (!model || strcmp(sa3_cached_directory, package_directory)) {
        h3ddle_sa3_release();
        /* Sharing h3.c's Metal context keeps both models on one device and
         * one allocator; without it the transformer falls back to the CPU. */
        gpu = h3_gpu_create(NULL, error, error_size);
        model = sa3_load(package_directory, gpu, error, error_size);
        if (!model) {
            h3_gpu_free(gpu);
            return 0;
        }
        sa3_cached = model;
        sa3_cached_gpu = gpu;
        snprintf(sa3_cached_directory, sizeof(sa3_cached_directory), "%s",
                 package_directory);
    }

    sa3_request request = {
        .prompt = prompt,
        .seconds = (float)seconds,
        .steps = steps,
        .seed = seed,
    };
    float *audio = NULL;
    int samples = 0;
    int ok = sa3_generate(model, &request, on_step, opaque, &audio, &samples,
                          error, error_size);
    if (ok) {
        ok = sa3_write_wav(output_path, audio, samples);
        if (!ok && error && error_size)
            snprintf(error, error_size, "cannot write %s", output_path);
    }
    free(audio);
    return ok;
}
