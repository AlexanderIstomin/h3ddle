#include "zimage_tae.h"

#include "h3_gpu.h"
#include "h3_weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/* madebyollin/taef1, as represented by Diffusers' DecoderTiny. Three nearest
 * 2x upsamples turn a 16-channel FLUX latent into an 8x RGB canvas:
 *
 *   0: conv 16->64 + relu       2,3,4: residual blocks
 *   5: up   6: conv             7,8,9: residual blocks
 *  10: up  11: conv            12,13,14: residual blocks
 *  15: up  16: conv            17: residual block   18: conv 64->3
 *
 * Parameter-free activations and upsample layers account for the gaps in the
 * safetensors indices. Activations stay NHWC, which is the layout consumed by
 * the shared MPSGraph convolution and the interleaved layout the service uses.
 */

#define TAEF1_LATENT_CHANNELS 16
#define TAEF1_CHANNELS        64
#define TAEF1_SPATIAL_RATIO   8

typedef struct {
    h3_gpu_tensor *weight;
    h3_gpu_tensor *bias;
} taef1_conv;

typedef struct {
    taef1_conv conv0;
    taef1_conv conv2;
    taef1_conv conv4;
} taef1_block;

struct zimage_tae {
    h3_gpu *gpu;
    h3_weight_store *store;
    taef1_conv conv_in;
    taef1_block blocks_a[3];
    taef1_conv conv_up1;
    taef1_block blocks_b[3];
    taef1_conv conv_up2;
    taef1_block blocks_c[3];
    taef1_conv conv_up3;
    taef1_block block_d;
    taef1_conv conv_out;
    h3_gpu_tensor *work[3];
    int latent_height;
    int latent_width;
    const char *stage;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int load_conv(zimage_tae *tae, taef1_conv *conv, int index,
                     int input_channels, int output_channels, int has_bias,
                     char *error, size_t error_size) {
    char name[96];
    const uint64_t weight_shape[] = {
        (uint64_t)output_channels, (uint64_t)input_channels, 3, 3
    };
    snprintf(name, sizeof(name), "decoder.layers.%d.weight", index);
    conv->weight = h3_weight_load_f32(tae->store, tae->gpu, name, 4,
                                      weight_shape, error, error_size);
    if (!conv->weight) return 0;
    conv->bias = NULL;
    if (has_bias) {
        const uint64_t bias_shape[] = {(uint64_t)output_channels};
        snprintf(name, sizeof(name), "decoder.layers.%d.bias", index);
        conv->bias = h3_weight_load_f32(tae->store, tae->gpu, name, 1,
                                        bias_shape, error, error_size);
        if (!conv->bias) return 0;
    }
    return 1;
}

static int load_block(zimage_tae *tae, taef1_block *block, int index,
                      char *error, size_t error_size) {
    char name[96];
    const uint64_t weight_shape[] = {
        TAEF1_CHANNELS, TAEF1_CHANNELS, 3, 3
    };
    const uint64_t bias_shape[] = {TAEF1_CHANNELS};
    taef1_conv *convs[] = {&block->conv0, &block->conv2, &block->conv4};
    const int layers[] = {0, 2, 4};
    for (int position = 0; position < 3; position++) {
        snprintf(name, sizeof(name), "decoder.layers.%d.conv.%d.weight",
                 index, layers[position]);
        convs[position]->weight = h3_weight_load_f32(
            tae->store, tae->gpu, name, 4, weight_shape, error, error_size);
        if (!convs[position]->weight) return 0;
        snprintf(name, sizeof(name), "decoder.layers.%d.conv.%d.bias",
                 index, layers[position]);
        convs[position]->bias = h3_weight_load_f32(
            tae->store, tae->gpu, name, 1, bias_shape, error, error_size);
        if (!convs[position]->bias) return 0;
    }
    return 1;
}

static int allocate_work(zimage_tae *tae, char *error, size_t error_size) {
    const size_t canvas =
        (size_t)tae->latent_height * TAEF1_SPATIAL_RATIO *
        (size_t)tae->latent_width * TAEF1_SPATIAL_RATIO;
    const size_t elements = canvas * TAEF1_CHANNELS;
    for (int index = 0; index < 3; index++) {
        tae->work[index] = h3_gpu_tensor_new_f32(tae->gpu, elements);
        if (!tae->work[index]) {
            fail(error, error_size, "out of memory for Z-Image preview buffers");
            return 0;
        }
    }
    return 1;
}

zimage_tae *zimage_tae_create(const char *shader_source_path,
                              const char *weight_path,
                              int latent_height, int latent_width,
                              char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!shader_source_path || !*shader_source_path || !weight_path ||
        latent_height < 1 || latent_width < 1) {
        fail(error, error_size, "invalid Z-Image preview decoder arguments");
        return NULL;
    }
    zimage_tae *tae = calloc(1, sizeof(*tae));
    if (!tae) {
        fail(error, error_size, "out of memory for the Z-Image preview decoder");
        return NULL;
    }
    tae->latent_height = latent_height;
    tae->latent_width = latent_width;
    tae->gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (tae->gpu)
        h3_gpu_profile_set_label(tae->gpu, "Z-Image TAEF1 preview decoder");
    if (tae->gpu)
        tae->store = h3_weight_store_open(weight_path, error, error_size);
    int ok = tae->gpu && tae->store &&
        load_conv(tae, &tae->conv_in, 0, TAEF1_LATENT_CHANNELS,
                  TAEF1_CHANNELS, 1, error, error_size) &&
        load_block(tae, &tae->blocks_a[0], 2, error, error_size) &&
        load_block(tae, &tae->blocks_a[1], 3, error, error_size) &&
        load_block(tae, &tae->blocks_a[2], 4, error, error_size) &&
        load_conv(tae, &tae->conv_up1, 6, TAEF1_CHANNELS,
                  TAEF1_CHANNELS, 0, error, error_size) &&
        load_block(tae, &tae->blocks_b[0], 7, error, error_size) &&
        load_block(tae, &tae->blocks_b[1], 8, error, error_size) &&
        load_block(tae, &tae->blocks_b[2], 9, error, error_size) &&
        load_conv(tae, &tae->conv_up2, 11, TAEF1_CHANNELS,
                  TAEF1_CHANNELS, 0, error, error_size) &&
        load_block(tae, &tae->blocks_c[0], 12, error, error_size) &&
        load_block(tae, &tae->blocks_c[1], 13, error, error_size) &&
        load_block(tae, &tae->blocks_c[2], 14, error, error_size) &&
        load_conv(tae, &tae->conv_up3, 16, TAEF1_CHANNELS,
                  TAEF1_CHANNELS, 0, error, error_size) &&
        load_block(tae, &tae->block_d, 17, error, error_size) &&
        load_conv(tae, &tae->conv_out, 18, TAEF1_CHANNELS, 3, 1,
                  error, error_size) &&
        allocate_work(tae, error, error_size);
    if (!ok) {
        zimage_tae_release(tae);
        return NULL;
    }
    return tae;
}

void zimage_tae_release(zimage_tae *tae) {
    if (!tae) return;
    for (int index = 0; index < 3; index++)
        h3_gpu_tensor_free(tae->work[index]);
    h3_weight_store_free(tae->store);
    h3_gpu_free(tae->gpu);
    free(tae);
}

static int run_conv(zimage_tae *tae, const taef1_conv *conv,
                    h3_gpu_tensor *output, const h3_gpu_tensor *input,
                    int height, int width, int input_channels,
                    int output_channels) {
    return h3_gpu_conv3d_same_f32(
        tae->gpu, output, input, conv->weight, conv->bias, 1, 1,
        (uint32_t)height, (uint32_t)width, (uint32_t)input_channels,
        (uint32_t)output_channels, 1, 3, 3, 1, 1, 1);
}

static int run_block(zimage_tae *tae, const taef1_block *block, int a,
                     int height, int width) {
    h3_gpu_tensor *input = tae->work[a];
    h3_gpu_tensor *first = tae->work[(a + 1) % 3];
    h3_gpu_tensor *second = tae->work[(a + 2) % 3];
    const uint32_t elements =
        (uint32_t)height * (uint32_t)width * TAEF1_CHANNELS;
    if (!run_conv(tae, &block->conv0, first, input, height, width,
                  TAEF1_CHANNELS, TAEF1_CHANNELS) ||
        !h3_gpu_relu_f32(tae->gpu, first, first, elements) ||
        !run_conv(tae, &block->conv2, second, first, height, width,
                  TAEF1_CHANNELS, TAEF1_CHANNELS) ||
        !h3_gpu_relu_f32(tae->gpu, second, second, elements) ||
        !run_conv(tae, &block->conv4, first, second, height, width,
                  TAEF1_CHANNELS, TAEF1_CHANNELS))
        return 0;
    return h3_gpu_add_scaled_f32(tae->gpu, tae->work[a], first, input,
                                 1.0f, 1.0f, elements) &&
           h3_gpu_relu_f32(tae->gpu, tae->work[a], tae->work[a], elements);
}

static int run_upsample(zimage_tae *tae, int *a, int *height, int *width) {
    const int next = (*a + 1) % 3;
    if (!h3_gpu_nearest2x_nhwc_f32(
            tae->gpu, tae->work[next], tae->work[*a],
            (uint32_t)*height, (uint32_t)*width, TAEF1_CHANNELS))
        return 0;
    *a = next;
    *height *= 2;
    *width *= 2;
    return 1;
}

static int run_group(zimage_tae *tae, const taef1_block *blocks,
                     int block_count, const taef1_conv *upsample_conv,
                     int *a, int *height, int *width) {
    for (int index = 0; index < block_count; index++)
        if (!run_block(tae, &blocks[index], *a, *height, *width)) return 0;
    if (!upsample_conv) return 1;
    if (!run_upsample(tae, a, height, width)) return 0;
    const int next = (*a + 1) % 3;
    if (!run_conv(tae, upsample_conv, tae->work[next], tae->work[*a],
                  *height, *width, TAEF1_CHANNELS, TAEF1_CHANNELS))
        return 0;
    *a = next;
    return 1;
}

int zimage_tae_decode(zimage_tae *tae, const float *latent, float *rgb,
                      char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tae || !latent || !rgb) {
        fail(error, error_size, "invalid Z-Image preview decode arguments");
        return 0;
    }
    int height = tae->latent_height;
    int width = tae->latent_width;
    const size_t plane = (size_t)height * (size_t)width;
    float *nhwc = malloc(plane * TAEF1_LATENT_CHANNELS * sizeof(*nhwc));
    if (!nhwc) {
        fail(error, error_size, "out of memory staging the Z-Image preview");
        return 0;
    }
    /* Diffusers' DecoderTiny applies this soft clamp before its first layer.
     * TAEF1 has scaling_factor 1 and shift_factor 0, so the transformer's
     * current diffusion-space latent otherwise goes in unchanged. */
    for (int channel = 0; channel < TAEF1_LATENT_CHANNELS; channel++)
        for (size_t pixel = 0; pixel < plane; pixel++)
            nhwc[pixel * TAEF1_LATENT_CHANNELS + channel] =
                3.0f * tanhf(latent[(size_t)channel * plane + pixel] / 3.0f);
    const int uploaded = h3_gpu_tensor_write_f32(
        tae->work[0], nhwc, plane * TAEF1_LATENT_CHANNELS);
    free(nhwc);
    if (!uploaded) {
        fail(error, error_size, "cannot upload the Z-Image preview latent");
        return 0;
    }
    if (!h3_gpu_begin(tae->gpu)) {
        fail(error, error_size, "cannot begin the Z-Image preview command: %s",
             h3_gpu_error(tae->gpu));
        return 0;
    }

    int a = 0;
    int next = 1;
    tae->stage = "the input convolution";
    int ok = run_conv(tae, &tae->conv_in, tae->work[next], tae->work[a],
                      height, width, TAEF1_LATENT_CHANNELS, TAEF1_CHANNELS) &&
             h3_gpu_relu_f32(
                 tae->gpu, tae->work[next], tae->work[next],
                 (uint32_t)(plane * TAEF1_CHANNELS));
    a = next;
    if (ok) tae->stage = "the first block group";
    ok = ok && run_group(tae, tae->blocks_a, 3, &tae->conv_up1,
                         &a, &height, &width);
    if (ok) tae->stage = "the second block group";
    ok = ok && run_group(tae, tae->blocks_b, 3, &tae->conv_up2,
                         &a, &height, &width);
    if (ok) tae->stage = "the third block group";
    ok = ok && run_group(tae, tae->blocks_c, 3, &tae->conv_up3,
                         &a, &height, &width);
    if (ok) tae->stage = "the final block";
    ok = ok && run_group(tae, &tae->block_d, 1, NULL, &a, &height, &width);
    if (ok) {
        tae->stage = "the output convolution";
        next = (a + 1) % 3;
        ok = run_conv(tae, &tae->conv_out, tae->work[next], tae->work[a],
                      height, width, TAEF1_CHANNELS, 3);
        a = next;
    }
    ok = ok && h3_gpu_submit(tae->gpu);
    if (!ok) {
        const char *reason = h3_gpu_error(tae->gpu);
        fail(error, error_size, "Z-Image preview failed at %s: %s",
             tae->stage ? tae->stage : "an unlabelled stage",
             reason && *reason ? reason : "no GPU detail");
        return 0;
    }
    const size_t values = (size_t)height * (size_t)width * 3;
    if (!h3_gpu_tensor_read_f32(tae->work[a], rgb, values)) {
        fail(error, error_size, "cannot read the Z-Image preview back");
        return 0;
    }
    for (size_t index = 0; index < values; index++) {
        const float value = rgb[index];
        rgb[index] = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
    }
    return 1;
}
