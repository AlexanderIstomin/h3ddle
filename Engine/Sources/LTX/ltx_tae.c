#include "ltx_tae.h"

#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* madebyollin/taehv's LTX-2.3 decoder, which also supports LTX-2.5. This is a
 * clean native implementation of the published module layout. Its temporal
 * layers normally turn one latent into eight causally ordered pictures; for a
 * cheap still preview they are patched to stride one exactly as upstream
 * supports, retaining the last output slice from each learned TGrow.
 *
 * A MemBlock convolves concat(current, previous). There is no previous frame
 * in this stride-one preview, so its second half is zero and can be removed
 * from the first convolution rather than materialized on every pass.
 */

enum {
    LATENT_CHANNELS = 128,
    PATCH_SIZE = 4,
    SPATIAL_GROWTH = 8,
    OUTPUT_CHANNELS = 3 * PATCH_SIZE * PATCH_SIZE
};

typedef struct {
    h3_gpu_tensor *weight;
    h3_gpu_tensor *bias;
    int input_channels;
    int output_channels;
    int kernel;
} tae_conv;

typedef struct {
    tae_conv conv0;
    tae_conv conv2;
    tae_conv conv4;
    int channels;
} tae_block;

struct ltx_tae {
    h3_gpu *gpu;                    /* borrowed from the LTX pipeline */
    h3_weight_store *store;
    tae_conv input;
    tae_block blocks[3][3];
    tae_conv grow[3];
    tae_conv transition[3];
    tae_conv output;
    h3_gpu_tensor *work[3];
    float *staging;
    float *packed;
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

static float f16_to_f32(uint16_t value) {
    const unsigned sign = value >> 15;
    const unsigned exponent = (value >> 10) & 31u;
    const unsigned fraction = value & 1023u;
    float result;
    if (!exponent)
        result = ldexpf((float)fraction, -24);
    else if (exponent == 31u)
        result = fraction ? NAN : INFINITY;
    else
        result = ldexpf((float)(1024u + fraction), (int)exponent - 25);
    return sign ? -result : result;
}

static h3_gpu_tensor *load_weight_slice(
    ltx_tae *tae, const char *name, int full_outputs, int full_inputs,
    int output_begin, int outputs, int input_begin, int inputs, int kernel,
    char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(tae->store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return NULL;
    }
    if (!header || tensor->dtype != H3_DTYPE_F16 || tensor->ndim != 4 ||
        tensor->shape[0] != (uint64_t)full_outputs ||
        tensor->shape[1] != (uint64_t)full_inputs ||
        tensor->shape[2] != (uint64_t)kernel ||
        tensor->shape[3] != (uint64_t)kernel) {
        fail(error, error_size, "%s does not have the expected F16 shape", name);
        return NULL;
    }
    if (output_begin < 0 || outputs < 1 ||
        output_begin + outputs > full_outputs || input_begin < 0 ||
        inputs < 1 || input_begin + inputs > full_inputs) {
        fail(error, error_size, "invalid preview slice for %s", name);
        return NULL;
    }
    const size_t kernel_area = (size_t)kernel * (size_t)kernel;
    const size_t source_count =
        (size_t)full_outputs * (size_t)full_inputs * kernel_area;
    const size_t selected_count = (size_t)outputs * (size_t)inputs * kernel_area;
    uint16_t *source = malloc(source_count * sizeof(*source));
    float *selected = malloc(selected_count * sizeof(*selected));
    if (!source || !selected) {
        free(source); free(selected);
        fail(error, error_size, "out of memory loading %s", name);
        return NULL;
    }
    if (!h3_st_read_data(header, tensor, source,
                         source_count * sizeof(*source), error, error_size)) {
        free(source); free(selected);
        return NULL;
    }
    for (int output = 0; output < outputs; output++)
        for (int input = 0; input < inputs; input++)
            for (size_t tap = 0; tap < kernel_area; tap++) {
                const size_t from =
                    ((size_t)(output + output_begin) * (size_t)full_inputs +
                     (size_t)(input + input_begin)) * kernel_area + tap;
                const size_t to =
                    ((size_t)output * (size_t)inputs + (size_t)input) *
                    kernel_area + tap;
                selected[to] = f16_to_f32(source[from]);
            }
    free(source);
    h3_gpu_tensor *result =
        h3_gpu_tensor_from_f32(tae->gpu, selected, selected_count);
    free(selected);
    if (!result)
        fail(error, error_size, "cannot upload %s: %s", name,
             h3_gpu_error(tae->gpu));
    return result;
}

static int load_conv(ltx_tae *tae, tae_conv *conv, const char *stem,
                     int input_channels, int output_channels, int kernel,
                     int has_bias, char *error, size_t error_size) {
    char name[96];
    const uint64_t weight_shape[] = {
        (uint64_t)output_channels, (uint64_t)input_channels,
        (uint64_t)kernel, (uint64_t)kernel
    };
    snprintf(name, sizeof(name), "%s.weight", stem);
    conv->weight = h3_weight_load_f16_as_f32(
        tae->store, tae->gpu, name, 4, weight_shape, error, error_size);
    if (!conv->weight) return 0;
    conv->input_channels = input_channels;
    conv->output_channels = output_channels;
    conv->kernel = kernel;
    if (has_bias) {
        const uint64_t bias_shape[] = {(uint64_t)output_channels};
        snprintf(name, sizeof(name), "%s.bias", stem);
        conv->bias = h3_weight_load_f16_as_f32(
            tae->store, tae->gpu, name, 1, bias_shape, error, error_size);
        if (!conv->bias) return 0;
    }
    return 1;
}

static int load_block(ltx_tae *tae, tae_block *block, int index, int channels,
                      char *error, size_t error_size) {
    char stem[96];
    char name[96];
    block->channels = channels;
    snprintf(name, sizeof(name), "decoder.%d.conv.0.weight", index);
    block->conv0.weight = load_weight_slice(
        tae, name, channels, channels * 2, 0, channels, 0, channels, 3,
        error, error_size);
    block->conv0.input_channels = channels;
    block->conv0.output_channels = channels;
    block->conv0.kernel = 3;
    if (!block->conv0.weight) return 0;
    const uint64_t bias_shape[] = {(uint64_t)channels};
    snprintf(name, sizeof(name), "decoder.%d.conv.0.bias", index);
    block->conv0.bias = h3_weight_load_f16_as_f32(
        tae->store, tae->gpu, name, 1, bias_shape, error, error_size);
    if (!block->conv0.bias) return 0;
    snprintf(stem, sizeof(stem), "decoder.%d.conv.2", index);
    if (!load_conv(tae, &block->conv2, stem, channels, channels, 3, 1,
                   error, error_size))
        return 0;
    snprintf(stem, sizeof(stem), "decoder.%d.conv.4", index);
    return load_conv(tae, &block->conv4, stem, channels, channels, 3, 1,
                     error, error_size);
}

static int load_grow(ltx_tae *tae, tae_conv *grow, int index, int channels,
                     char *error, size_t error_size) {
    char name[96];
    snprintf(name, sizeof(name), "decoder.%d.conv.weight", index);
    /* TGrow's two temporal outputs are contiguous in output-channel order.
     * Upstream's stride-one patch keeps the last output slice. */
    grow->weight = load_weight_slice(
        tae, name, channels * 2, channels, channels, channels, 0, channels, 1,
        error, error_size);
    grow->input_channels = channels;
    grow->output_channels = channels;
    grow->kernel = 1;
    return grow->weight != NULL;
}

static int load_all(ltx_tae *tae, char *error, size_t error_size) {
    static const int block_indices[3][3] = {{3, 4, 5}, {9, 10, 11},
                                             {15, 16, 17}};
    static const int channels[3] = {256, 128, 64};
    static const int grow_indices[3] = {7, 13, 19};
    static const int transition_indices[3] = {8, 14, 20};
    static const int next_channels[3] = {128, 64, 64};
    if (!load_conv(tae, &tae->input, "decoder.1", LATENT_CHANNELS, 256,
                   3, 1, error, error_size))
        return 0;
    for (int group = 0; group < 3; group++) {
        for (int block = 0; block < 3; block++)
            if (!load_block(tae, &tae->blocks[group][block],
                            block_indices[group][block], channels[group],
                            error, error_size))
                return 0;
        if (!load_grow(tae, &tae->grow[group], grow_indices[group],
                       channels[group], error, error_size))
            return 0;
        char stem[48];
        snprintf(stem, sizeof(stem), "decoder.%d", transition_indices[group]);
        if (!load_conv(tae, &tae->transition[group], stem, channels[group],
                       next_channels[group], 3, 0, error, error_size))
            return 0;
    }
    return load_conv(tae, &tae->output, "decoder.22", 64, OUTPUT_CHANNELS,
                     3, 1, error, error_size);
}

static int allocate_work(ltx_tae *tae, char *error, size_t error_size) {
    const size_t latent_area =
        (size_t)tae->latent_height * (size_t)tae->latent_width;
    const size_t grown_area = latent_area * SPATIAL_GROWTH * SPATIAL_GROWTH;
    const size_t work_elements = grown_area * 64;
    const size_t packed_elements = grown_area * OUTPUT_CHANNELS;
    tae->staging = malloc(latent_area * LATENT_CHANNELS * sizeof(*tae->staging));
    tae->packed = malloc(packed_elements * sizeof(*tae->packed));
    for (int index = 0; index < 3; index++)
        tae->work[index] = h3_gpu_tensor_new_f32(tae->gpu, work_elements);
    if (!tae->staging || !tae->packed || !tae->work[0] || !tae->work[1] ||
        !tae->work[2]) {
        fail(error, error_size, "out of memory for LTX preview buffers");
        return 0;
    }
    return 1;
}

ltx_tae *ltx_tae_create(h3_gpu *gpu, const char *weight_path,
                        int latent_height, int latent_width,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!gpu || !weight_path || !*weight_path || latent_height < 1 ||
        latent_width < 1) {
        fail(error, error_size, "invalid LTX preview decoder arguments");
        return NULL;
    }
    ltx_tae *tae = calloc(1, sizeof(*tae));
    if (!tae) {
        fail(error, error_size, "out of memory for the LTX preview decoder");
        return NULL;
    }
    tae->gpu = gpu;
    tae->latent_height = latent_height;
    tae->latent_width = latent_width;
    tae->store = h3_weight_store_open(weight_path, error, error_size);
    if (!tae->store || !load_all(tae, error, error_size) ||
        !allocate_work(tae, error, error_size)) {
        ltx_tae_release(tae);
        return NULL;
    }
    return tae;
}

static void free_conv(tae_conv *conv) {
    h3_gpu_tensor_free(conv->weight);
    h3_gpu_tensor_free(conv->bias);
}

void ltx_tae_release(ltx_tae *tae) {
    if (!tae) return;
    free_conv(&tae->input);
    for (int group = 0; group < 3; group++) {
        for (int block = 0; block < 3; block++) {
            free_conv(&tae->blocks[group][block].conv0);
            free_conv(&tae->blocks[group][block].conv2);
            free_conv(&tae->blocks[group][block].conv4);
        }
        free_conv(&tae->grow[group]);
        free_conv(&tae->transition[group]);
    }
    free_conv(&tae->output);
    for (int index = 0; index < 3; index++)
        h3_gpu_tensor_free(tae->work[index]);
    h3_weight_store_free(tae->store);
    free(tae->staging);
    free(tae->packed);
    free(tae);
}

static int run_conv(ltx_tae *tae, const tae_conv *conv,
                    h3_gpu_tensor *output, const h3_gpu_tensor *input,
                    int height, int width) {
    return h3_gpu_conv3d_same_f32(
        tae->gpu, output, input, conv->weight, conv->bias, 1, 1,
        (uint32_t)height, (uint32_t)width,
        (uint32_t)conv->input_channels, (uint32_t)conv->output_channels,
        1, (uint32_t)conv->kernel, (uint32_t)conv->kernel, 1, 1, 1);
}

static int run_block(ltx_tae *tae, const tae_block *block, int a,
                     int height, int width) {
    h3_gpu_tensor *input = tae->work[a];
    h3_gpu_tensor *first = tae->work[(a + 1) % 3];
    h3_gpu_tensor *second = tae->work[(a + 2) % 3];
    const uint32_t elements =
        (uint32_t)((size_t)height * (size_t)width * (size_t)block->channels);
    if (!run_conv(tae, &block->conv0, first, input, height, width) ||
        !h3_gpu_relu_f32(tae->gpu, first, first, elements) ||
        !run_conv(tae, &block->conv2, second, first, height, width) ||
        !h3_gpu_relu_f32(tae->gpu, second, second, elements) ||
        !run_conv(tae, &block->conv4, first, second, height, width))
        return 0;
    return h3_gpu_add_scaled_f32(tae->gpu, tae->work[a], first, input,
                                 1.0f, 1.0f, elements) &&
           h3_gpu_relu_f32(tae->gpu, tae->work[a], tae->work[a], elements);
}

static int run_group(ltx_tae *tae, int group, int *a,
                     int *height, int *width) {
    const int channels = tae->blocks[group][0].channels;
    for (int block = 0; block < 3; block++)
        if (!run_block(tae, &tae->blocks[group][block], *a, *height, *width))
            return 0;
    int next = (*a + 1) % 3;
    if (!h3_gpu_nearest2x_nhwc_f32(
            tae->gpu, tae->work[next], tae->work[*a],
            (uint32_t)*height, (uint32_t)*width, (uint32_t)channels))
        return 0;
    *a = next;
    *height *= 2;
    *width *= 2;
    next = (*a + 1) % 3;
    if (!run_conv(tae, &tae->grow[group], tae->work[next], tae->work[*a],
                  *height, *width))
        return 0;
    *a = next;
    next = (*a + 1) % 3;
    if (!run_conv(tae, &tae->transition[group], tae->work[next],
                  tae->work[*a], *height, *width))
        return 0;
    *a = next;
    return 1;
}

int ltx_tae_decode(ltx_tae *tae, const float *latent, float *rgb,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tae || !latent || !rgb) {
        fail(error, error_size, "invalid LTX preview decode arguments");
        return 0;
    }
    int height = tae->latent_height;
    int width = tae->latent_width;
    const size_t latent_values =
        (size_t)height * (size_t)width * LATENT_CHANNELS;
    for (size_t index = 0; index < latent_values; index++)
        tae->staging[index] = 3.0f * tanhf(latent[index] / 3.0f);
    if (!h3_gpu_tensor_write_f32(tae->work[0], tae->staging, latent_values)) {
        fail(error, error_size, "cannot upload the LTX preview latent");
        return 0;
    }
    if (!h3_gpu_begin(tae->gpu)) {
        fail(error, error_size, "cannot begin the LTX preview command: %s",
             h3_gpu_error(tae->gpu));
        return 0;
    }

    int a = 0;
    int next = 1;
    tae->stage = "the input convolution";
    int ok = run_conv(tae, &tae->input, tae->work[next], tae->work[a],
                      height, width) &&
             h3_gpu_relu_f32(
                 tae->gpu, tae->work[next], tae->work[next],
                 (uint32_t)((size_t)height * (size_t)width * 256));
    a = next;
    for (int group = 0; ok && group < 3; group++) {
        tae->stage = group == 0 ? "the first block group" :
                     (group == 1 ? "the second block group" :
                                   "the third block group");
        ok = run_group(tae, group, &a, &height, &width);
    }
    if (ok) {
        tae->stage = "the output convolution";
        ok = h3_gpu_relu_f32(
            tae->gpu, tae->work[a], tae->work[a],
            (uint32_t)((size_t)height * (size_t)width * 64));
        next = (a + 1) % 3;
        ok = ok && run_conv(tae, &tae->output, tae->work[next], tae->work[a],
                            height, width);
        a = next;
    }
    /* Close the command buffer even if dispatch encoding failed. Preview
     * errors are non-fatal to generation, so leaving an open buffer behind
     * would turn a recoverable preview failure into a broken DiT context. */
    const int submitted = h3_gpu_submit(tae->gpu);
    ok = ok && submitted;
    if (!ok) {
        const char *reason = h3_gpu_error(tae->gpu);
        fail(error, error_size, "LTX preview failed at %s: %s",
             tae->stage ? tae->stage : "an unlabelled stage",
             reason && *reason ? reason : "no GPU detail");
        return 0;
    }
    const size_t packed_values =
        (size_t)height * (size_t)width * OUTPUT_CHANNELS;
    if (!h3_gpu_tensor_read_f32(tae->work[a], tae->packed, packed_values)) {
        fail(error, error_size, "cannot read the LTX preview back");
        return 0;
    }
    ltx_tae_pixel_shuffle4(tae->packed, height, width, rgb);
    return 1;
}
