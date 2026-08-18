#include "zimage_vae_gpu.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LATENT_CHANNELS 16
#define MID_CHANNELS    512
#define GROUPS          32
#define EPSILON         1e-6f
#define UP_BLOCKS       4
#define RESNETS         3

/* Per up block, in order; the picture doubles after each of the first three. */
static const int CHANNELS[UP_BLOCKS] = {512, 512, 256, 128};

typedef struct { h3_gpu_tensor *weight, *bias; int out, in, kernel; } conv;
typedef struct { h3_gpu_tensor *weight, *bias; } norm;
typedef struct {
    norm norm1, norm2;
    conv conv1, conv2, shortcut;   /* shortcut.weight NULL when unchanged */
} resnet;

/* Per down block, in order; the picture halves after each of the first three. */
static const int DOWN_CHANNELS[UP_BLOCKS] = {128, 256, 512, 512};
#define DOWN_RESNETS    2
#define IMAGE_CHANNELS  3
#define MOMENT_CHANNELS 32   /* conv_out: a mean and a log-variance, stacked */

struct zimage_vae_gpu {
    h3_gpu *gpu;
    int owns_gpu;
    /* Built as the encoder rather than the decoder. The two run the same
     * pieces in opposite order, so they share these fields: `up` holds the
     * down blocks and `upsampler` the downsamplers. Asserted on entry to
     * each direction so the pair can never be crossed silently. */
    int is_encoder;
    h3_weight_store *store;
    conv conv_in, conv_out;
    resnet mid[2];
    norm mid_norm, out_norm;
    h3_gpu_tensor *to_q, *to_q_bias, *to_k, *to_k_bias;
    h3_gpu_tensor *to_v, *to_v_bias, *to_out, *to_out_bias;
    resnet up[UP_BLOCKS][RESNETS];
    conv upsampler[UP_BLOCKS - 1];
    h3_gpu_tensor *work[5];  /* x, normed, query, key, value */
    size_t work_elements;
};

static int fail(char *error, size_t error_size, const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
    return 0;
}

/* The autoencoder ships bf16 and runs f32 (`force_upcast`), so widen once on
 * the way in and compute f32 throughout, as the CPU decoder does. */
static h3_gpu_tensor *widen(zimage_vae_gpu *vae, const char *name, int ndim,
                            const uint64_t *shape, int required,
                            char *error, size_t error_size) {
    h3_gpu_tensor *raw = h3_weight_load_bf16(vae->store, vae->gpu, name, ndim,
                                             shape, error, error_size);
    if (!raw) {
        if (!required) { if (error_size) error[0] = '\0'; return NULL; }
        return NULL;
    }
    size_t count = 1;
    for (int index = 0; index < ndim; index++) count *= (size_t)shape[index];
    uint16_t *narrow = malloc(count * sizeof(uint16_t));
    float *wide = malloc(count * sizeof(float));
    if (!narrow || !wide || !h3_gpu_tensor_read_bf16(raw, narrow, count)) {
        free(narrow); free(wide);
        h3_gpu_tensor_free(raw);
        fail(error, error_size, "cannot widen %s", name);
        return NULL;
    }
    for (size_t index = 0; index < count; index++) {
        union { uint32_t bits; float number; } cast;
        cast.bits = (uint32_t)narrow[index] << 16;
        wide[index] = cast.number;
    }
    h3_gpu_tensor *result = h3_gpu_tensor_from_f32(vae->gpu, wide, count);
    free(narrow); free(wide);
    h3_gpu_tensor_free(raw);
    return result;
}

static void format(char *buffer, size_t size, const char *pattern, int a, int b) {
    if (b >= 0)      snprintf(buffer, size, pattern, a, b);
    else if (a >= 0) snprintf(buffer, size, pattern, a);
    else             snprintf(buffer, size, "%s", pattern);
}

static conv take_conv(zimage_vae_gpu *vae, const char *pattern, int a, int b,
                      int out, int in, int kernel, int required,
                      char *error, size_t error_size) {
    char name[256], full[300];
    format(name, sizeof(name), pattern, a, b);
    /* The checkpoint's own layout: the kernel indexes [out, in, kh, kw]. */
    const uint64_t shape[4] = {(uint64_t)out, (uint64_t)in,
                               (uint64_t)kernel, (uint64_t)kernel};
    snprintf(full, sizeof(full), "%s.weight", name);
    h3_gpu_tensor *weight = widen(vae, full, 4, shape, required, error, error_size);
    if (!weight) return (conv){NULL, NULL, 0, 0, 0};
    snprintf(full, sizeof(full), "%s.bias", name);
    const uint64_t bias_shape[1] = {(uint64_t)out};
    return (conv){weight, widen(vae, full, 1, bias_shape, 1, error, error_size),
                  out, in, kernel};
}

static norm take_norm(zimage_vae_gpu *vae, const char *pattern, int a, int b,
                      int channels, char *error, size_t error_size) {
    char name[256], full[300];
    format(name, sizeof(name), pattern, a, b);
    const uint64_t shape[1] = {(uint64_t)channels};
    snprintf(full, sizeof(full), "%s.weight", name);
    h3_gpu_tensor *weight = widen(vae, full, 1, shape, 1, error, error_size);
    snprintf(full, sizeof(full), "%s.bias", name);
    return (norm){weight, widen(vae, full, 1, shape, 1, error, error_size)};
}

static h3_gpu_tensor *take_linear(zimage_vae_gpu *vae, const char *name,
                                  int out, int in, char *error, size_t size) {
    const uint64_t shape[2] = {(uint64_t)out, (uint64_t)in};
    return widen(vae, name, 2, shape, 1, error, size);
}

static h3_gpu_tensor *take_vector(zimage_vae_gpu *vae, const char *name,
                                  int count, char *error, size_t size) {
    const uint64_t shape[1] = {(uint64_t)count};
    return widen(vae, name, 1, shape, 1, error, size);
}

zimage_vae_gpu *zimage_vae_gpu_create(const char *shaders, const char *decoder,
                                      h3_gpu *device, int max_side,
                                      char *error, size_t error_size) {
    if (!device && !h3_gpu_prepare(shaders, error, error_size)) return NULL;
    zimage_vae_gpu *vae = calloc(1, sizeof(*vae));
    if (!vae) { fail(error, error_size, "out of memory"); return NULL; }
    vae->gpu = device ? device : h3_gpu_create(shaders, error, error_size);
    vae->owns_gpu = device ? 0 : 1;
    vae->store = h3_weight_store_open(decoder, error, error_size);
    if (!vae->gpu || !vae->store) { zimage_vae_gpu_release(vae); return NULL; }

    /* The widest the stack gets: the third upsample carries 256 channels at
     * the full picture size, and nothing later exceeds it. */
    const size_t peak = (size_t)256 * (size_t)(max_side * 8) * (size_t)(max_side * 8);
    for (int index = 0; index < 5; index++) {
        vae->work[index] = h3_gpu_tensor_new_f32(vae->gpu, peak);
        if (!vae->work[index]) {
            fail(error, error_size, "out of GPU memory for the decoder");
            zimage_vae_gpu_release(vae);
            return NULL;
        }
    }
    vae->work_elements = peak;

    vae->conv_in = take_conv(vae, "decoder.conv_in", -1, -1, MID_CHANNELS,
                             LATENT_CHANNELS, 3, 1, error, error_size);
    for (int index = 0; index < 2; index++) {
        vae->mid[index].norm1 = take_norm(vae, "decoder.mid_block.resnets.%d.norm1",
                                          index, -1, MID_CHANNELS, error, error_size);
        vae->mid[index].norm2 = take_norm(vae, "decoder.mid_block.resnets.%d.norm2",
                                          index, -1, MID_CHANNELS, error, error_size);
        vae->mid[index].conv1 = take_conv(vae, "decoder.mid_block.resnets.%d.conv1",
                                          index, -1, MID_CHANNELS, MID_CHANNELS, 3, 1,
                                          error, error_size);
        vae->mid[index].conv2 = take_conv(vae, "decoder.mid_block.resnets.%d.conv2",
                                          index, -1, MID_CHANNELS, MID_CHANNELS, 3, 1,
                                          error, error_size);
    }
    vae->mid_norm = take_norm(vae, "decoder.mid_block.attentions.0.group_norm",
                              -1, -1, MID_CHANNELS, error, error_size);
#define PROJECTION(field, leaf) do {                                          \
        vae->field = take_linear(vae,                                          \
            "decoder.mid_block.attentions.0." leaf ".weight",                  \
            MID_CHANNELS, MID_CHANNELS, error, error_size);                    \
        vae->field##_bias = take_vector(vae,                                   \
            "decoder.mid_block.attentions.0." leaf ".bias",                    \
            MID_CHANNELS, error, error_size);                                  \
    } while (0)
    PROJECTION(to_q, "to_q");
    PROJECTION(to_k, "to_k");
    PROJECTION(to_v, "to_v");
    PROJECTION(to_out, "to_out.0");
#undef PROJECTION

    int current = MID_CHANNELS;
    for (int block = 0; block < UP_BLOCKS; block++) {
        const int out_channels = CHANNELS[block];
        for (int index = 0; index < RESNETS; index++) {
            const int in_channels = index == 0 ? current : out_channels;
            resnet *r = &vae->up[block][index];
            r->norm1 = take_norm(vae, "decoder.up_blocks.%d.resnets.%d.norm1",
                                 block, index, in_channels, error, error_size);
            r->norm2 = take_norm(vae, "decoder.up_blocks.%d.resnets.%d.norm2",
                                 block, index, out_channels, error, error_size);
            r->conv1 = take_conv(vae, "decoder.up_blocks.%d.resnets.%d.conv1",
                                 block, index, out_channels, in_channels, 3, 1,
                                 error, error_size);
            r->conv2 = take_conv(vae, "decoder.up_blocks.%d.resnets.%d.conv2",
                                 block, index, out_channels, out_channels, 3, 1,
                                 error, error_size);
            /* Only the two resnets that change channel count carry one. */
            r->shortcut = take_conv(vae, "decoder.up_blocks.%d.resnets.%d.conv_shortcut",
                                    block, index, out_channels, in_channels, 1, 0,
                                    error, error_size);
        }
        current = out_channels;
        if (block < UP_BLOCKS - 1)
            vae->upsampler[block] = take_conv(vae,
                "decoder.up_blocks.%d.upsamplers.0.conv", block, -1,
                current, current, 3, 1, error, error_size);
    }
    vae->out_norm = take_norm(vae, "decoder.conv_norm_out", -1, -1, current,
                              error, error_size);
    vae->conv_out = take_conv(vae, "decoder.conv_out", -1, -1, 3, current, 3, 1,
                              error, error_size);
    if (!vae->conv_in.weight || !vae->conv_out.weight || !vae->to_q) {
        fail(error, error_size, "the decoder is missing weights");
        zimage_vae_gpu_release(vae);
        return NULL;
    }
    return vae;
}

void zimage_vae_gpu_release(zimage_vae_gpu *vae) {
    if (!vae) return;
    for (int index = 0; index < 5; index++) h3_gpu_tensor_free(vae->work[index]);
    if (vae->store) h3_weight_store_free(vae->store);
    if (vae->gpu && vae->owns_gpu) h3_gpu_free(vae->gpu);
    free(vae);
}

#define OP(call, label) do {                                                  \
    if (!(call)) return fail(error, error_size, "%s failed", label);           \
} while (0)

static int run_conv(zimage_vae_gpu *vae, const conv *c, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, int height, int width,
                    char *error, size_t error_size) {
    OP(h3_gpu_conv3d_same_f32(vae->gpu, output, input, c->weight, c->bias,
                              1, 1, (uint32_t)height, (uint32_t)width,
                              (uint32_t)c->in, (uint32_t)c->out,
                              1, (uint32_t)c->kernel, (uint32_t)c->kernel,
                              1, 1, 1), "convolution");
    return 1;
}

/* GroupNorm, SiLU, conv, GroupNorm, SiLU, conv, plus the residual — which
 * takes a 1x1 projection only where the channel count changes. */
static int run_resnet(zimage_vae_gpu *vae, const resnet *block,
                      int in_channels, int out_channels, int height, int width,
                      char *error, size_t error_size) {
    h3_gpu_tensor *x = vae->work[0], *scratch = vae->work[1];
    h3_gpu_tensor *branch = vae->work[2];
    const uint32_t in_count = (uint32_t)height * width * in_channels;
    const uint32_t out_count = (uint32_t)height * width * out_channels;

    OP(h3_gpu_vae_encoder_group_norm_silu_f32(vae->gpu, scratch, x,
        block->norm1.weight, block->norm1.bias, 1, 1, height, width,
        in_channels, GROUPS, EPSILON), "resnet norm 1");
    if (!run_conv(vae, &block->conv1, branch, scratch, height, width,
                  error, error_size)) return 0;
    OP(h3_gpu_vae_encoder_group_norm_silu_f32(vae->gpu, scratch, branch,
        block->norm2.weight, block->norm2.bias, 1, 1, height, width,
        out_channels, GROUPS, EPSILON), "resnet norm 2");
    if (!run_conv(vae, &block->conv2, branch, scratch, height, width,
                  error, error_size)) return 0;

    const h3_gpu_tensor *residual = x;
    if (block->shortcut.weight) {
        if (!run_conv(vae, &block->shortcut, scratch, x, height, width,
                      error, error_size)) return 0;
        residual = scratch;
    } else if (in_channels != out_channels) {
        return fail(error, error_size,
                    "resnet changes %d to %d channels with no shortcut",
                    in_channels, out_channels);
    }
    (void)in_count;
    OP(h3_gpu_add_scaled_f32(vae->gpu, x, branch, residual, 1.0f, 1.0f,
                             out_count), "resnet residual");
    return 1;
}

/* Single-head attention over every pixel, at the smallest resolution in the
 * stack. Channel-last already puts the [height*width][channels] matrix the
 * projections want in front of us, so no reshaping is needed.
 *
 * Every buffer here is distinct on purpose. A GEMM whose output aliases one of
 * its inputs is a race — some threads read rows others have already
 * overwritten — and it does not fail, it quietly returns a slightly wrong
 * answer. That cost this decoder 3.58e-02 against a golden it should have met
 * at 1e-06, which reads exactly like a precision problem and is not one. */
static int run_attention(zimage_vae_gpu *vae, int height, int width,
                         char *error, size_t error_size) {
    /* Both directions carry their own projections; the fields are whichever
     * set `create` loaded. */
    h3_gpu_tensor *x = vae->work[0], *normed = vae->work[1];
    h3_gpu_tensor *query = vae->work[2], *key = vae->work[3];
    h3_gpu_tensor *value = vae->work[4];
    const uint32_t pixels = (uint32_t)height * width;

    OP(h3_gpu_vae_group_norm_f32(vae->gpu, normed, x, vae->mid_norm.weight,
        vae->mid_norm.bias, 1, 1, height, width, MID_CHANNELS, GROUPS,
        EPSILON), "attention norm");
    OP(h3_gpu_linear_f32(vae->gpu, query, normed, vae->to_q, vae->to_q_bias,
                         pixels, MID_CHANNELS, MID_CHANNELS), "attention query");
    OP(h3_gpu_linear_f32(vae->gpu, key, normed, vae->to_k, vae->to_k_bias,
                         pixels, MID_CHANNELS, MID_CHANNELS), "attention key");
    OP(h3_gpu_linear_f32(vae->gpu, value, normed, vae->to_v, vae->to_v_bias,
                         pixels, MID_CHANNELS, MID_CHANNELS), "attention value");
    /* Lands in `normed`, which the three projections have finished with. */
    OP(h3_gpu_sdpa_f32(vae->gpu, normed, query, key, value, pixels, 1,
                       MID_CHANNELS, 1.0f / sqrtf((float)MID_CHANNELS)),
       "attention");
    OP(h3_gpu_linear_f32(vae->gpu, query, normed, vae->to_out, vae->to_out_bias,
                         pixels, MID_CHANNELS, MID_CHANNELS), "attention output");
    OP(h3_gpu_add_scaled_f32(vae->gpu, x, x, query, 1.0f, 1.0f,
                             pixels * MID_CHANNELS), "attention residual");
    return 1;
}

int zimage_vae_gpu_decode(zimage_vae_gpu *vae, const float *latent, int side,
                          float *image, char *error, size_t error_size) {
    if (vae->is_encoder)
        return fail(error, error_size, "this autoencoder was built to encode");
    const int final_side = side * 8;
    if ((size_t)256 * final_side * final_side > vae->work_elements)
        return fail(error, error_size, "%d is larger than this was sized for",
                    side);

    /* Channel-major in, channel-last on the device. */
    float *staged = malloc((size_t)LATENT_CHANNELS * side * side * sizeof(float));
    if (!staged) return fail(error, error_size, "out of memory");
    for (int y = 0; y < side; y++)
        for (int x = 0; x < side; x++)
            for (int channel = 0; channel < LATENT_CHANNELS; channel++)
                staged[((size_t)y * side + x) * LATENT_CHANNELS + channel] =
                    latent[((size_t)channel * side + y) * side + x];
    const int uploaded = h3_gpu_tensor_write_f32(
        vae->work[1], staged, (size_t)LATENT_CHANNELS * side * side);
    free(staged);
    if (!uploaded) return fail(error, error_size, "cannot upload the latent");

    if (!h3_gpu_begin(vae->gpu)) return fail(error, error_size, "cannot begin");
    if (!run_conv(vae, &vae->conv_in, vae->work[0], vae->work[1], side, side,
                  error, error_size)) return 0;

    for (int index = 0; index < 2; index++) {
        if (!run_resnet(vae, &vae->mid[index], MID_CHANNELS, MID_CHANNELS,
                        side, side, error, error_size)) return 0;
        if (index == 0 && !run_attention(vae, side, side, error, error_size))
            return 0;
    }

    int current = MID_CHANNELS, height = side, width = side;
    for (int block = 0; block < UP_BLOCKS; block++) {
        const int out_channels = CHANNELS[block];
        for (int index = 0; index < RESNETS; index++) {
            const int in_channels = index == 0 ? current : out_channels;
            if (!run_resnet(vae, &vae->up[block][index], in_channels,
                            out_channels, height, width, error, error_size))
                return 0;
        }
        current = out_channels;
        if (block < UP_BLOCKS - 1) {
            OP(h3_gpu_nearest2x_nhwc_f32(vae->gpu, vae->work[1], vae->work[0],
                                         (uint32_t)height, (uint32_t)width,
                                         (uint32_t)current), "upsample");
            height *= 2; width *= 2;
            if (!run_conv(vae, &vae->upsampler[block], vae->work[0], vae->work[1],
                          height, width, error, error_size)) return 0;
        }
    }

    OP(h3_gpu_vae_group_norm_activated_f32(vae->gpu, vae->work[1], vae->work[0],
        vae->out_norm.weight, vae->out_norm.bias, 1, 1, height, width,
        current, GROUPS, EPSILON, 1), "output norm");
    if (!run_conv(vae, &vae->conv_out, vae->work[0], vae->work[1], height, width,
                  error, error_size)) return 0;
    if (!h3_gpu_submit(vae->gpu)) return fail(error, error_size, "cannot submit");

    float *packed = malloc((size_t)3 * height * width * sizeof(float));
    if (!packed) return fail(error, error_size, "out of memory");
    if (!h3_gpu_tensor_read_f32(vae->work[0], packed, (size_t)3 * height * width)) {
        free(packed);
        return fail(error, error_size, "cannot read the picture back");
    }
    for (int channel = 0; channel < 3; channel++)
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                image[((size_t)channel * height + y) * width + x] =
                    packed[((size_t)y * width + x) * 3 + channel];
    free(packed);
    return 1;
}

/* ------------------------------------------------------------------ encode */

/* The encoder on the device, which is the same pieces in the opposite order.
 *
 * It exists because the CPU one is unusable at the sizes people render:
 * measured on an M1 Pro it takes 9 s at 256 square, 37 s at 512 and 176 s at
 * 1024 — longer than the entire eight-step render it is meant to precede, and
 * it lands before the first denoising step, so the bar sits still through all
 * of it.
 */
zimage_vae_gpu *zimage_vae_gpu_create_encoder(const char *shaders,
                                              const char *encoder,
                                              h3_gpu *device, int max_side,
                                              char *error, size_t error_size) {
    if (!device && !h3_gpu_prepare(shaders, error, error_size)) return NULL;
    zimage_vae_gpu *vae = calloc(1, sizeof(*vae));
    if (!vae) { fail(error, error_size, "out of memory"); return NULL; }
    vae->is_encoder = 1;
    vae->gpu = device ? device : h3_gpu_create(shaders, error, error_size);
    vae->owns_gpu = device ? 0 : 1;
    vae->store = h3_weight_store_open(encoder, error, error_size);
    if (!vae->gpu || !vae->store) { zimage_vae_gpu_release(vae); return NULL; }

    /* Widest at the start: 128 channels at the whole picture. Every later
     * stage doubles the channels only after quartering the area. */
    const size_t peak = (size_t)DOWN_CHANNELS[0] * (size_t)max_side * (size_t)max_side;
    for (int index = 0; index < 5; index++) {
        vae->work[index] = h3_gpu_tensor_new_f32(vae->gpu, peak);
        if (!vae->work[index]) {
            fail(error, error_size, "out of GPU memory for the encoder");
            zimage_vae_gpu_release(vae);
            return NULL;
        }
    }
    vae->work_elements = peak;

    vae->conv_in = take_conv(vae, "encoder.conv_in", -1, -1, DOWN_CHANNELS[0],
                             IMAGE_CHANNELS, 3, 1, error, error_size);

    int current = DOWN_CHANNELS[0];
    for (int block = 0; block < UP_BLOCKS; block++) {
        const int out_channels = DOWN_CHANNELS[block];
        for (int index = 0; index < DOWN_RESNETS; index++) {
            const int in_channels = index == 0 ? current : out_channels;
            resnet *r = &vae->up[block][index];
            r->norm1 = take_norm(vae, "encoder.down_blocks.%d.resnets.%d.norm1",
                                 block, index, in_channels, error, error_size);
            r->norm2 = take_norm(vae, "encoder.down_blocks.%d.resnets.%d.norm2",
                                 block, index, out_channels, error, error_size);
            r->conv1 = take_conv(vae, "encoder.down_blocks.%d.resnets.%d.conv1",
                                 block, index, out_channels, in_channels, 3, 1,
                                 error, error_size);
            r->conv2 = take_conv(vae, "encoder.down_blocks.%d.resnets.%d.conv2",
                                 block, index, out_channels, out_channels, 3, 1,
                                 error, error_size);
            if (in_channels != out_channels) {
                r->shortcut = take_conv(vae,
                    "encoder.down_blocks.%d.resnets.%d.conv_shortcut", block,
                    index, out_channels, in_channels, 1, 1, error, error_size);
            }
        }
        current = out_channels;
        if (block < UP_BLOCKS - 1) {
            vae->upsampler[block] = take_conv(vae,
                "encoder.down_blocks.%d.downsamplers.0.conv", block, -1,
                current, current, 3, 1, error, error_size);
        }
    }

    for (int index = 0; index < 2; index++) {
        vae->mid[index].norm1 = take_norm(vae, "encoder.mid_block.resnets.%d.norm1",
                                          index, -1, MID_CHANNELS, error, error_size);
        vae->mid[index].norm2 = take_norm(vae, "encoder.mid_block.resnets.%d.norm2",
                                          index, -1, MID_CHANNELS, error, error_size);
        vae->mid[index].conv1 = take_conv(vae, "encoder.mid_block.resnets.%d.conv1",
                                          index, -1, MID_CHANNELS, MID_CHANNELS, 3, 1,
                                          error, error_size);
        vae->mid[index].conv2 = take_conv(vae, "encoder.mid_block.resnets.%d.conv2",
                                          index, -1, MID_CHANNELS, MID_CHANNELS, 3, 1,
                                          error, error_size);
    }
    vae->mid_norm = take_norm(vae, "encoder.mid_block.attentions.0.group_norm",
                              -1, -1, MID_CHANNELS, error, error_size);
#define PROJECTION(field, leaf) do {                                          \
        vae->field = take_linear(vae,                                          \
            "encoder.mid_block.attentions.0." leaf ".weight",                  \
            MID_CHANNELS, MID_CHANNELS, error, error_size);                    \
        vae->field##_bias = take_vector(vae,                                   \
            "encoder.mid_block.attentions.0." leaf ".bias",                    \
            MID_CHANNELS, error, error_size);                                  \
    } while (0)
    PROJECTION(to_q, "to_q");
    PROJECTION(to_k, "to_k");
    PROJECTION(to_v, "to_v");
    PROJECTION(to_out, "to_out.0");
#undef PROJECTION

    vae->out_norm = take_norm(vae, "encoder.conv_norm_out", -1, -1,
                              MID_CHANNELS, error, error_size);
    vae->conv_out = take_conv(vae, "encoder.conv_out", -1, -1, MOMENT_CHANNELS,
                              MID_CHANNELS, 3, 1, error, error_size);

    if (!vae->conv_out.weight || !vae->conv_in.weight) {
        zimage_vae_gpu_release(vae);
        return NULL;
    }
    return vae;
}

/* Halve the picture, the way `Downsample2D` does.
 *
 * The reference pads one column on the right and one row at the bottom with
 * zeros, then convolves 3x3 at stride 2 with no padding of its own. The device
 * has a same-padded stride-1 convolution and no asymmetric zero pad — but the
 * two are exactly related: sampling a same-padded stride-1 result at the odd
 * offsets gives the strided one, since both read `in[2y + ky][2x + kx]`. So
 * the convolution runs at full size and every other pixel is kept.
 *
 * Keeping them costs a trip through the host, which is the price of not
 * writing a kernel for one op; it moves half a gigabyte at the first block and
 * a fraction of a second, against the 176 seconds this whole path replaces. */
static int run_downsample(zimage_vae_gpu *vae, const conv *c, int height,
                          int width, char *error, size_t error_size) {
    if (!run_conv(vae, c, vae->work[1], vae->work[0], height, width,
                  error, error_size)) return 0;
    if (!h3_gpu_submit(vae->gpu)) return fail(error, error_size, "cannot submit");

    const int out_height = height / 2, out_width = width / 2;
    const size_t channels = (size_t)c->out;
    float *full = malloc((size_t)height * width * channels * sizeof(float));
    float *kept = malloc((size_t)out_height * out_width * channels * sizeof(float));
    if (!full || !kept) {
        free(full); free(kept);
        return fail(error, error_size, "out of memory halving the picture");
    }
    if (!h3_gpu_tensor_read_f32(vae->work[1], full,
                                (size_t)height * width * channels)) {
        free(full); free(kept);
        return fail(error, error_size, "cannot read the picture back");
    }
    for (int y = 0; y < out_height; y++)
        for (int x = 0; x < out_width; x++)
            memcpy(kept + ((size_t)y * out_width + x) * channels,
                   full + ((size_t)(y * 2 + 1) * width + (x * 2 + 1)) * channels,
                   channels * sizeof(float));
    const int uploaded = h3_gpu_tensor_write_f32(
        vae->work[0], kept, (size_t)out_height * out_width * channels);
    free(full); free(kept);
    if (!uploaded) return fail(error, error_size, "cannot upload the picture");
    if (!h3_gpu_begin(vae->gpu)) return fail(error, error_size, "cannot begin");
    return 1;
}

int zimage_vae_gpu_encode(zimage_vae_gpu *vae, const float *image,
                          int image_side, float *latent,
                          char *error, size_t error_size) {
    if (!vae->is_encoder)
        return fail(error, error_size, "this autoencoder was built to decode");
    if ((size_t)DOWN_CHANNELS[0] * image_side * image_side > vae->work_elements)
        return fail(error, error_size, "%d is larger than this was sized for",
                    image_side);

    /* Channel-major in, channel-last on the device. */
    const size_t pixels = (size_t)image_side * image_side;
    float *staged = malloc(pixels * IMAGE_CHANNELS * sizeof(float));
    if (!staged) return fail(error, error_size, "out of memory");
    for (int y = 0; y < image_side; y++)
        for (int x = 0; x < image_side; x++)
            for (int channel = 0; channel < IMAGE_CHANNELS; channel++)
                staged[((size_t)y * image_side + x) * IMAGE_CHANNELS + channel] =
                    image[((size_t)channel * image_side + y) * image_side + x];
    const int uploaded = h3_gpu_tensor_write_f32(vae->work[1], staged,
                                                 pixels * IMAGE_CHANNELS);
    free(staged);
    if (!uploaded) return fail(error, error_size, "cannot upload the picture");

    if (!h3_gpu_begin(vae->gpu)) return fail(error, error_size, "cannot begin");
    if (!run_conv(vae, &vae->conv_in, vae->work[0], vae->work[1], image_side,
                  image_side, error, error_size)) return 0;

    int current = DOWN_CHANNELS[0], height = image_side, width = image_side;
    for (int block = 0; block < UP_BLOCKS; block++) {
        const int out_channels = DOWN_CHANNELS[block];
        for (int index = 0; index < DOWN_RESNETS; index++) {
            const int in_channels = index == 0 ? current : out_channels;
            if (!run_resnet(vae, &vae->up[block][index], in_channels,
                            out_channels, height, width, error, error_size))
                return 0;
        }
        current = out_channels;
        if (block < UP_BLOCKS - 1) {
            if (!run_downsample(vae, &vae->upsampler[block], height, width,
                                error, error_size)) return 0;
            height /= 2; width /= 2;
        }
    }

    for (int index = 0; index < 2; index++) {
        if (!run_resnet(vae, &vae->mid[index], MID_CHANNELS, MID_CHANNELS,
                        height, width, error, error_size)) return 0;
        if (index == 0 && !run_attention(vae, height, width, error, error_size))
            return 0;
    }

    OP(h3_gpu_vae_group_norm_activated_f32(vae->gpu, vae->work[1], vae->work[0],
        vae->out_norm.weight, vae->out_norm.bias, 1, 1, height, width,
        MID_CHANNELS, GROUPS, EPSILON, 1), "output norm");
    if (!run_conv(vae, &vae->conv_out, vae->work[0], vae->work[1], height, width,
                  error, error_size)) return 0;
    if (!h3_gpu_submit(vae->gpu)) return fail(error, error_size, "cannot submit");

    float *packed = malloc((size_t)MOMENT_CHANNELS * height * width * sizeof(float));
    if (!packed) return fail(error, error_size, "out of memory");
    if (!h3_gpu_tensor_read_f32(vae->work[0], packed,
                                (size_t)MOMENT_CHANNELS * height * width)) {
        free(packed);
        return fail(error, error_size, "cannot read the latent back");
    }
    /* The first sixteen channels are the mean; the log-variance behind them is
     * small enough to ignore, which the CPU path measures and explains. */
    for (int channel = 0; channel < LATENT_CHANNELS; channel++)
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                latent[((size_t)channel * height + y) * width + x] =
                    packed[((size_t)y * width + x) * MOMENT_CHANNELS + channel];
    free(packed);
    return 1;
}
