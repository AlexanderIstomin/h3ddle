/* LTX-2.5's audio half, as a library stage.
 *
 * Lifted from `Vendor/h3.c/tests/ltx_audio_decode.c`, which is the driver this
 * was proven with and stays the place to reproduce a disagreement. The
 * arithmetic is unchanged; what differs is that a library may not exit, so the
 * stage machinery threads a run context and latches the first error instead of
 * calling `fail`. Once latched, every later step short-circuits and the tail
 * frees whatever was built — which is why the helpers check `failed` on entry
 * rather than each caller checking after every line.
 *
 * The conventions this decode depends on, and where they came from, are in the
 * README. The two that fit nowhere else are repeated at their use sites.
 */
#include "ltx_audio.h"

#include "h3_safetensors.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    STEREO = 2,
    Z_CHANNELS = 8, LATENT_MELS = 16, MEL_BINS = 64,
    BASE_CHANNELS = 128, WIDEST_CHANNELS = 512,
    LEVELS = 3, BLOCKS_PER_LEVEL = 3, KERNEL = 3,
    VOICE_CHANNELS = STEREO * MEL_BINS,
    INITIAL_CHANNELS = 1536,
    STAGES = 6, RESBLOCKS = 3, RESIDUAL_PAIRS = 3,
    HEAD_KERNEL = 7, FILTER_TAPS = 12,
    /* Mel frames per latent row, and the three the causal stack cannot make. */
    ROW_MEL_FRAMES = 4, CAUSAL_SHORTFALL = 3
};

#define PIXEL_NORM_EPSILON 1e-8f

static const uint32_t upsample_rates[STAGES] = {5, 2, 2, 2, 2, 2};
static const uint32_t upsample_kernels[STAGES] = {11, 4, 4, 4, 4, 4};
static const uint32_t residual_kernels[RESBLOCKS] = {3, 7, 11};
static const uint32_t residual_dilations[RESIDUAL_PAIRS] = {1, 3, 5};

uint32_t ltx_audio_rows_for(int pixel_frames, int fps) {
    if (pixel_frames <= 0 || fps <= 0) return 0;
    /* round(pixel_frames / fps * 25), in integer arithmetic. */
    const long latents_per_second =
        LTX_AUDIO_SAMPLE_RATE / LTX_AUDIO_HOP / ROW_MEL_FRAMES;
    return (uint32_t)((2L * pixel_frames * latents_per_second + fps) /
                      (2L * fps));
}

uint32_t ltx_audio_frames_for(uint32_t rows) {
    if (!rows) return 0;
    return (rows * ROW_MEL_FRAMES - CAUSAL_SHORTFALL) * LTX_AUDIO_HOP;
}

/* ---------------------------------------------------------------- the run */

/* How many frees can pile up inside one open command buffer. Two per
 * convolution and a handful of convolutions between a begin and a submit, so
 * this is an order of magnitude of slack rather than a tuned bound. */
enum { RETIRE_MAX = 32 };

typedef struct {
    h3_gpu *gpu;
    const h3_weight_store *store;
    h3_gpu_tensor *ones, *zeros;
    h3_gpu_tensor *upsample_filter, *downsample_filter;
    char *error;
    size_t error_size;
    int failed;
    /* Tensors the open command buffer still reads. See `retire`. */
    h3_gpu_tensor *retired[RETIRE_MAX];
    int retired_count;
} run;

static void oops(run *r, const char *format, ...) {
    if (r->failed) return;                 /* keep the first, which is the cause */
    r->failed = 1;
    if (!r->error || !r->error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(r->error, r->error_size, format, arguments);
    va_end(arguments);
}

#define GPU_OP(r, call, what) \
    do { if (!(r)->failed && !(call)) \
             oops((r), "%s: %s", (what), h3_gpu_error((r)->gpu)); } while (0)

typedef struct { uint32_t frames, mels, channels; } shape;
typedef struct { uint32_t length, channels; } span;

static size_t volume(shape of) {
    return (size_t)of.frames * of.mels * of.channels;
}
static size_t positions(shape of) {
    return (size_t)of.frames * of.mels;
}
static size_t extent(span of) {
    return (size_t)of.length * of.channels;
}

/* ----------------------------------------------------------------- loading */

/* Freeing a tensor marks its Metal buffer purgeable **immediately**, so a
 * tensor the open command buffer still reads cannot be freed until that buffer
 * is submitted -- the commit fails validation with "volatile or empty
 * purgeable state at commit".
 *
 * That is invisible without the validation layer, which the command-line
 * harnesses run without and the app turns on. It is why this file's header
 * says to submit before freeing anything encoded work still reads, and why
 * saying it was not enough: `run_conv` frees its padded input one line after
 * encoding the convolution that reads it. Deferring is the fix that does not
 * depend on remembering. */
static void retire(run *r, h3_gpu_tensor *tensor) {
    if (!tensor) return;
    if (r->retired_count < RETIRE_MAX) {
        r->retired[r->retired_count++] = tensor;
        return;
    }
    /* Not expected to happen; freeing early beats leaking. */
    h3_gpu_tensor_free(tensor);
}

static void drain(run *r) {
    for (int index = 0; index < r->retired_count; index++)
        h3_gpu_tensor_free(r->retired[index]);
    r->retired_count = 0;
}

static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float *read_tensor(run *r, const char *name, size_t expected) {
    if (r->failed) return NULL;
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(r->store, name, &header);
    if (!tensor) { oops(r, "no tensor %s", name); return NULL; }
    const size_t elements = (size_t)h3_st_tensor_elements(tensor);
    if (expected && elements != expected) {
        oops(r, "%s has %zu elements, expected %zu", name, elements, expected);
        return NULL;
    }
    float *values = malloc(elements * sizeof(*values));
    if (!values) { oops(r, "cannot allocate %s", name); return NULL; }
    char why[256];
    if (tensor->dtype == H3_DTYPE_F32) {
        if (!h3_st_read_data(header, tensor, values, elements * sizeof(*values),
                             why, sizeof(why))) {
            oops(r, "cannot read %s: %s", name, why);
            free(values);
            return NULL;
        }
    } else if (tensor->dtype == H3_DTYPE_BF16) {
        uint16_t *raw = malloc(elements * sizeof(*raw));
        if (!raw) { oops(r, "cannot stage %s", name); free(values); return NULL; }
        if (!h3_st_read_data(header, tensor, raw, elements * sizeof(*raw),
                             why, sizeof(why))) {
            oops(r, "cannot read %s: %s", name, why);
            free(raw); free(values);
            return NULL;
        }
        for (size_t index = 0; index < elements; index++)
            values[index] = from_bf16(raw[index]);
        free(raw);
    } else {
        oops(r, "%s is %s", name, h3_dtype_name(tensor->dtype));
        free(values);
        return NULL;
    }
    return values;
}

static h3_gpu_tensor *upload(run *r, const float *values, size_t count) {
    if (r->failed) return NULL;
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_f32(r->gpu, values, count);
    if (!tensor) oops(r, "cannot upload %zu floats: %s", count,
                      h3_gpu_error(r->gpu));
    return tensor;
}

static h3_gpu_tensor *allocate(run *r, size_t count) {
    if (r->failed) return NULL;
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_f32(r->gpu, count);
    if (!tensor) oops(r, "cannot allocate %zu floats: %s", count,
                      h3_gpu_error(r->gpu));
    return tensor;
}

/* -------------------------------------------------------- the VAE decoder */

typedef struct { h3_gpu_tensor *weight, *bias; uint32_t kernel; } conv2d;

/* Every load happens before `h3_gpu_begin`: an upload into an already-open
 * command buffer is not seen by work encoded after it, and the convolution
 * quietly reads its input as zeros. */
static void load_vae_conv(run *r, const char *name, uint32_t out_channels,
                          uint32_t in_channels, uint32_t kernel, conv2d *into) {
    memset(into, 0, sizeof(*into));
    char full[256];
    const size_t taps = (size_t)kernel * kernel;
    snprintf(full, sizeof(full), "audio_vae.decoder.%s.conv.weight", name);
    float *weight = read_tensor(r, full,
                                (size_t)out_channels * in_channels * taps);
    if (weight) {
        into->weight = upload(r, weight,
                              (size_t)out_channels * in_channels * taps);
        free(weight);
    }
    snprintf(full, sizeof(full), "audio_vae.decoder.%s.conv.bias", name);
    float *bias = read_tensor(r, full, out_channels);
    if (bias) {
        into->bias = upload(r, bias, out_channels);
        free(bias);
    }
    into->kernel = kernel;
}

static void free_vae_conv(conv2d *which) {
    h3_gpu_tensor_free(which->weight);
    h3_gpu_tensor_free(which->bias);
    memset(which, 0, sizeof(*which));
}

/* Two zero frames in front and none behind. The fill is *zeros*, not
 * replicated edge frames -- the video VAE replicates and this one does not,
 * and nothing but the reference says so. */
static h3_gpu_tensor *pad_frames(run *r, const h3_gpu_tensor *input, shape of,
                                 uint32_t kernel) {
    const uint32_t front = kernel - 1;
    const size_t frame = (size_t)of.mels * of.channels;
    h3_gpu_tensor *out = allocate(r, (size_t)(of.frames + front) * frame);
    if (!out) return NULL;
    if (front)
        GPU_OP(r, h3_gpu_copy_f32(r->gpu, out, 0, r->zeros, 0,
                                  (size_t)front * frame), "zero the causal pad");
    GPU_OP(r, h3_gpu_copy_f32(r->gpu, out, (size_t)front * frame, input, 0,
                              (size_t)of.frames * frame), "pad interior");
    return out;
}

/* Frames onto depth, mel onto width, a degenerate 1 onto height: depth is left
 * unpadded for the explicit causal pad while mel is same-padded by the kernel. */
static h3_gpu_tensor *run_vae_conv(run *r, const conv2d *kernel,
                                   const h3_gpu_tensor *input, shape of,
                                   uint32_t out_channels) {
    h3_gpu_tensor *padded = pad_frames(r, input, of, kernel->kernel);
    shape result = {of.frames, of.mels, out_channels};
    h3_gpu_tensor *out = allocate(r, volume(result));
    if (out)
        GPU_OP(r, h3_gpu_conv3d_same_f32(r->gpu, out, padded, kernel->weight,
                                         kernel->bias, 1,
                                         of.frames + kernel->kernel - 1, 1,
                                         of.mels, of.channels, out_channels,
                                         kernel->kernel, 1, kernel->kernel,
                                         1, 1, 1), "audio VAE convolution");
    retire(r, padded);
    return out;
}

static void pixel_norm(run *r, h3_gpu_tensor *out, const h3_gpu_tensor *input,
                       shape of) {
    GPU_OP(r, h3_gpu_rms_norm_f32(r->gpu, out, input, r->ones,
                                  (uint32_t)positions(of), of.channels,
                                  PIXEL_NORM_EPSILON), "pixel norm");
}

static h3_gpu_tensor *resnet(run *r, const char *prefix, h3_gpu_tensor *x,
                             shape of, uint32_t out_channels) {
    char name[192];
    conv2d first, second, shortcut;
    memset(&shortcut, 0, sizeof(shortcut));
    snprintf(name, sizeof(name), "%s.conv1", prefix);
    load_vae_conv(r, name, out_channels, of.channels, KERNEL, &first);
    snprintf(name, sizeof(name), "%s.conv2", prefix);
    load_vae_conv(r, name, out_channels, out_channels, KERNEL, &second);
    if (of.channels != out_channels) {
        snprintf(name, sizeof(name), "%s.nin_shortcut", prefix);
        load_vae_conv(r, name, out_channels, of.channels, 1, &shortcut);
    }
    shape wide = {of.frames, of.mels, out_channels};
    h3_gpu_tensor *scratch = allocate(r, volume(of));
    h3_gpu_tensor *inner = allocate(r, volume(wide));
    h3_gpu_tensor *hidden = NULL, *branch = NULL, *residual = NULL;

    GPU_OP(r, h3_gpu_begin(r->gpu), "begin resnet");
    pixel_norm(r, scratch, x, of);
    GPU_OP(r, h3_gpu_silu_f32(r->gpu, scratch, scratch, (uint32_t)volume(of)),
           "resnet activation");
    hidden = run_vae_conv(r, &first, scratch, of, out_channels);
    pixel_norm(r, inner, hidden, wide);
    GPU_OP(r, h3_gpu_silu_f32(r->gpu, inner, inner, (uint32_t)volume(wide)),
           "resnet activation");
    branch = run_vae_conv(r, &second, inner, wide, out_channels);
    /* The residual takes the block's *input*, projected when the width changes
     * -- not the normalized copy above. */
    residual = x;
    if (of.channels != out_channels)
        residual = run_vae_conv(r, &shortcut, x, of, out_channels);
    GPU_OP(r, h3_gpu_add_scaled_f32(r->gpu, branch, residual, branch, 1.0f,
                                    1.0f, (uint32_t)volume(wide)),
           "resnet residual");
    GPU_OP(r, h3_gpu_submit(r->gpu), "submit resnet");
    drain(r);

    if (residual != x) h3_gpu_tensor_free(residual);
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(inner);
    h3_gpu_tensor_free(scratch);
    h3_gpu_tensor_free(x);
    free_vae_conv(&first); free_vae_conv(&second);
    if (shortcut.weight) free_vae_conv(&shortcut);
    return branch;
}

/* Nearest-2x on both axes, a causal convolution, then discard the *first*
 * frame -- which is what keeps the length at 1 + 2n. Dropping the trailing one
 * instead keeps the same shape and is wrong. */
static h3_gpu_tensor *upsample(run *r, int level, h3_gpu_tensor *x, shape *of) {
    char name[192];
    conv2d kernel;
    snprintf(name, sizeof(name), "up.%d.upsample.conv", level);
    load_vae_conv(r, name, of->channels, of->channels, KERNEL, &kernel);

    shape doubled = {of->frames * 2, of->mels * 2, of->channels};
    h3_gpu_tensor *big = allocate(r, volume(doubled));
    GPU_OP(r, h3_gpu_begin(r->gpu), "begin upsample");
    GPU_OP(r, h3_gpu_nearest2x_nhwc_f32(r->gpu, big, x, of->frames, of->mels,
                                        of->channels), "nearest upsample");
    h3_gpu_tensor *convolved = run_vae_conv(r, &kernel, big, doubled,
                                            of->channels);
    GPU_OP(r, h3_gpu_submit(r->gpu), "submit upsample");
    drain(r);
    free_vae_conv(&kernel);
    h3_gpu_tensor_free(big);
    h3_gpu_tensor_free(x);

    shape kept = {doubled.frames - 1, doubled.mels, doubled.channels};
    h3_gpu_tensor *out = allocate(r, volume(kept));
    GPU_OP(r, h3_gpu_begin(r->gpu), "begin trim");
    GPU_OP(r, h3_gpu_copy_f32(r->gpu, out, 0, convolved,
                              (size_t)doubled.mels * doubled.channels,
                              volume(kept)), "drop the first frame");
    GPU_OP(r, h3_gpu_submit(r->gpu), "submit trim");
    drain(r);
    h3_gpu_tensor_free(convolved);
    *of = kept;
    return out;
}

/* ------------------------------------------------------------ the vocoder */

typedef struct {
    h3_gpu_tensor *weight, *bias;
    uint32_t in, out, kernel, padding, dilation, stride;
    int transpose;
} conv1d;

typedef struct { h3_gpu_tensor *alpha, *beta; } activation;

typedef struct {
    activation acts1[RESIDUAL_PAIRS], acts2[RESIDUAL_PAIRS];
    conv1d convs1[RESIDUAL_PAIRS], convs2[RESIDUAL_PAIRS];
} resblock;

typedef struct { conv1d up; resblock blocks[RESBLOCKS]; } stage;

static void load_voc_conv(run *r, const char *name, conv1d *into, uint32_t in,
                          uint32_t out, uint32_t kernel, uint32_t padding,
                          uint32_t dilation, uint32_t stride, int transpose,
                          int biased) {
    memset(into, 0, sizeof(*into));
    char full[256];
    const size_t taps = (size_t)in * out * kernel;
    snprintf(full, sizeof(full), "vocoder.vocoder.%s.weight", name);
    float *weight = read_tensor(r, full, taps);
    if (weight) { into->weight = upload(r, weight, taps); free(weight); }
    if (biased) {
        snprintf(full, sizeof(full), "vocoder.vocoder.%s.bias", name);
        float *bias = read_tensor(r, full, out);
        if (bias) { into->bias = upload(r, bias, out); free(bias); }
    } else {
        /* `use_bias_at_final` is false, so `conv_post` has no bias tensor at
         * all. The kernel still wants one; zeros are the same convolution. */
        float *zero = calloc(out, sizeof(*zero));
        if (!zero) oops(r, "cannot allocate a zero bias");
        else { into->bias = upload(r, zero, out); free(zero); }
    }
    into->in = in; into->out = out; into->kernel = kernel;
    into->padding = padding; into->dilation = dilation; into->stride = stride;
    into->transpose = transpose;
}

static void free_voc_conv(conv1d *which) {
    h3_gpu_tensor_free(which->weight);
    h3_gpu_tensor_free(which->bias);
    memset(which, 0, sizeof(*which));
}

static void load_activation(run *r, const char *name, activation *into,
                            uint32_t channels) {
    memset(into, 0, sizeof(*into));
    char full[256];
    snprintf(full, sizeof(full), "vocoder.vocoder.%s.act.alpha", name);
    float *alpha = read_tensor(r, full, channels);
    if (alpha) { into->alpha = upload(r, alpha, channels); free(alpha); }
    snprintf(full, sizeof(full), "vocoder.vocoder.%s.act.beta", name);
    float *beta = read_tensor(r, full, channels);
    if (beta) { into->beta = upload(r, beta, channels); free(beta); }
}

static void free_activation(activation *which) {
    h3_gpu_tensor_free(which->alpha);
    h3_gpu_tensor_free(which->beta);
    memset(which, 0, sizeof(*which));
}

static void run_activation(run *r, h3_gpu_tensor *out,
                           const h3_gpu_tensor *in, const activation *act,
                           span of) {
    GPU_OP(r, h3_gpu_alias_free_snake_f32(r->gpu, out, in, act->alpha,
                                          act->beta, r->upsample_filter,
                                          r->downsample_filter, 1, of.length,
                                          of.channels),
           "alias-free SnakeBeta");
}

static void run_voc_conv(run *r, h3_gpu_tensor *out, const h3_gpu_tensor *in,
                         const conv1d *conv, uint32_t length) {
    if (conv->transpose)
        GPU_OP(r, h3_gpu_conv_transpose1d_f32(r->gpu, out, in, conv->weight,
                                              conv->bias, 1, length, conv->in,
                                              conv->out, conv->kernel,
                                              conv->stride, conv->padding),
               "vocoder transposed convolution");
    else
        GPU_OP(r, h3_gpu_conv1d_f32(r->gpu, out, in, conv->weight, conv->bias,
                                    1, length, conv->in, conv->out,
                                    conv->kernel, conv->padding,
                                    conv->dilation), "vocoder convolution");
}

/* All 109 activations share one filter pair -- verified tap by tap in
 * `test_real_ltx_vocoder.c`, taken on trust here. */
static void load_filters(run *r) {
    float *up = read_tensor(r, "vocoder.vocoder.act_post.upsample.filter",
                            FILTER_TAPS);
    float *down = read_tensor(
        r, "vocoder.vocoder.act_post.downsample.lowpass.filter", FILTER_TAPS);
    if (up) { r->upsample_filter = upload(r, up, FILTER_TAPS); free(up); }
    if (down) { r->downsample_filter = upload(r, down, FILTER_TAPS); free(down); }
}

/* `get_padding(k, d) = d * (k - 1) / 2`; convs2 is always dilation 1. */
static void load_stage(run *r, stage *into, int index) {
    memset(into, 0, sizeof(*into));
    const uint32_t in = INITIAL_CHANNELS >> index;
    const uint32_t out = INITIAL_CHANNELS >> (index + 1);
    const uint32_t rate = upsample_rates[index];
    const uint32_t kernel = upsample_kernels[index];
    char name[192];
    snprintf(name, sizeof(name), "ups.%d", index);
    load_voc_conv(r, name, &into->up, in, out, kernel, (kernel - rate) / 2, 1,
                  rate, 1, 1);
    for (int block = 0; block < RESBLOCKS; block++) {
        const int global = index * RESBLOCKS + block;
        const uint32_t residual = residual_kernels[block];
        for (int pair = 0; pair < RESIDUAL_PAIRS; pair++) {
            const uint32_t dilation = residual_dilations[pair];
            snprintf(name, sizeof(name), "resblocks.%d.acts1.%d", global, pair);
            load_activation(r, name, &into->blocks[block].acts1[pair], out);
            snprintf(name, sizeof(name), "resblocks.%d.acts2.%d", global, pair);
            load_activation(r, name, &into->blocks[block].acts2[pair], out);
            snprintf(name, sizeof(name), "resblocks.%d.convs1.%d", global, pair);
            load_voc_conv(r, name, &into->blocks[block].convs1[pair], out, out,
                          residual, dilation * (residual - 1) / 2, dilation, 1,
                          0, 1);
            snprintf(name, sizeof(name), "resblocks.%d.convs2.%d", global, pair);
            load_voc_conv(r, name, &into->blocks[block].convs2[pair], out, out,
                          residual, (residual - 1) / 2, 1, 1, 0, 1);
        }
    }
}

static void free_stage(stage *which) {
    free_voc_conv(&which->up);
    for (int block = 0; block < RESBLOCKS; block++)
        for (int pair = 0; pair < RESIDUAL_PAIRS; pair++) {
            free_activation(&which->blocks[block].acts1[pair]);
            free_activation(&which->blocks[block].acts2[pair]);
            free_voc_conv(&which->blocks[block].convs1[pair]);
            free_voc_conv(&which->blocks[block].convs2[pair]);
        }
    memset(which, 0, sizeof(*which));
}

static void run_resblock(run *r, h3_gpu_tensor *out, const h3_gpu_tensor *in,
                         const resblock *block, span of,
                         h3_gpu_tensor *activated, h3_gpu_tensor *branch) {
    GPU_OP(r, h3_gpu_copy_f32(r->gpu, out, 0, in, 0, extent(of)),
           "seed the residual");
    for (int pair = 0; pair < RESIDUAL_PAIRS; pair++) {
        run_activation(r, activated, out, &block->acts1[pair], of);
        run_voc_conv(r, branch, activated, &block->convs1[pair], of.length);
        run_activation(r, activated, branch, &block->acts2[pair], of);
        run_voc_conv(r, branch, activated, &block->convs2[pair], of.length);
        GPU_OP(r, h3_gpu_add_scaled_f32(r->gpu, out, out, branch, 1.0f, 1.0f,
                                        (uint32_t)extent(of)), "residual add");
    }
}

/* The three blocks read the same input and the stage takes their *mean*.
 * Summing is a factor of three and still sounds like something. */
static void encode_blocks(run *r, h3_gpu_tensor *out, const h3_gpu_tensor *in,
                          const stage *weights, span of, h3_gpu_tensor *work,
                          h3_gpu_tensor *activated, h3_gpu_tensor *branch) {
    for (int block = 0; block < RESBLOCKS; block++) {
        h3_gpu_tensor *target = block == 0 ? out : work;
        run_resblock(r, target, in, &weights->blocks[block], of, activated,
                     branch);
        if (block == 0) continue;
        const float scale =
            block == RESBLOCKS - 1 ? 1.0f / (float)RESBLOCKS : 1.0f;
        GPU_OP(r, h3_gpu_add_scaled_f32(r->gpu, out, out, work, scale, scale,
                                        (uint32_t)extent(of)), "resblock mean");
    }
}

/* ------------------------------------------------------------------ decode */

int ltx_audio_decode(h3_gpu *gpu, const h3_weight_store *store,
                     const float *tokens, uint32_t rows, float *samples,
                     char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!gpu || !store || !tokens || !samples || !rows) {
        if (error && error_size)
            snprintf(error, error_size, "ltx_audio_decode wants a gpu, a "
                                        "store, a latent and somewhere to put "
                                        "the sound");
        return 0;
    }
    run r = {0};
    r.gpu = gpu; r.store = store; r.error = error; r.error_size = error_size;

    h3_gpu_tensor *x = NULL;
    float *staged = NULL, *helper = NULL, *mel = NULL, *folded = NULL;

    /* ------------------------------------- unpatchify and denormalize */

    /* The statistics are 128 wide while the latent is 8 channels, because the
     * normalization lives in the *patchified* space the DiT works in:
     * `[8, T, 16]` reads as `[T, 128]` with channel index `c * 16 + f`. A port
     * that takes the shapes at face value does not even fit. */
    float *latent_mean = read_tensor(
        &r, "audio_vae.per_channel_statistics.mean-of-means", LTX_AUDIO_PATCHED);
    float *latent_std = read_tensor(
        &r, "audio_vae.per_channel_statistics.std-of-means", LTX_AUDIO_PATCHED);

    shape at = {rows, LATENT_MELS, Z_CHANNELS};
    if (!r.failed) {
        staged = malloc(volume(at) * sizeof(*staged));
        if (!staged) oops(&r, "cannot allocate the staged latent");
    }
    if (!r.failed)
        for (uint32_t frame = 0; frame < rows; frame++)
            for (uint32_t bin = 0; bin < LATENT_MELS; bin++)
                for (uint32_t channel = 0; channel < Z_CHANNELS; channel++) {
                    const uint32_t patched = channel * LATENT_MELS + bin;
                    staged[((size_t)frame * LATENT_MELS + bin) * Z_CHANNELS +
                           channel] =
                        tokens[(size_t)frame * LTX_AUDIO_PATCHED + patched] *
                        latent_std[patched] + latent_mean[patched];
                }
    free(latent_mean); free(latent_std);

    /* Two frames of the widest stage for the causal pad, and a norm weight of
     * ones. The widest is an upsample's *output*: nearest-2x doubles the mel
     * axis while leaving the channel count alone. */
    if (!r.failed) {
        helper = calloc((size_t)MEL_BINS * WIDEST_CHANNELS, sizeof(*helper));
        if (!helper) oops(&r, "cannot allocate a pad buffer");
    }
    if (!r.failed) {
        r.zeros = upload(&r, helper, (size_t)MEL_BINS * WIDEST_CHANNELS);
        for (int index = 0; index < WIDEST_CHANNELS; index++)
            helper[index] = 1.0f;
        r.ones = upload(&r, helper, WIDEST_CHANNELS);
    }
    free(helper);

    x = upload(&r, staged, volume(at));
    free(staged);

    /* ------------------------------------------------- the VAE decoder */

    {
        conv2d kernel;
        load_vae_conv(&r, "conv_in", WIDEST_CHANNELS, Z_CHANNELS, KERNEL,
                      &kernel);
        GPU_OP(&r, h3_gpu_begin(gpu), "begin conv_in");
        h3_gpu_tensor *out = run_vae_conv(&r, &kernel, x, at, WIDEST_CHANNELS);
        GPU_OP(&r, h3_gpu_submit(gpu), "submit conv_in");
        drain(&r);
        free_vae_conv(&kernel);
        h3_gpu_tensor_free(x);
        x = out;
        at.channels = WIDEST_CHANNELS;
    }
    /* `mid_block_add_attention` is false, so the mid block is two residuals
     * and the checkpoint carries no attention tensors at all. */
    x = resnet(&r, "mid.block_1", x, at, at.channels);
    x = resnet(&r, "mid.block_2", x, at, at.channels);

    static const uint32_t LEVEL_CHANNELS[LEVELS] = {
        BASE_CHANNELS, BASE_CHANNELS * 2, BASE_CHANNELS * 4
    };
    for (int level = LEVELS - 1; level >= 0 && !r.failed; level--) {
        char prefix[192];
        for (int block = 0; block < BLOCKS_PER_LEVEL; block++) {
            snprintf(prefix, sizeof(prefix), "up.%d.block.%d", level, block);
            x = resnet(&r, prefix, x, at, LEVEL_CHANNELS[level]);
            at.channels = LEVEL_CHANNELS[level];
        }
        if (level != 0) x = upsample(&r, level, x, &at);
    }

    {
        h3_gpu_tensor *normed = allocate(&r, volume(at));
        GPU_OP(&r, h3_gpu_begin(gpu), "begin tail");
        pixel_norm(&r, normed, x, at);
        GPU_OP(&r, h3_gpu_submit(gpu), "submit norm");
        drain(&r);
        h3_gpu_tensor_free(x);
        x = normed;
    }
    {
        conv2d tail;
        load_vae_conv(&r, "conv_out", STEREO, at.channels, KERNEL, &tail);
        GPU_OP(&r, h3_gpu_begin(gpu), "begin head");
        GPU_OP(&r, h3_gpu_silu_f32(gpu, x, x, (uint32_t)volume(at)),
               "output activation");
        h3_gpu_tensor *out = run_vae_conv(&r, &tail, x, at, STEREO);
        GPU_OP(&r, h3_gpu_submit(gpu), "submit head");
        drain(&r);
        free_vae_conv(&tail);
        h3_gpu_tensor_free(x);
        x = out;
        at.channels = STEREO;
    }
    if (!r.failed && (at.frames != rows * ROW_MEL_FRAMES - CAUSAL_SHORTFALL ||
                      at.mels != MEL_BINS))
        oops(&r, "the decode reached [%u, %u] rather than the mel geometry",
             at.frames, at.mels);

    /* -------------------------------------------------------- the vocoder */

    /* `(stereo, frames, mels)` folds to `(stereo * mels, frames)`: the reason
     * `conv_pre` takes 128 channels rather than 64. The VAE left it
     * channels-last, so stereo is the fastest axis on the way in. */
    span voice = {at.frames, VOICE_CHANNELS};
    if (!r.failed) {
        mel = malloc(volume(at) * sizeof(*mel));
        if (!mel) oops(&r, "cannot allocate the mel readback");
        else if (!h3_gpu_tensor_read_f32(x, mel, volume(at)))
            oops(&r, "cannot read the mel back");
    }
    h3_gpu_tensor_free(x);
    x = NULL;
    if (!r.failed) {
        folded = malloc(extent(voice) * sizeof(*folded));
        if (!folded) oops(&r, "cannot allocate the folded mel");
    }
    if (!r.failed)
        for (uint32_t frame = 0; frame < at.frames; frame++)
            for (uint32_t side = 0; side < STEREO; side++)
                for (uint32_t bin = 0; bin < MEL_BINS; bin++)
                    folded[(size_t)frame * VOICE_CHANNELS + side * MEL_BINS +
                           bin] =
                        mel[((size_t)frame * MEL_BINS + bin) * STEREO + side];
    free(mel);
    x = upload(&r, folded, extent(voice));
    free(folded);

    load_filters(&r);
    {
        conv1d pre;
        load_voc_conv(&r, "conv_pre", &pre, VOICE_CHANNELS, INITIAL_CHANNELS,
                      HEAD_KERNEL, HEAD_KERNEL / 2, 1, 1, 0, 1);
        span wide = {voice.length, INITIAL_CHANNELS};
        h3_gpu_tensor *out = allocate(&r, extent(wide));
        GPU_OP(&r, h3_gpu_begin(gpu), "begin conv_pre");
        run_voc_conv(&r, out, x, &pre, voice.length);
        GPU_OP(&r, h3_gpu_submit(gpu), "submit conv_pre");
        drain(&r);
        free_voc_conv(&pre);
        h3_gpu_tensor_free(x);
        x = out;
        voice = wide;
    }

    for (int index = 0; index < STAGES && !r.failed; index++) {
        stage weights;
        load_stage(&r, &weights, index);
        const uint32_t stride = upsample_rates[index];
        const uint32_t kernel = upsample_kernels[index];
        const uint32_t padding = (kernel - stride) / 2;
        span wider = {(voice.length - 1) * stride + kernel - 2 * padding,
                      INITIAL_CHANNELS >> (index + 1)};
        h3_gpu_tensor *upsampled = allocate(&r, extent(wider));
        h3_gpu_tensor *accumulated = allocate(&r, extent(wider));
        h3_gpu_tensor *work = allocate(&r, extent(wider));
        h3_gpu_tensor *activated = allocate(&r, extent(wider));
        h3_gpu_tensor *branch = allocate(&r, extent(wider));

        GPU_OP(&r, h3_gpu_begin(gpu), "begin vocoder stage");
        run_voc_conv(&r, upsampled, x, &weights.up, voice.length);
        encode_blocks(&r, accumulated, upsampled, &weights, wider, work,
                      activated, branch);
        GPU_OP(&r, h3_gpu_submit(gpu), "submit vocoder stage");
        drain(&r);

        h3_gpu_tensor_free(x);
        x = accumulated;
        voice = wider;
        h3_gpu_tensor_free(upsampled);
        h3_gpu_tensor_free(work);
        h3_gpu_tensor_free(activated);
        h3_gpu_tensor_free(branch);
        free_stage(&weights);
    }

    {
        activation post;
        load_activation(&r, "act_post", &post, voice.channels);
        h3_gpu_tensor *out = allocate(&r, extent(voice));
        GPU_OP(&r, h3_gpu_begin(gpu), "begin act_post");
        run_activation(&r, out, x, &post, voice);
        GPU_OP(&r, h3_gpu_submit(gpu), "submit act_post");
        drain(&r);
        free_activation(&post);
        h3_gpu_tensor_free(x);
        x = out;
    }
    {
        conv1d post;
        load_voc_conv(&r, "conv_post", &post, voice.channels, STEREO,
                      HEAD_KERNEL, HEAD_KERNEL / 2, 1, 1, 0, 0);
        span narrow = {voice.length, STEREO};
        h3_gpu_tensor *out = allocate(&r, extent(narrow));
        GPU_OP(&r, h3_gpu_begin(gpu), "begin conv_post");
        run_voc_conv(&r, out, x, &post, voice.length);
        GPU_OP(&r, h3_gpu_submit(gpu), "submit conv_post");
        drain(&r);
        free_voc_conv(&post);
        h3_gpu_tensor_free(x);
        x = out;
        voice = narrow;
    }

    if (!r.failed && voice.length != ltx_audio_frames_for(rows))
        oops(&r, "the vocoder produced %u samples, expected %u", voice.length,
             ltx_audio_frames_for(rows));
    if (!r.failed && !h3_gpu_tensor_read_f32(x, samples, extent(voice)))
        oops(&r, "cannot read the waveform back");
    /* `apply_final_activation` is true and `use_tanh_at_final` is false, so
     * `forward` ends in a clamp. In-distribution content does not reach it --
     * 256x of input gain moves the peak from 0.230 to 0.491 -- but it is in
     * the model and a caller downstream may not clamp. */
    if (!r.failed)
        for (size_t index = 0; index < extent(voice); index++) {
            if (samples[index] > 1.0f) samples[index] = 1.0f;
            if (samples[index] < -1.0f) samples[index] = -1.0f;
        }

    h3_gpu_tensor_free(x);
    h3_gpu_tensor_free(r.ones);
    h3_gpu_tensor_free(r.zeros);
    h3_gpu_tensor_free(r.upsample_filter);
    h3_gpu_tensor_free(r.downsample_filter);
    return !r.failed;
}
