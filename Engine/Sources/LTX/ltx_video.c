/* LTX-2.5's video VAE decoder, as a library stage.
 *
 * Lifted from `Vendor/h3.c/tests/ltx_decode.c`, which keeps both halves and the
 * stage comparisons and stays the place to reproduce a disagreement. Only the
 * decoding half is here -- the encoder exists to condition on an existing clip,
 * which nothing calls yet -- and the exits are error returns, as in
 * `ltx_audio.c`.
 *
 * Three conventions the shapes do not give, each of which produces a plausible
 * image when read the other way:
 *
 *   - **the temporal padding replicates and the spatial padding is zeros.**
 *     The reference concatenates edge frames explicitly and then applies
 *     `nn.Conv3d(padding=(0, 1, 1))`, so depth is handled by hand and height
 *     and width by the kernel;
 *
 *   - **the upsample discards its first output frame** whenever the temporal
 *     factor is 2, unconditionally, which is why 2 latent frames become 3
 *     rather than 4; and
 *
 *   - **the patch channel order is (c, t, w, h), not (c, t, h, w).** The
 *     reference reads `(c p r q) -> (f p) (h q) (w r)`, so the fastest-varying
 *     channel index lands on *height*. Backwards transposes every 4x4 patch,
 *     which still looks like an image.
 *
 * F32 throughout is not a convenience: activations peak around 1e5 a couple of
 * blocks in and the final pixel norm is what brings that back to single
 * digits. BF16 would not survive the trip.
 */
#include "ltx_video.h"

#include "h3_safetensors.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    LATENT_CHANNELS = LTX_VIDEO_LATENT_CHANNELS,
    /* `latent_log_var: uniform` gives the encoder one channel beyond the
     * means: a single shared log variance the deterministic path never reads.
     */
    MOMENT_CHANNELS = LATENT_CHANNELS + 1,
    PATCH = 4,
    IMAGE_CHANNELS = LTX_VIDEO_CHANNELS,
    PACKED_CHANNELS = IMAGE_CHANNELS * PATCH * PATCH,
    WIDEST_CHANNELS = 1024,
    BLOCKS = 9,
    KERNEL = 3
};

#define PIXEL_NORM_EPSILON 1e-8f

uint32_t ltx_video_pixel_frames(uint32_t latent_frames) {
    return latent_frames ? 8 * (latent_frames - 1) + 1 : 0;
}

/* ---------------------------------------------------------------- the run */

/* How many frees can pile up inside one open command buffer. Two per
 * convolution and a handful of convolutions between a begin and a submit, so
 * this is an order of magnitude of slack rather than a tuned bound. */
enum { RETIRE_MAX = 32 };

typedef struct {
    h3_gpu *gpu;
    const h3_weight_store *store;
    h3_gpu_tensor *ones;
    char *error;
    size_t error_size;
    int failed;
    /* Tensors the open command buffer still reads. See `retire`. */
    h3_gpu_tensor *retired[RETIRE_MAX];
    int retired_count;
} run;

static void oops(run *r, const char *format, ...) {
    if (r->failed) return;
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

/* Channels last: the convolution kernel takes NDHWC and OIDHW, and OIDHW is
 * exactly how torch stores a Conv3d weight, so weights need no rearranging. */
typedef struct { uint32_t depth, height, width, channels; } shape;

static size_t volume(shape of) {
    return (size_t)of.depth * of.height * of.width * of.channels;
}
static size_t positions(shape of) {
    return (size_t)of.depth * of.height * of.width;
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

static float *download(run *r, const h3_gpu_tensor *tensor, size_t count) {
    if (r->failed) return NULL;
    float *values = malloc(count * sizeof(*values));
    if (!values) { oops(r, "cannot allocate a readback buffer"); return NULL; }
    if (!h3_gpu_tensor_read_f32(tensor, values, count)) {
        oops(r, "cannot read a GPU tensor back");
        free(values);
        return NULL;
    }
    return values;
}

typedef struct { h3_gpu_tensor *weight, *bias; } conv3d;

/* Uploading writes through a staging path a *later* dispatch in an already-open
 * command buffer does not see: a convolution encoded after an upload into the
 * same buffer reads the allocation as zeros and emits its bias alone. So every
 * load happens before `h3_gpu_begin`, and the mirror rule applies at the end --
 * submit before freeing anything encoded work still reads. */
static void load_conv(run *r, const char *half, const char *name,
                      uint32_t out_channels, uint32_t in_channels,
                      conv3d *into) {
    memset(into, 0, sizeof(*into));
    char full[256];
    const size_t taps = (size_t)KERNEL * KERNEL * KERNEL;
    snprintf(full, sizeof(full), "%s.%s.conv.weight", half, name);
    float *weight = read_tensor(r, full,
                                (size_t)out_channels * in_channels * taps);
    if (weight) {
        into->weight = upload(r, weight,
                              (size_t)out_channels * in_channels * taps);
        free(weight);
    }
    snprintf(full, sizeof(full), "%s.%s.conv.bias", half, name);
    float *bias = read_tensor(r, full, out_channels);
    if (bias) { into->bias = upload(r, bias, out_channels); free(bias); }
}

static void free_conv(conv3d *which) {
    h3_gpu_tensor_free(which->weight);
    h3_gpu_tensor_free(which->bias);
    memset(which, 0, sizeof(*which));
}

/* ------------------------------------------------------------- the padding */

/* Frames are the outermost axis of a channels-last buffer, so each is a
 * contiguous run and a handful of copies do the whole job. The decoder pads one
 * frame at each end; the encoder (not here) pads two in front. */
static h3_gpu_tensor *pad_time(run *r, const h3_gpu_tensor *input, shape of,
                               int causal) {
    const size_t frame = (size_t)of.height * of.width * of.channels;
    h3_gpu_tensor *out = allocate(r, (size_t)(of.depth + 2) * frame);
    if (!out) return NULL;
    if (causal) {
        /* The encoder is causal: both pad frames go in *front*, duplicating
         * the first, so no output frame ever reads a later input one. The
         * decoder replicates at both ends instead. That asymmetry is what
         * makes the two halves inverses rather than mirrors. */
        GPU_OP(r, h3_gpu_copy_f32(r->gpu, out, 2 * frame, input, 0,
                                  (size_t)of.depth * frame), "pad interior");
        GPU_OP(r, h3_gpu_copy_f32(r->gpu, out, 0, input, 0, frame),
               "pad first");
        GPU_OP(r, h3_gpu_copy_f32(r->gpu, out, frame, input, 0, frame),
               "pad second");
        return out;
    }
    GPU_OP(r, h3_gpu_copy_f32(r->gpu, out, frame, input, 0,
                              (size_t)of.depth * frame), "pad interior");
    GPU_OP(r, h3_gpu_copy_f32(r->gpu, out, 0, input, 0, frame), "pad front");
    GPU_OP(r, h3_gpu_copy_f32(r->gpu, out, (size_t)(of.depth + 1) * frame,
                              input, (size_t)(of.depth - 1) * frame, frame),
           "pad back");
    return out;
}

/* Replicate in time, zeros in height and width -- the kernel same-pads the
 * spatial axes and leaves depth alone, which is exactly the split the
 * reference's explicit temporal concatenation plus `padding=(0, 1, 1)` gives. */
static h3_gpu_tensor *run_conv(run *r, const conv3d *kernel,
                               const h3_gpu_tensor *input, shape of,
                               uint32_t out_channels, int causal) {
    h3_gpu_tensor *padded = pad_time(r, input, of, causal);
    shape result = {of.depth, of.height, of.width, out_channels};
    h3_gpu_tensor *out = allocate(r, volume(result));
    if (out)
        GPU_OP(r, h3_gpu_conv3d_same_f32(r->gpu, out, padded, kernel->weight,
                                         kernel->bias, 1, of.depth + 2,
                                         of.height, of.width, of.channels,
                                         out_channels, KERNEL, KERNEL, KERNEL,
                                         1, 1, 1), "video VAE convolution");
    retire(r, padded);
    return out;
}

/* x / sqrt(mean(x^2 over channels) + eps): the RMS norm kernel with a weight of
 * ones, the epsilon inside the root in both. The checkpoint carries no norm
 * tensors at all, which is the tell that it is parameter-free. */
static void pixel_norm(run *r, h3_gpu_tensor *out, const h3_gpu_tensor *input,
                       shape of) {
    GPU_OP(r, h3_gpu_rms_norm_f32(r->gpu, out, input, r->ones,
                                  (uint32_t)positions(of), of.channels,
                                  PIXEL_NORM_EPSILON), "pixel norm");
}

/* --------------------------------------------------- packing on the host */

/* The inverse of `depth_to_space`: `b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3)
 * d h w`, channels last. Same index order, read the other way -- the depth
 * factor stays the slowest of the three sub-indices, which is the detail that
 * makes the two exact inverses rather than merely similar. */
static void space_to_depth(const float *input, float *out, shape of,
                           uint32_t depth_down, uint32_t height_down,
                           uint32_t width_down) {
    const uint32_t factor = depth_down * height_down * width_down;
    const uint32_t out_depth = of.depth / depth_down;
    const uint32_t out_height = of.height / height_down;
    const uint32_t out_width = of.width / width_down;
    const uint32_t out_channels = of.channels * factor;
    for (uint32_t d = 0; d < out_depth; d++)
        for (uint32_t h = 0; h < out_height; h++)
            for (uint32_t w = 0; w < out_width; w++) {
                float *target = out +
                    (((size_t)d * out_height + h) * out_width + w) * out_channels;
                for (uint32_t c = 0; c < of.channels; c++)
                    for (uint32_t i = 0; i < depth_down; i++)
                        for (uint32_t j = 0; j < height_down; j++)
                            for (uint32_t k = 0; k < width_down; k++) {
                                const size_t at =
                                    ((((size_t)(d * depth_down + i) * of.height +
                                       h * height_down + j) * of.width +
                                      w * width_down + k) * of.channels) + c;
                                target[((c * depth_down + i) * height_down + j) *
                                       width_down + k] = input[at];
                            }
            }
}

/* The downsample's skip path averages each group of `group` channels down to
 * one, which is how the widened stack meets the narrowed convolution. */
static void group_mean(const float *input, float *out, size_t count,
                       uint32_t width, uint32_t group) {
    const uint32_t narrow = width / group;
    for (size_t row = 0; row < count; row++)
        for (uint32_t channel = 0; channel < narrow; channel++) {
            double sum = 0.0;
            for (uint32_t index = 0; index < group; index++)
                sum += input[row * width + channel * group + index];
            out[row * narrow + channel] = (float)(sum / (double)group);
        }
}

/* `b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)`, channels last. The
 * channel index decomposes major-to-minor as (c, p1, p2, p3), so the depth
 * factor is the *slowest* of the three sub-indices. */
static void depth_to_space(const float *input, float *out, shape of,
                           uint32_t depth_up, uint32_t height_up,
                           uint32_t width_up) {
    const uint32_t factor = depth_up * height_up * width_up;
    const uint32_t out_channels = of.channels / factor;
    const uint32_t out_height = of.height * height_up;
    const uint32_t out_width = of.width * width_up;
    for (uint32_t d = 0; d < of.depth; d++)
        for (uint32_t h = 0; h < of.height; h++)
            for (uint32_t w = 0; w < of.width; w++) {
                const float *source = input +
                    (((size_t)d * of.height + h) * of.width + w) * of.channels;
                for (uint32_t c = 0; c < out_channels; c++)
                    for (uint32_t i = 0; i < depth_up; i++)
                        for (uint32_t j = 0; j < height_up; j++)
                            for (uint32_t k = 0; k < width_up; k++) {
                                const uint32_t from =
                                    ((c * depth_up + i) * height_up + j) *
                                    width_up + k;
                                const size_t to =
                                    (((size_t)(d * depth_up + i) * out_height +
                                      h * height_up + j) * out_width +
                                     w * width_up + k) * out_channels + c;
                                out[to] = source[from];
                            }
            }
}

/* `(c p r q) -> (f p) (h q) (w r)`: the fastest-varying channel index lands on
 * height and the next on width. Writes channel-major, which is what a caller
 * turning frames into an image or a video wants. */
static void unpatchify(const float *input, float *out, shape of) {
    const uint32_t channels = of.channels / (PATCH * PATCH);
    const uint32_t height = of.height * PATCH, width = of.width * PATCH;
    const size_t plane = (size_t)of.depth * height * width;
    for (uint32_t d = 0; d < of.depth; d++)
        for (uint32_t h = 0; h < of.height; h++)
            for (uint32_t w = 0; w < of.width; w++) {
                const float *source = input +
                    (((size_t)d * of.height + h) * of.width + w) * of.channels;
                for (uint32_t c = 0; c < channels; c++)
                    for (uint32_t r = 0; r < PATCH; r++)
                        for (uint32_t q = 0; q < PATCH; q++)
                            out[(size_t)c * plane +
                                ((size_t)d * height + h * PATCH + q) * width +
                                w * PATCH + r] =
                                source[(c * PATCH + r) * PATCH + q];
            }
}

/* --------------------------------------------------------------- the blocks */

/* Frames in [C][D][H][W] to the packed [D][H][W][C'] the encoder's first
 * convolution reads: each `PATCH`-square tile of every frame becomes one
 * position whose channels run (c, y, x). The decoder's `unpatchify` undoes
 * exactly this. */
static void patchify(const float *image, float *out, shape packed) {
    const uint32_t height = packed.height * PATCH, width = packed.width * PATCH;
    const size_t plane = (size_t)packed.depth * height * width;
    for (uint32_t d = 0; d < packed.depth; d++)
        for (uint32_t h = 0; h < packed.height; h++)
            for (uint32_t w = 0; w < packed.width; w++) {
                float *target = out +
                    (((size_t)d * packed.height + h) * packed.width + w) *
                    packed.channels;
                for (uint32_t c = 0; c < IMAGE_CHANNELS; c++)
                    for (uint32_t r = 0; r < PATCH; r++)
                        for (uint32_t q = 0; q < PATCH; q++)
                            /* The sub-index that moves fastest in the *target*
                             * is the patch's row, and the one that selects the
                             * channel group is its column -- the tile is
                             * transposed on the way in. Writing it the
                             * intuitive way round packs a plausible tensor
                             * that the first convolution then reads across. */
                            target[(c * PATCH + r) * PATCH + q] =
                                image[(size_t)c * plane +
                                      ((size_t)d * height + h * PATCH + q) *
                                      width + w * PATCH + r];
            }
}

typedef struct {
    const char *kind;
    int layers;                 /* res_x only */
    int multiplier;             /* compress only */
    uint32_t depth_step, height_step, width_step;
} vae_block;

/* `decoder_blocks` from the checkpoint config, reversed -- the decoder walks it
 * backwards. Written out rather than derived so it can be read against the
 * config by eye. */
static const vae_block DECODE[BLOCKS] = {
    {"res_x",          2, 0, 0, 0, 0},
    {"compress_all",   0, 2, 2, 2, 2},
    {"res_x",          2, 0, 0, 0, 0},
    {"compress_all",   0, 1, 2, 2, 2},
    {"res_x",          4, 0, 0, 0, 0},
    {"compress_time",  0, 2, 2, 1, 1},
    {"res_x",          6, 0, 0, 0, 0},
    {"compress_space", 0, 2, 1, 2, 2},
    {"res_x",          4, 0, 0, 0, 0}
};

/* pixel norm, SiLU, conv, again, then add the input back -- in place. Every
 * res_x block keeps its channel count, so the reference's shortcut projection
 * and its third norm are both Identity, which is why the checkpoint carries
 * neither. */
/* `encoder_blocks` from the checkpoint config, in order. The decoder's list
 * reversed is not the same thing: the encoder's compressions carry a residual
 * the decoder's expansions do not, which is what the `_res` suffix names. */
static const vae_block ENCODE[BLOCKS] = {
    {"res_x",              4, 0, 0, 0, 0},
    {"compress_space_res", 0, 2, 1, 2, 2},
    {"res_x",              6, 0, 0, 0, 0},
    {"compress_time_res",  0, 2, 2, 1, 1},
    {"res_x",              4, 0, 0, 0, 0},
    {"compress_all_res",   0, 2, 2, 2, 2},
    {"res_x",              2, 0, 0, 0, 0},
    {"compress_all_res",   0, 1, 2, 2, 2},
    {"res_x",              2, 0, 0, 0, 0}
};

static void resnet(run *r, const char *half, int block, int layer,
                   h3_gpu_tensor *x, shape of, int causal) {
    char name[160];
    conv3d first, second;
    snprintf(name, sizeof(name), "%s_blocks.%d.res_blocks.%d.conv1",
             causal ? "down" : "up", block, layer);
    load_conv(r, half, name, of.channels, of.channels, &first);
    snprintf(name, sizeof(name), "%s_blocks.%d.res_blocks.%d.conv2",
             causal ? "down" : "up", block, layer);
    load_conv(r, half, name, of.channels, of.channels, &second);
    h3_gpu_tensor *scratch = allocate(r, volume(of));

    GPU_OP(r, h3_gpu_begin(r->gpu), "begin resnet");
    pixel_norm(r, scratch, x, of);
    GPU_OP(r, h3_gpu_silu_f32(r->gpu, scratch, scratch, (uint32_t)volume(of)),
           "resnet activation");
    h3_gpu_tensor *hidden = run_conv(r, &first, scratch, of, of.channels, causal);
    pixel_norm(r, scratch, hidden, of);
    GPU_OP(r, h3_gpu_silu_f32(r->gpu, scratch, scratch, (uint32_t)volume(of)),
           "resnet activation");
    h3_gpu_tensor *second_out = run_conv(r, &second, scratch, of, of.channels, causal);
    GPU_OP(r, h3_gpu_add_scaled_f32(r->gpu, x, x, second_out, 1.0f, 1.0f,
                                    (uint32_t)volume(of)), "resnet residual");
    GPU_OP(r, h3_gpu_submit(r->gpu), "submit resnet");
    drain(r);
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(second_out);
    h3_gpu_tensor_free(scratch);
    free_conv(&first); free_conv(&second);
}

/* Convolve up to `prod(stride) * channels / multiplier`, spread those channels
 * out into space and time, then -- when the temporal factor is 2 -- discard the
 * first output frame. The discard is unconditional in the reference, so 2
 * latent frames become 3 rather than 4. */
static h3_gpu_tensor *upsample(run *r, int block, const vae_block *plan,
                               h3_gpu_tensor *x, shape *of) {
    char name[160];
    const uint32_t factor = plan->depth_step * plan->height_step *
                            plan->width_step;
    const uint32_t wide = of->channels * factor / (uint32_t)plan->multiplier;
    snprintf(name, sizeof(name), "up_blocks.%d.conv", block);
    conv3d kernel;
    load_conv(r, "decoder", name, wide, of->channels, &kernel);
    GPU_OP(r, h3_gpu_begin(r->gpu), "begin upsample");
    h3_gpu_tensor *convolved = run_conv(r, &kernel, x, *of, wide, 0);
    GPU_OP(r, h3_gpu_submit(r->gpu), "submit upsample");
    drain(r);
    free_conv(&kernel);
    h3_gpu_tensor_free(x);

    shape wide_shape = {of->depth, of->height, of->width, wide};
    float *host = download(r, convolved, volume(wide_shape));
    h3_gpu_tensor_free(convolved);
    shape spread = {of->depth * plan->depth_step,
                    of->height * plan->height_step,
                    of->width * plan->width_step, wide / factor};
    float *expanded = NULL;
    if (!r->failed) {
        expanded = malloc(volume(spread) * sizeof(*expanded));
        if (!expanded) oops(r, "cannot allocate an expanded stack");
    }
    if (!r->failed)
        depth_to_space(host, expanded, wide_shape, plan->depth_step,
                       plan->height_step, plan->width_step);
    free(host);

    const float *keep = expanded;
    if (plan->depth_step == 2 && expanded) {
        keep = expanded + (size_t)spread.height * spread.width * spread.channels;
        spread.depth -= 1;
    }
    h3_gpu_tensor *out = upload(r, keep, volume(spread));
    free(expanded);
    *of = spread;
    return out;
}

/* ------------------------------------------------------------------ decode */

/* The encoder's compression: a strided convolution and a skip, both packed
 * down by `space_to_depth`, added. The skip averages groups of channels to
 * meet the convolution's narrower output — that averaged residual is the whole
 * difference between these blocks and the decoder's expansions. */
static h3_gpu_tensor *downsample(run *r, int block, const vae_block *plan,
                                 h3_gpu_tensor *x, shape *of) {
    char name[160];
    const uint32_t factor = plan->depth_step * plan->height_step *
                            plan->width_step;
    const uint32_t out_channels = of->channels * (uint32_t)plan->multiplier;
    const uint32_t group = of->channels * factor / out_channels;
    const uint32_t narrow = out_channels / factor;

    shape extended = *of;
    h3_gpu_tensor *source = x;
    if (plan->depth_step == 2) {
        /* A duplicated leading frame, so an odd frame count halves cleanly and
         * the first output frame still sees only itself. */
        const size_t frame = (size_t)of->height * of->width * of->channels;
        extended.depth += 1;
        source = allocate(r, (size_t)extended.depth * frame);
        if (!source) return NULL;
        GPU_OP(r, h3_gpu_begin(r->gpu), "begin frame duplication");
        GPU_OP(r, h3_gpu_copy_f32(r->gpu, source, frame, x, 0,
                                  (size_t)of->depth * frame), "extend interior");
        GPU_OP(r, h3_gpu_copy_f32(r->gpu, source, 0, x, 0, frame),
               "extend front");
        GPU_OP(r, h3_gpu_submit(r->gpu), "submit frame duplication");
        drain(r);
        h3_gpu_tensor_free(x);
    }

    snprintf(name, sizeof(name), "down_blocks.%d.conv", block);
    conv3d kernel;
    load_conv(r, "encoder", name, narrow, extended.channels, &kernel);
    GPU_OP(r, h3_gpu_begin(r->gpu), "begin downsample");
    h3_gpu_tensor *convolved = run_conv(r, &kernel, source, extended, narrow, 1);
    GPU_OP(r, h3_gpu_submit(r->gpu), "submit downsample");
    drain(r);
    free_conv(&kernel);

    shape packed = {extended.depth / plan->depth_step,
                    extended.height / plan->height_step,
                    extended.width / plan->width_step, out_channels};
    float *skip = NULL, *branch = NULL;
    if (!r->failed) {
        skip = malloc(volume(packed) * sizeof(*skip));
        branch = malloc(volume(packed) * sizeof(*branch));
        if (!skip || !branch) oops(r, "cannot allocate a downsample stack");
    }
    if (!r->failed) {
        float *host = download(r, source, volume(extended));
        float *wide = malloc(positions(packed) * extended.channels * factor *
                             sizeof(*wide));
        if (!host || !wide) oops(r, "cannot allocate the skip stack");
        else {
            space_to_depth(host, wide, extended, plan->depth_step,
                           plan->height_step, plan->width_step);
            group_mean(wide, skip, positions(packed),
                       extended.channels * factor, group);
        }
        free(wide); free(host);
    }
    if (!r->failed) {
        shape narrow_shape = {extended.depth, extended.height, extended.width,
                              narrow};
        float *host = download(r, convolved, volume(narrow_shape));
        if (host) {
            space_to_depth(host, branch, narrow_shape, plan->depth_step,
                           plan->height_step, plan->width_step);
            free(host);
        }
    }
    h3_gpu_tensor_free(source);
    h3_gpu_tensor_free(convolved);
    h3_gpu_tensor *out = NULL;
    if (!r->failed) {
        for (size_t index = 0; index < volume(packed); index++)
            branch[index] += skip[index];
        out = upload(r, branch, volume(packed));
    }
    free(skip); free(branch);
    *of = packed;
    return out;
}

/* `H3_LTX_DUMP=/prefix_` writes each encoder stage as raw F32, the way the
 * vocoder runner does. The halves of this VAE are long enough that "the answer
 * is wrong" localizes nothing on its own. */
static void dump_stage(run *r, const char *label, const h3_gpu_tensor *tensor,
                       shape of) {
    const char *prefix = getenv("H3_LTX_DUMP");
    if (!prefix || !*prefix || r->failed) return;
    float *host = download(r, tensor, volume(of));
    if (!host) return;
    char path[512];
    snprintf(path, sizeof(path), "%s%s.bin", prefix, label);
    FILE *out = fopen(path, "wb");
    if (out) {
        fwrite(host, sizeof(float), volume(of), out);
        fclose(out);
    }
    free(host);
}

int ltx_video_encode(h3_gpu *gpu, const h3_weight_store *store,
                     const float *pixels, uint32_t frames, uint32_t height,
                     uint32_t width, float *latent,
                     char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!gpu || !store || !pixels || !latent || !frames || !height || !width) {
        if (error && error_size)
            snprintf(error, error_size, "ltx_video_encode wants a gpu, a "
                                        "store, frames and somewhere to put "
                                        "the latent");
        return 0;
    }
    if (height % LTX_VIDEO_SPATIAL || width % LTX_VIDEO_SPATIAL) {
        if (error && error_size)
            snprintf(error, error_size, "%ux%u is not a multiple of %d",
                     width, height, LTX_VIDEO_SPATIAL);
        return 0;
    }
    if (frames != 1 && (frames - 1) % 8) {
        if (error && error_size)
            snprintf(error, error_size, "%u frames is not 1 or 8k+1", frames);
        return 0;
    }
    run r = {0};
    r.gpu = gpu; r.store = store; r.error = error; r.error_size = error_size;

    /* F32, because this half runs in F32 throughout and `h3_gpu_rms_norm_f32`
     * reads its weight as F32 — a BF16 buffer here is reinterpreted rather
     * than converted, and every pixel norm quietly divides by nonsense. */
    float *ones = malloc(WIDEST_CHANNELS * sizeof(*ones));
    if (!ones) { oops(&r, "cannot allocate the norm weight"); return 0; }
    for (int index = 0; index < WIDEST_CHANNELS; index++) ones[index] = 1.0f;
    r.ones = upload(&r, ones, WIDEST_CHANNELS);
    free(ones);
    if (!r.ones) return 0;

    shape at = {frames, height / PATCH, width / PATCH, PACKED_CHANNELS};
    h3_gpu_tensor *x = NULL;
    {
        float *staged = malloc(volume(at) * sizeof(*staged));
        if (!staged) oops(&r, "cannot allocate the patchified frames");
        else {
            patchify(pixels, staged, at);
            x = upload(&r, staged, volume(at));
            free(staged);
        }
    }
    if (!r.failed) {
        conv3d kernel;
        load_conv(&r, "encoder", "conv_in", 128, PACKED_CHANNELS, &kernel);
        GPU_OP(&r, h3_gpu_begin(gpu), "begin enc conv_in");
        h3_gpu_tensor *out = run_conv(&r, &kernel, x, at, 128, 1);
        GPU_OP(&r, h3_gpu_submit(gpu), "submit enc conv_in");
        drain(&r);
        free_conv(&kernel);
        h3_gpu_tensor_free(x);
        x = out;
        at.channels = 128;
    }
    dump_stage(&r, "enc_conv_in", x, at);

    for (int index = 0; index < BLOCKS && !r.failed; index++) {
        const vae_block *plan = &ENCODE[index];
        if (plan->layers)
            for (int layer = 0; layer < plan->layers && !r.failed; layer++)
                resnet(&r, "encoder", index, layer, x, at, 1);
        else
            x = downsample(&r, index, plan, x, &at);
        char label[64];
        snprintf(label, sizeof(label), "enc_block%d_%s", index, plan->kind);
        dump_stage(&r, label, x, at);
    }

    h3_gpu_tensor *normed = allocate(&r, volume(at));
    if (normed) {
        GPU_OP(&r, h3_gpu_begin(gpu), "begin enc tail");
        pixel_norm(&r, normed, x, at);
        GPU_OP(&r, h3_gpu_submit(gpu), "submit enc norm");
        drain(&r);
    }
    dump_stage(&r, "enc_norm_out", normed, at);
    h3_gpu_tensor_free(x);
    x = NULL;

    h3_gpu_tensor *moments = NULL;
    if (!r.failed) {
        conv3d tail;
        load_conv(&r, "encoder", "conv_out", MOMENT_CHANNELS, at.channels,
                  &tail);
        GPU_OP(&r, h3_gpu_begin(gpu), "begin enc head");
        GPU_OP(&r, h3_gpu_silu_f32(gpu, normed, normed, (uint32_t)volume(at)),
               "encoder output activation");
        moments = run_conv(&r, &tail, normed, at, MOMENT_CHANNELS, 1);
        GPU_OP(&r, h3_gpu_submit(gpu), "submit enc head");
        drain(&r);
        free_conv(&tail);
    }
    h3_gpu_tensor_free(normed);
    at.channels = MOMENT_CHANNELS;

    /* The leading 128 channels are the means and the last is the shared log
     * variance, which the deterministic path drops. Normalizing by the
     * checkpoint's per-channel statistics is what puts the result in the space
     * the DiT works in -- and it is invisible to a round trip, because the
     * decoder's inverse cancels it exactly. */
    float *host = download(&r, moments, volume(at));
    h3_gpu_tensor_free(moments);
    float *mean = read_tensor(&r, "per_channel_statistics.mean-of-means",
                              LATENT_CHANNELS);
    float *deviation = read_tensor(
        &r, "per_channel_statistics.std-of-means", LATENT_CHANNELS);
    if (!r.failed && host && mean && deviation) {
        const shape out_shape = {at.depth, at.height, at.width, LATENT_CHANNELS};
        for (size_t position = 0; position < positions(out_shape); position++)
            for (uint32_t channel = 0; channel < LATENT_CHANNELS; channel++)
                latent[position * LATENT_CHANNELS + channel] =
                    (host[position * MOMENT_CHANNELS + channel] -
                     mean[channel]) / deviation[channel];
    }
    free(host); free(mean); free(deviation);
    h3_gpu_tensor_free(r.ones);
    return !r.failed;
}

int ltx_video_decode(h3_gpu *gpu, const h3_weight_store *store,
                     const float *latent, uint32_t frames, uint32_t height,
                     uint32_t width, float *pixels,
                     char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!gpu || !store || !latent || !pixels || !frames || !height || !width) {
        if (error && error_size)
            snprintf(error, error_size, "ltx_video_decode wants a gpu, a "
                                        "store, a latent and somewhere to put "
                                        "the frames");
        return 0;
    }
    run r = {0};
    r.gpu = gpu; r.store = store; r.error = error; r.error_size = error_size;

    shape at = {frames, height, width, LATENT_CHANNELS};
    h3_gpu_tensor *x = NULL;
    float *denormalized = NULL, *helper = NULL;

    /* The DiT works in the *normalized* latent space and the VAE is what maps
     * out of it. std spans 0.074 to 0.914 and mean -0.55 to +0.43, so a decoder
     * without this is wrong by up to 13x per channel plus an offset -- and no
     * round trip can catch it, because the encoder's inverse cancels it
     * exactly. */
    float *latent_mean = read_tensor(&r, "per_channel_statistics.mean-of-means",
                                     LATENT_CHANNELS);
    float *latent_std = read_tensor(&r, "per_channel_statistics.std-of-means",
                                    LATENT_CHANNELS);
    if (!r.failed) {
        denormalized = malloc(volume(at) * sizeof(*denormalized));
        if (!denormalized) oops(&r, "cannot allocate the denormalized latent");
    }
    if (!r.failed)
        for (size_t position = 0; position < positions(at); position++)
            for (uint32_t channel = 0; channel < LATENT_CHANNELS; channel++)
                denormalized[position * LATENT_CHANNELS + channel] =
                    latent[position * LATENT_CHANNELS + channel] *
                    latent_std[channel] + latent_mean[channel];
    free(latent_mean); free(latent_std);

    if (!r.failed) {
        helper = malloc(WIDEST_CHANNELS * sizeof(*helper));
        if (!helper) oops(&r, "cannot allocate a norm weight");
        else for (int index = 0; index < WIDEST_CHANNELS; index++)
            helper[index] = 1.0f;
    }
    r.ones = upload(&r, helper, WIDEST_CHANNELS);
    free(helper);

    x = upload(&r, denormalized, volume(at));
    free(denormalized);

    {
        conv3d kernel;
        load_conv(&r, "decoder", "conv_in", WIDEST_CHANNELS, at.channels, &kernel);
        GPU_OP(&r, h3_gpu_begin(gpu), "begin conv_in");
        h3_gpu_tensor *out = run_conv(&r, &kernel, x, at, WIDEST_CHANNELS, 0);
        GPU_OP(&r, h3_gpu_submit(gpu), "submit conv_in");
        drain(&r);
        free_conv(&kernel);
        h3_gpu_tensor_free(x);
        x = out;
        at.channels = WIDEST_CHANNELS;
    }

    for (int index = 0; index < BLOCKS && !r.failed; index++) {
        const vae_block *plan = &DECODE[index];
        if (plan->layers)
            for (int layer = 0; layer < plan->layers; layer++)
                resnet(&r, "decoder", index, layer, x, at, 0);
        else
            x = upsample(&r, index, plan, x, &at);
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
        conv3d tail;
        load_conv(&r, "decoder", "conv_out", PACKED_CHANNELS, at.channels, &tail);
        GPU_OP(&r, h3_gpu_begin(gpu), "begin head");
        GPU_OP(&r, h3_gpu_silu_f32(gpu, x, x, (uint32_t)volume(at)),
               "output activation");
        h3_gpu_tensor *out = run_conv(&r, &tail, x, at, PACKED_CHANNELS, 0);
        GPU_OP(&r, h3_gpu_submit(gpu), "submit head");
        drain(&r);
        free_conv(&tail);
        h3_gpu_tensor_free(x);
        x = out;
        at.channels = PACKED_CHANNELS;
    }

    /* Three temporal doublings with a discarded leading frame at each, which
     * is what makes 8(n-1)+1 rather than 8n. Asserted because the block plan
     * and this arithmetic are written down separately and have to agree. */
    if (!r.failed && at.depth != ltx_video_pixel_frames(frames))
        oops(&r, "the decode reached %u frames, expected %u", at.depth,
             ltx_video_pixel_frames(frames));

    float *host = download(&r, x, volume(at));
    h3_gpu_tensor_free(x);
    if (!r.failed) unpatchify(host, pixels, at);
    free(host);

    h3_gpu_tensor_free(r.ones);
    return !r.failed;
}
