/* LTX-2.5's dual-stream DiT and its sampler, as a library stage.
 *
 * Lifted from `Vendor/h3.c/tests/ltx_generate.c`, which keeps the reference
 * comparisons and stays the place to reproduce a disagreement. Unlike the four
 * stages before it this is not a pure lift: the driver takes its geometry from
 * -D flags because the block helpers and one stack array were sized by them,
 * and the service cannot ship a binary per clip length. So the shape lives in
 * `run` and is threaded, the exits are error returns, and the prefetch worker
 * carries its own error latch rather than writing the shared one.
 *
 * Four findings are load bearing here, each of which produced a well formed
 * result while being wrong:
 *
 *   - **the rope reads pixel coordinates, and its time axis is in seconds.**
 *     Latent cells squeeze every position into 8/2048 of the range and the
 *     model cannot tell one column from the next -- 32-pixel blocking, exactly
 *     one latent cell wide. Frames instead of seconds keeps the picture and
 *     desynchronizes the soundtrack, because the two streams attend to each
 *     other through these positions and so must share units;
 *
 *   - **each cross-modal gate reads the *other* stream's sigma.** Swap them
 *     and a run where both streams share a sigma agrees exactly, which is why
 *     no component test could see it;
 *
 *   - **the audio row count is derived, not chosen.** It follows the video's
 *     duration in seconds; see `ltx_audio_rows_for`; and
 *
 *   - **the head's modulation is per token.** `_process_output` indexes the
 *     embedded timestep per token, so a held token's head modulation is the
 *     one built at timestep zero rather than the one the frame uses.
 */
#include "ltx_dit.h"

#include "h3_safetensors.h"

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    VIDEO_DIM = LTX_DIT_VIDEO_DIM,
    AUDIO_DIM = LTX_DIT_AUDIO_DIM,
    HEADS = 32,
    VIDEO_FF = VIDEO_DIM * 4,
    AUDIO_FF = AUDIO_DIM * 4,
    LATENT = LTX_DIT_LATENT,
    TIMESTEP_FEATURES = 256,
    ADA_SLOTS = 9,
    PROMPT_SLOTS = 2,
    CROSS_SLOTS = 4,
    CROSS_TABLE_SLOTS = 5,
    GATE_SLOTS = 1,
    HEAD_SLOTS = 2,
    CONVROT_GROUP = 256,
    TOTAL_BLOCKS = LTX_DIT_BLOCKS,
    MAX_STEPS = LTX_DIT_MAX_STEPS,
    VIDEO_AXES = 3,
    AUDIO_AXES = 1,
    /* Mel frames per audio latent frame: the audio VAE compresses 4x. */
    AUDIO_DOWNSAMPLE = 4,
    /* Two modulation rows per module: one for tokens a mask holds at zero, one
     * for the tokens being denoised. Nothing is held here -- both streams
     * denoise from noise -- so the two rows carry the same value and the row
     * map stays at zero. The pair is kept because the layout is what
     * `h3_adaln_bf16` indexes and because conditioning will need it. */
    LEVELS = 2,
    MODULE_COUNT = 8,
    /* A ceiling on the context, not its length: the span is the tokenizer's
     * (256 for `LTXVGemmaTokenizer`'s default) and arrives as an argument.
     * This only refuses a garbage one before it sizes an allocation. */
    MAX_CONTEXT = 4096
};

#define BLOCK_EPSILON 1e-6f
#define TIMESTEP_MULTIPLIER 1000.0
#define MAX_PERIOD 10000.0
/* hop over sample rate: 160 / 16000. */
#define AUDIO_SECONDS 0.01f

static const double VIDEO_MAX_POS[VIDEO_AXES] = {20.0, 2048.0, 2048.0};
static const double AUDIO_MAX_POS[AUDIO_AXES] = {20.0};
static const double ROPE_THETA = 10000.0;

/* ---------------------------------------------------------------- the run */

typedef struct {
    h3_gpu *gpu;
    const h3_weight_store *store;
    /* Latent geometry, which the driver had as compile-time constants. */
    uint32_t frames, height, width;
    uint32_t video_rows, audio_rows, text_rows;
    int fps;
    char *error;
    size_t error_size;
    int failed;
    int cancelled;
} run;

static void oops(run *r, const char *format, ...) {
    if (r->failed) return;   /* keep the first, which is the cause */
    r->failed = 1;
    if (!r->error || !r->error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(r->error, r->error_size, format, arguments);
    va_end(arguments);
}

/* Stops the run without an error, which is how the caller's tick refuses. The
 * empty message is the signal: `ltx_dit_sample` returns 0 either way. */
static void stop(run *r) {
    if (r->failed) return;
    r->failed = 1;
    r->cancelled = 1;
}

#define GPU_OP(r, call, what) \
    do { if (!(r)->failed && !(call)) \
             oops((r), "%s: %s", (what), h3_gpu_error((r)->gpu)); } while (0)

static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* Round to nearest even, as the GPU's own narrowing does. */
static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

/* ------------------------------------------------------------------ loading */

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

/* Everything in this checkpoint sits under the one prefix. */
static float *weight_of(run *r, const char *name, size_t expected) {
    char full[256];
    snprintf(full, sizeof(full), "model.diffusion_model.%s", name);
    return read_tensor(r, full, expected);
}

static h3_gpu_tensor *upload(run *r, const float *values, size_t count) {
    if (r->failed) return NULL;
    uint16_t *staged = malloc(count * sizeof(*staged));
    if (!staged) { oops(r, "cannot allocate a staging buffer"); return NULL; }
    for (size_t index = 0; index < count; index++)
        staged[index] = to_bf16(values[index]);
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(r->gpu, staged, count);
    free(staged);
    if (!tensor) oops(r, "cannot upload a tensor");
    return tensor;
}

static void read_bf16_as_f32(run *r, const h3_gpu_tensor *tensor, float *out,
                             size_t count) {
    if (r->failed) return;
    uint16_t *staged = malloc(count * sizeof(*staged));
    if (!staged) { oops(r, "cannot allocate a readback buffer"); return; }
    if (!h3_gpu_tensor_read_bf16(tensor, staged, count))
        oops(r, "cannot read a GPU tensor back");
    else
        for (size_t index = 0; index < count; index++)
            out[index] = from_bf16(staged[index]);
    free(staged);
}

/* -------------------------------------------------------- the timestep path */

static float silu(float value) { return value / (1.0f + expf(-value)); }

static void linear_rows(float *out, const float *in, const float *weight,
                        const float *bias, uint32_t rows, uint32_t in_dim,
                        uint32_t out_dim) {
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < out_dim; column++) {
            float sum = bias ? bias[column] : 0.0f;
            const float *w = weight + (size_t)column * in_dim;
            const float *x = in + (size_t)row * in_dim;
            for (uint32_t index = 0; index < in_dim; index++)
                sum = fmaf(x[index], w[index], sum);
            out[(size_t)row * out_dim + column] = sum;
        }
    }
}

/* The argument reaches 732 radians at sigma 0.732, so this is F32 whatever the
 * rest of the model is. Cosine first, and the exponent divides by half rather
 * than half - 1. */
static void sinusoid(float *out, double timestep) {
    const int half = TIMESTEP_FEATURES / 2;
    for (int index = 0; index < half; index++) {
        const double exponent = -log(MAX_PERIOD) * (double)index / (double)half;
        const double angle = timestep * exp(exponent);
        out[index] = (float)cos(angle);
        out[half + index] = (float)sin(angle);
    }
}

/* Which of the four noise quantities a module is driven by. Under a scalar
 * sigma with no mask all four collapse to the same number, which is why the
 * distinction only shows up in a whole-path run. */
typedef enum {
    VIDEO_TOKENS,   /* video's per-token timesteps, sigma * mask */
    AUDIO_TOKENS,
    VIDEO_SIGMA,    /* video's scalar noise level, zero when frozen */
    AUDIO_SIGMA
} timestep_source;

typedef struct {
    const char *name;
    uint32_t width, slots;
    timestep_source source;
} adaln_module;

/* Which noise quantity drives each module. Everything else here is machinery;
 * this table is the finding. Note the last two: each gate reads the *cross*
 * stream's sigma, so the gate on audio-into-video asks how noisy the *audio*
 * is. */
static const adaln_module MODULES[MODULE_COUNT] = {
    {"adaln_single", VIDEO_DIM, ADA_SLOTS, VIDEO_TOKENS},
    {"audio_adaln_single", AUDIO_DIM, ADA_SLOTS, AUDIO_TOKENS},
    {"prompt_adaln_single", VIDEO_DIM, PROMPT_SLOTS, VIDEO_SIGMA},
    {"audio_prompt_adaln_single", AUDIO_DIM, PROMPT_SLOTS, AUDIO_SIGMA},
    {"av_ca_video_scale_shift_adaln_single", VIDEO_DIM, CROSS_SLOTS, VIDEO_TOKENS},
    {"av_ca_audio_scale_shift_adaln_single", AUDIO_DIM, CROSS_SLOTS, AUDIO_TOKENS},
    {"av_ca_a2v_gate_adaln_single", VIDEO_DIM, GATE_SLOTS, AUDIO_SIGMA},
    {"av_ca_v2a_gate_adaln_single", AUDIO_DIM, GATE_SLOTS, VIDEO_SIGMA}
};

/* Evaluate one module at `count` timesteps in a single pass over its weights.
 * The projection is 4096x36864 for the largest of them, so reading it once per
 * step rather than once per timestep is the difference between a few hundred
 * megabytes of I/O and a few gigabytes.
 *
 * `modulation` comes back as [level][slot][width] and `embedded` as
 * [level][width], which is the layout h3_adaln_bf16 indexes: it reads
 * `row_map[token] * slots * width` as the base of a token's modulation. */
static void run_adaln(run *r, const adaln_module *module,
                      const double *timesteps, uint32_t count,
                      double multiplier, float **modulation, float **embedded) {
    char name[256];
    const size_t width = module->width;
    if (r->failed) return;
    float *features = malloc((size_t)count * TIMESTEP_FEATURES * sizeof(*features));
    if (!features) { oops(r, "cannot allocate the timestep features"); return; }
    for (uint32_t level = 0; level < count; level++)
        sinusoid(features + (size_t)level * TIMESTEP_FEATURES,
                 timesteps[level] * multiplier);
    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_1.weight",
             module->name);
    float *first = weight_of(r, name, width * TIMESTEP_FEATURES);
    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_1.bias",
             module->name);
    float *first_bias = weight_of(r, name, module->width);
    float *hidden = malloc((size_t)count * width * sizeof(*hidden));
    if (!hidden) oops(r, "cannot allocate a timestep hidden state");
    if (!r->failed) {
        linear_rows(hidden, features, first, first_bias, count,
                    TIMESTEP_FEATURES, module->width);
        for (size_t index = 0; index < (size_t)count * width; index++)
            hidden[index] = silu(hidden[index]);
    }
    free(first); free(first_bias); free(features);
    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_2.weight",
             module->name);
    float *second = weight_of(r, name, width * width);
    snprintf(name, sizeof(name), "%s.emb.timestep_embedder.linear_2.bias",
             module->name);
    float *second_bias = weight_of(r, name, module->width);
    float *activated = NULL;
    if (!r->failed) {
        *embedded = malloc((size_t)count * width * sizeof(**embedded));
        activated = malloc((size_t)count * width * sizeof(*activated));
        if (!*embedded || !activated)
            oops(r, "cannot allocate an embedded timestep");
    }
    if (!r->failed) {
        linear_rows(*embedded, hidden, second, second_bias, count,
                    module->width, module->width);
        for (size_t index = 0; index < (size_t)count * width; index++)
            activated[index] = silu((*embedded)[index]);
    }
    free(second); free(second_bias); free(hidden);
    snprintf(name, sizeof(name), "%s.linear.weight", module->name);
    float *projection = weight_of(r, name, (size_t)module->slots * width * width);
    snprintf(name, sizeof(name), "%s.linear.bias", module->name);
    float *projection_bias = weight_of(r, name, (size_t)module->slots * width);
    if (!r->failed) {
        *modulation = malloc((size_t)count * module->slots * width *
                             sizeof(**modulation));
        if (!*modulation) oops(r, "cannot allocate a modulation");
    }
    if (!r->failed)
        linear_rows(*modulation, activated, projection, projection_bias, count,
                    module->width, module->slots * module->width);
    free(projection); free(projection_bias); free(activated);
}

/* ------------------------------------------------------------------- rope */

/* [head][row][half / heads], which is the layout the rotation kernel's head
 * stride walks. The axes interleave -- slot = frequency * axes + axis -- and
 * the padding sits in front. */
static void rope_tables(run *r, h3_gpu_tensor **cosine, h3_gpu_tensor **sine,
                        const float *grid, uint32_t rows, uint32_t dim,
                        int axes, const double *max_pos) {
    if (r->failed) return;
    const uint32_t half = dim / 2;
    const uint32_t per_axis = dim / (2 * (uint32_t)axes);
    const uint32_t pad = half - per_axis * (uint32_t)axes;
    const uint32_t per_head = half / HEADS;
    float *flat_cos = calloc((size_t)rows * half, sizeof(*flat_cos));
    float *flat_sin = calloc((size_t)rows * half, sizeof(*flat_sin));
    float *head_cos = malloc((size_t)rows * half * sizeof(*head_cos));
    float *head_sin = malloc((size_t)rows * half * sizeof(*head_sin));
    if (!flat_cos || !flat_sin || !head_cos || !head_sin) {
        oops(r, "cannot allocate rotary tables");
        free(flat_cos); free(flat_sin); free(head_cos); free(head_sin);
        return;
    }
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t slot = 0; slot < pad; slot++)
            flat_cos[(size_t)row * half + slot] = 1.0f;
        for (int axis = 0; axis < axes; axis++) {
            const float *pair = grid + ((size_t)axis * rows + row) * 2;
            const float middle = (pair[0] + pair[1]) / 2.0f;
            const float position = middle / (float)max_pos[axis] * 2.0f - 1.0f;
            for (uint32_t index = 0; index < per_axis; index++) {
                const double exponent = per_axis > 1 ?
                    (double)index / (double)(per_axis - 1) : 0.0;
                /* Rounded to F32 on purpose: the reference computes these in
                 * double and hands back a float32 tensor, so keeping the
                 * double is more accurate and disagrees. */
                const float frequency =
                    (float)(pow(ROPE_THETA, exponent) * M_PI / 2.0);
                const float angle = frequency * position;
                const size_t at = (size_t)row * half + pad +
                                  (size_t)index * (uint32_t)axes + (uint32_t)axis;
                flat_cos[at] = cosf(angle);
                flat_sin[at] = sinf(angle);
            }
        }
    }
    for (uint32_t head = 0; head < HEADS; head++)
        for (uint32_t row = 0; row < rows; row++)
            for (uint32_t index = 0; index < per_head; index++) {
                const size_t to = ((size_t)head * rows + row) * per_head + index;
                const size_t from = (size_t)row * half + head * per_head + index;
                head_cos[to] = flat_cos[from];
                head_sin[to] = flat_sin[from];
            }
    *cosine = h3_gpu_tensor_from_f32(r->gpu, head_cos, (size_t)rows * half);
    *sine = h3_gpu_tensor_from_f32(r->gpu, head_sin, (size_t)rows * half);
    if (!*cosine || !*sine) oops(r, "cannot upload rotary tables");
    free(flat_cos); free(flat_sin); free(head_cos); free(head_sin);
}

/* ------------------------------------------------------------------ weights */

typedef struct { h3_gpu_tensor *weight, *scales, *bias; } projection;

static void load_projection(run *r, const char *prefix, const char *suffix,
                            uint64_t out_dim, uint64_t in_dim, int bias,
                            projection *into) {
    if (r->failed) return;
    char name[256], why[512];
    snprintf(name, sizeof(name), "%s%s.weight", prefix, suffix);
    if (!h3_weight_load_i8_linear(r->store, r->gpu, name, out_dim, in_dim,
                                  &into->weight, &into->scales,
                                  why, sizeof(why))) {
        oops(r, "cannot load %s: %s", name, why);
        return;
    }
    /* Every quantized projection here is ConvRot at group 256, and the
     * activation is rotated once for the several projections that share it.
     * Rotating for a group one of them disagrees with would corrupt the others
     * quietly rather than fail, so the marker is checked rather than assumed. */
    uint32_t group = 0;
    if (!h3_weight_i8_linear_convrot_group(r->store, name, &group,
                                           why, sizeof(why))) {
        oops(r, "cannot read the ConvRot marker for %s: %s", name, why);
        return;
    }
    if (group != CONVROT_GROUP) {
        oops(r, "%s has ConvRot group %u, expected %d", name, group,
             CONVROT_GROUP);
        return;
    }
    if (bias) {
        char bias_name[256];
        snprintf(bias_name, sizeof(bias_name), "%s%s.bias", prefix, suffix);
        uint64_t shape[1] = {out_dim};
        into->bias = h3_weight_load_bf16(r->store, r->gpu, bias_name, 1, shape,
                                         why, sizeof(why));
        if (!into->bias) oops(r, "cannot load %s: %s", bias_name, why);
    }
}

static void free_projection(projection *which) {
    h3_gpu_tensor_free(which->weight);
    h3_gpu_tensor_free(which->scales);
    h3_gpu_tensor_free(which->bias);
    memset(which, 0, sizeof(*which));
}

typedef struct {
    const char *name;
    uint32_t query_in, kv_in, inner, out_dim, gate_in;
    projection query, key, value, output;
    h3_gpu_tensor *query_norm, *key_norm, *gate_weight, *gate_bias;
} attention;

static void load_attention(run *r, const char *block, attention *into) {
    char prefix[256];
    snprintf(prefix, sizeof(prefix), "%s%s.", block, into->name);
    load_projection(r, prefix, "to_q", into->inner, into->query_in, 1,
                    &into->query);
    load_projection(r, prefix, "to_k", into->inner, into->kv_in, 1, &into->key);
    load_projection(r, prefix, "to_v", into->inner, into->kv_in, 1,
                    &into->value);
    load_projection(r, prefix, "to_out.0", into->out_dim, into->inner, 1,
                    &into->output);
    if (r->failed) return;
    char name[256], why[512];
    uint64_t inner[1] = {into->inner}, heads[1] = {HEADS};
    uint64_t gate[2] = {HEADS, into->gate_in};
    snprintf(name, sizeof(name), "%sq_norm.weight", prefix);
    into->query_norm = h3_weight_load_bf16(r->store, r->gpu, name, 1, inner,
                                           why, sizeof(why));
    snprintf(name, sizeof(name), "%sk_norm.weight", prefix);
    into->key_norm = h3_weight_load_bf16(r->store, r->gpu, name, 1, inner, why,
                                         sizeof(why));
    /* The gate is a small dense projection and reads the *unrotated* input,
     * unlike every quantized projection beside it. */
    snprintf(name, sizeof(name), "%sto_gate_logits.weight", prefix);
    into->gate_weight = h3_weight_load_bf16(r->store, r->gpu, name, 2, gate,
                                            why, sizeof(why));
    snprintf(name, sizeof(name), "%sto_gate_logits.bias", prefix);
    into->gate_bias = h3_weight_load_bf16(r->store, r->gpu, name, 1, heads, why,
                                          sizeof(why));
    if (!into->query_norm || !into->key_norm || !into->gate_weight ||
        !into->gate_bias)
        oops(r, "cannot load %s's dense weights: %s", prefix, why);
}

static void free_attention(attention *which) {
    free_projection(&which->query); free_projection(&which->key);
    free_projection(&which->value); free_projection(&which->output);
    h3_gpu_tensor_free(which->query_norm); h3_gpu_tensor_free(which->key_norm);
    h3_gpu_tensor_free(which->gate_weight); h3_gpu_tensor_free(which->gate_bias);
    which->query_norm = which->key_norm = NULL;
    which->gate_weight = which->gate_bias = NULL;
}

/* --------------------------------------------------------------- modulation */

/* The block's static table added to every level of the module's output. The
 * table is per block and shared by all levels; the timestep part is what
 * varies, so the sum is [level][slot][width] -- the layout the row map
 * indexes. Built on the host and uploaded once per block.
 *
 * `timestep` is the whole [level][table_slots][width] output of the module,
 * from which `used_slots` are taken starting at `first_slot`; the two
 * cross-modal consumers split one five-row table between them that way. */
static h3_gpu_tensor *modulation_of(run *r, const char *block,
                                    const char *table_name,
                                    const float *timestep, uint32_t width,
                                    uint32_t table_slots, uint32_t first_slot,
                                    uint32_t used_slots, uint32_t module_slots,
                                    uint32_t levels) {
    char name[256];
    snprintf(name, sizeof(name), "%s%s", block, table_name);
    float *table = read_tensor(r, name, (size_t)table_slots * width);
    if (!table) return NULL;
    float *sum = malloc((size_t)levels * used_slots * width * sizeof(*sum));
    if (!sum) { oops(r, "cannot allocate a modulation"); free(table); return NULL; }
    for (uint32_t level = 0; level < levels; level++)
        for (uint32_t slot = 0; slot < used_slots; slot++)
            for (uint32_t index = 0; index < width; index++)
                sum[((size_t)level * used_slots + slot) * width + index] =
                    table[(size_t)(first_slot + slot) * width + index] +
                    timestep[((size_t)level * module_slots + slot) * width +
                             index];
    h3_gpu_tensor *tensor = upload(r, sum, (size_t)levels * used_slots * width);
    free(table);
    free(sum);
    return tensor;
}

/* The context modulation: the one affine with no norm in front of it, so the
 * AdaLN kernel cannot serve it. The context is a few hundred rows and this
 * runs once per block, which is nothing beside the block itself. */
static h3_gpu_tensor *modulate_context(run *r, const char *block,
                                       const char *table_name,
                                       const float *context,
                                       const float *timestep, uint32_t rows,
                                       uint32_t width) {
    char name[256];
    snprintf(name, sizeof(name), "%s%s", block, table_name);
    float *table = read_tensor(r, name, (size_t)PROMPT_SLOTS * width);
    if (!table) return NULL;
    float *out = malloc((size_t)rows * width * sizeof(*out));
    if (!out) {
        oops(r, "cannot allocate a modulated context");
        free(table);
        return NULL;
    }
    for (uint32_t row = 0; row < rows; row++)
        for (uint32_t index = 0; index < width; index++) {
            const float shift = table[index] + timestep[index];
            const float scale = table[width + index] + timestep[width + index];
            out[(size_t)row * width + index] =
                context[(size_t)row * width + index] * (1.0f + scale) + shift;
        }
    h3_gpu_tensor *tensor = upload(r, out, (size_t)rows * width);
    free(table);
    free(out);
    return tensor;
}

/* ------------------------------------------------------------------- block */

typedef struct {
    h3_gpu_tensor *query, *key, *value, *heads, *logits;
    h3_gpu_tensor *rotated_query, *rotated_kv, *inner;
    h3_gpu_tensor *ones_video, *ones_audio, *row_map_video, *row_map_audio;
} scratch;

static void run_attention(run *r, const attention *a, h3_gpu_tensor *out,
                          const h3_gpu_tensor *query_input, uint32_t query_rows,
                          const h3_gpu_tensor *kv_input, uint32_t kv_rows,
                          const h3_gpu_tensor *query_cos,
                          const h3_gpu_tensor *query_sin,
                          const h3_gpu_tensor *key_cos,
                          const h3_gpu_tensor *key_sin, scratch *space) {
    const uint32_t head_dim = a->inner / HEADS;
    GPU_OP(r, h3_gpu_linear_bf16(r->gpu, space->logits, query_input,
                                 a->gate_weight, a->gate_bias, query_rows,
                                 a->gate_in, HEADS), "gate logits");
    GPU_OP(r, h3_gpu_convrot_bf16(r->gpu, space->rotated_query, query_input,
                                  query_rows, a->query_in, CONVROT_GROUP),
           "query ConvRot");
    const h3_gpu_tensor *rotated_kv = space->rotated_query;
    if (kv_input != query_input) {
        GPU_OP(r, h3_gpu_convrot_bf16(r->gpu, space->rotated_kv, kv_input,
                                      kv_rows, a->kv_in, CONVROT_GROUP),
               "context ConvRot");
        rotated_kv = space->rotated_kv;
    }
    GPU_OP(r, h3_gpu_linear_i8_weight_bf16_square_output_major(
                  r->gpu, space->query, space->rotated_query, a->query.weight,
                  a->query.scales, a->query.bias, query_rows, a->query_in,
                  a->inner), "query projection");
    GPU_OP(r, h3_gpu_linear_i8_weight_bf16_square_output_major(
                  r->gpu, space->key, rotated_kv, a->key.weight, a->key.scales,
                  a->key.bias, kv_rows, a->kv_in, a->inner), "key projection");
    GPU_OP(r, h3_gpu_linear_i8_weight_bf16_square_output_major(
                  r->gpu, space->value, rotated_kv, a->value.weight,
                  a->value.scales, a->value.bias, kv_rows, a->kv_in, a->inner),
           "value projection");
    GPU_OP(r, h3_gpu_rms_norm_bf16(r->gpu, space->query, space->query,
                                   a->query_norm, query_rows, a->inner,
                                   BLOCK_EPSILON), "query norm");
    GPU_OP(r, h3_gpu_rms_norm_bf16(r->gpu, space->key, space->key, a->key_norm,
                                   kv_rows, a->inner, BLOCK_EPSILON),
           "key norm");
    if (query_cos)
        GPU_OP(r, h3_gpu_rope_rows_bf16(r->gpu, space->query, query_cos,
                                        query_sin, query_rows, HEADS, head_dim,
                                        query_rows * (head_dim / 2)),
               "query rope");
    if (key_cos)
        GPU_OP(r, h3_gpu_rope_rows_bf16(r->gpu, space->key, key_cos, key_sin,
                                        kv_rows, HEADS, head_dim,
                                        kv_rows * (head_dim / 2)), "key rope");
    GPU_OP(r, h3_gpu_sdpa_cross_bf16(r->gpu, space->heads, space->query,
                                     space->key, space->value, query_rows,
                                     kv_rows, HEADS, head_dim,
                                     1.0f / sqrtf((float)head_dim)),
           "attention");
    GPU_OP(r, h3_gpu_head_gate_bf16(r->gpu, space->heads, space->logits,
                                    query_rows, HEADS, head_dim), "head gate");
    GPU_OP(r, h3_gpu_convrot_bf16(r->gpu, space->heads, space->heads,
                                  query_rows, a->inner, CONVROT_GROUP),
           "output ConvRot");
    GPU_OP(r, h3_gpu_linear_i8_weight_bf16_square_output_major(
                  r->gpu, out, space->heads, a->output.weight,
                  a->output.scales, a->output.bias, query_rows, a->inner,
                  a->out_dim), "output projection");
}

typedef struct {
    attention attn1, attn2, audio_attn1, audio_attn2, a2v, v2a;
    projection ff_in, ff_out, audio_ff_in, audio_ff_out;
    h3_gpu_tensor *video_modulation, *audio_modulation;
    h3_gpu_tensor *video_cross, *audio_cross, *video_gate, *audio_gate;
    h3_gpu_tensor *video_context, *audio_context;
} block_weights;

typedef struct {
    const float *video_timestep, *audio_timestep;
    const float *video_prompt, *audio_prompt;
    const float *video_cross, *audio_cross, *video_gate, *audio_gate;
    const float *video_embedded, *audio_embedded;
    const float *video_context, *audio_context;
} conditioning;

static void free_block(block_weights *weights) {
    free_attention(&weights->attn1); free_attention(&weights->attn2);
    free_attention(&weights->audio_attn1); free_attention(&weights->audio_attn2);
    free_attention(&weights->a2v); free_attention(&weights->v2a);
    free_projection(&weights->ff_in); free_projection(&weights->ff_out);
    free_projection(&weights->audio_ff_in); free_projection(&weights->audio_ff_out);
    h3_gpu_tensor_free(weights->video_modulation);
    h3_gpu_tensor_free(weights->audio_modulation);
    h3_gpu_tensor_free(weights->video_cross);
    h3_gpu_tensor_free(weights->audio_cross);
    h3_gpu_tensor_free(weights->video_gate);
    h3_gpu_tensor_free(weights->audio_gate);
    h3_gpu_tensor_free(weights->video_context);
    h3_gpu_tensor_free(weights->audio_context);
    memset(weights, 0, sizeof(*weights));
}

static void load_block(run *r, int index, const conditioning *from,
                       block_weights *weights) {
    char block[192];
    snprintf(block, sizeof(block),
             "model.diffusion_model.transformer_blocks.%d.", index);
    memset(weights, 0, sizeof(*weights));
    weights->attn1 = (attention){"attn1", VIDEO_DIM, VIDEO_DIM, VIDEO_DIM,
                                 VIDEO_DIM, VIDEO_DIM, {0}, {0}, {0}, {0},
                                 NULL, NULL, NULL, NULL};
    weights->attn2 = (attention){"attn2", VIDEO_DIM, VIDEO_DIM, VIDEO_DIM,
                                 VIDEO_DIM, VIDEO_DIM, {0}, {0}, {0}, {0},
                                 NULL, NULL, NULL, NULL};
    weights->audio_attn1 = (attention){"audio_attn1", AUDIO_DIM, AUDIO_DIM,
                                       AUDIO_DIM, AUDIO_DIM, AUDIO_DIM, {0},
                                       {0}, {0}, {0}, NULL, NULL, NULL, NULL};
    weights->audio_attn2 = (attention){"audio_attn2", AUDIO_DIM, AUDIO_DIM,
                                       AUDIO_DIM, AUDIO_DIM, AUDIO_DIM, {0},
                                       {0}, {0}, {0}, NULL, NULL, NULL, NULL};
    weights->a2v = (attention){"audio_to_video_attn", VIDEO_DIM, AUDIO_DIM,
                               AUDIO_DIM, VIDEO_DIM, VIDEO_DIM, {0}, {0}, {0},
                               {0}, NULL, NULL, NULL, NULL};
    weights->v2a = (attention){"video_to_audio_attn", AUDIO_DIM, VIDEO_DIM,
                               AUDIO_DIM, AUDIO_DIM, AUDIO_DIM, {0}, {0}, {0},
                               {0}, NULL, NULL, NULL, NULL};
    load_attention(r, block, &weights->attn1);
    load_attention(r, block, &weights->attn2);
    load_attention(r, block, &weights->audio_attn1);
    load_attention(r, block, &weights->audio_attn2);
    load_attention(r, block, &weights->a2v);
    load_attention(r, block, &weights->v2a);
    load_projection(r, block, "ff.net.0.proj", VIDEO_FF, VIDEO_DIM, 0,
                    &weights->ff_in);
    load_projection(r, block, "ff.net.2", VIDEO_DIM, VIDEO_FF, 0,
                    &weights->ff_out);
    load_projection(r, block, "audio_ff.net.0.proj", AUDIO_FF, AUDIO_DIM, 1,
                    &weights->audio_ff_in);
    load_projection(r, block, "audio_ff.net.2", AUDIO_DIM, AUDIO_FF, 1,
                    &weights->audio_ff_out);

    weights->video_modulation = modulation_of(
        r, block, "scale_shift_table", from->video_timestep, VIDEO_DIM,
        ADA_SLOTS, 0, ADA_SLOTS, ADA_SLOTS, LEVELS);
    weights->audio_modulation = modulation_of(
        r, block, "audio_scale_shift_table", from->audio_timestep, AUDIO_DIM,
        ADA_SLOTS, 0, ADA_SLOTS, ADA_SLOTS, LEVELS);
    weights->video_cross = modulation_of(
        r, block, "scale_shift_table_a2v_ca_video", from->video_cross,
        VIDEO_DIM, CROSS_TABLE_SLOTS, 0, CROSS_SLOTS, CROSS_SLOTS, LEVELS);
    weights->audio_cross = modulation_of(
        r, block, "scale_shift_table_a2v_ca_audio", from->audio_cross,
        AUDIO_DIM, CROSS_TABLE_SLOTS, 0, CROSS_SLOTS, CROSS_SLOTS, LEVELS);
    /* The gates take the fifth row of the same five-row table the scale and
     * shift above take the first four from. */
    weights->video_gate = modulation_of(
        r, block, "scale_shift_table_a2v_ca_video", from->video_gate, VIDEO_DIM,
        CROSS_TABLE_SLOTS, CROSS_SLOTS, GATE_SLOTS, GATE_SLOTS, LEVELS);
    weights->audio_gate = modulation_of(
        r, block, "scale_shift_table_a2v_ca_audio", from->audio_gate, AUDIO_DIM,
        CROSS_TABLE_SLOTS, CROSS_SLOTS, GATE_SLOTS, GATE_SLOTS, LEVELS);
    weights->video_context = modulate_context(
        r, block, "prompt_scale_shift_table", from->video_context,
        from->video_prompt, r->text_rows, VIDEO_DIM);
    weights->audio_context = modulate_context(
        r, block, "audio_prompt_scale_shift_table", from->audio_context,
        from->audio_prompt, r->text_rows, AUDIO_DIM);
}

typedef struct { h3_gpu_tensor *cos, *sin; } rope;

typedef struct {
    h3_gpu_tensor *video, *audio, *video_pre, *audio_pre;
    h3_gpu_tensor *video_scaled, *audio_scaled;
    h3_gpu_tensor *video_branch, *audio_branch;
    rope video_pe, audio_pe, video_cross_pe, audio_cross_pe;
} state;

static void self_and_text(run *r, const attention *self, const attention *text,
                          h3_gpu_tensor *x, h3_gpu_tensor *scaled,
                          h3_gpu_tensor *branch,
                          const h3_gpu_tensor *modulation,
                          const h3_gpu_tensor *context,
                          const h3_gpu_tensor *ones,
                          const h3_gpu_tensor *row_map, const rope *pe,
                          uint32_t rows, uint32_t dim, scratch *space) {
    GPU_OP(r, h3_gpu_adaln_bf16(r->gpu, scaled, x, ones, modulation, row_map,
                                rows, dim, ADA_SLOTS, 0, 1, BLOCK_EPSILON),
           "self-attention modulation");
    run_attention(r, self, branch, scaled, rows, scaled, rows,
                  pe->cos, pe->sin, pe->cos, pe->sin, space);
    GPU_OP(r, h3_gpu_gate_bf16(r->gpu, x, x, branch, modulation, row_map, rows,
                               dim, ADA_SLOTS, 2), "self-attention residual");
    GPU_OP(r, h3_gpu_adaln_bf16(r->gpu, scaled, x, ones, modulation, row_map,
                                rows, dim, ADA_SLOTS, 6, 7, BLOCK_EPSILON),
           "text cross-attention modulation");
    /* No rope on the text cross-attention: the prompt has no position in the
     * clip's frame of reference. */
    run_attention(r, text, branch, scaled, rows, context, r->text_rows,
                  NULL, NULL, NULL, NULL, space);
    GPU_OP(r, h3_gpu_gate_bf16(r->gpu, x, x, branch, modulation, row_map, rows,
                               dim, ADA_SLOTS, 8), "text cross-attention residual");
}

static void feed_forward(run *r, const projection *in, const projection *out,
                         h3_gpu_tensor *x, h3_gpu_tensor *scaled,
                         h3_gpu_tensor *branch,
                         const h3_gpu_tensor *modulation,
                         const h3_gpu_tensor *ones,
                         const h3_gpu_tensor *row_map, uint32_t rows,
                         uint32_t dim, uint32_t width, scratch *space) {
    GPU_OP(r, h3_gpu_adaln_bf16(r->gpu, scaled, x, ones, modulation, row_map,
                                rows, dim, ADA_SLOTS, 3, 4, BLOCK_EPSILON),
           "feed-forward modulation");
    GPU_OP(r, h3_gpu_convrot_bf16(r->gpu, scaled, scaled, rows, dim,
                                  CONVROT_GROUP), "feed-forward ConvRot");
    GPU_OP(r, h3_gpu_linear_i8_weight_bf16_square_output_major(
                  r->gpu, space->inner, scaled, in->weight, in->scales,
                  in->bias, rows, dim, width), "feed-forward in");
    GPU_OP(r, h3_gpu_gelu_bf16(r->gpu, space->inner, space->inner, rows * width,
                               1), "feed-forward activation");
    GPU_OP(r, h3_gpu_convrot_bf16(r->gpu, space->inner, space->inner, rows,
                                  width, CONVROT_GROUP),
           "feed-forward out ConvRot");
    GPU_OP(r, h3_gpu_linear_i8_weight_bf16_square_output_major(
                  r->gpu, branch, space->inner, out->weight, out->scales,
                  out->bias, rows, width, dim), "feed-forward out");
    GPU_OP(r, h3_gpu_gate_bf16(r->gpu, x, x, branch, modulation, row_map, rows,
                               dim, ADA_SLOTS, 5), "feed-forward residual");
}

static void run_block(run *r, const block_weights *weight, state *s,
                      scratch *space) {
    const uint32_t video_rows = r->video_rows, audio_rows = r->audio_rows;
    self_and_text(r, &weight->attn1, &weight->attn2, s->video, s->video_scaled,
                  s->video_branch, weight->video_modulation,
                  weight->video_context, space->ones_video,
                  space->row_map_video, &s->video_pe, video_rows, VIDEO_DIM,
                  space);
    self_and_text(r, &weight->audio_attn1, &weight->audio_attn2, s->audio,
                  s->audio_scaled, s->audio_branch, weight->audio_modulation,
                  weight->audio_context, space->ones_audio,
                  space->row_map_audio, &s->audio_pe, audio_rows, AUDIO_DIM,
                  space);
    /* Snapshot both streams: video-into-audio must read the video stream as it
     * was before audio-into-video changed it. */
    GPU_OP(r, h3_gpu_copy_bf16(r->gpu, s->video_pre, 0, s->video, 0,
                               (size_t)video_rows * VIDEO_DIM),
           "video snapshot");
    GPU_OP(r, h3_gpu_copy_bf16(r->gpu, s->audio_pre, 0, s->audio, 0,
                               (size_t)audio_rows * AUDIO_DIM),
           "audio snapshot");
    /* Scale then shift, the opposite order to the nine-row table's. */
    GPU_OP(r, h3_gpu_adaln_bf16(r->gpu, s->video_scaled, s->video_pre,
                                space->ones_video, weight->video_cross,
                                space->row_map_video, video_rows, VIDEO_DIM,
                                CROSS_SLOTS, 1, 0, BLOCK_EPSILON),
           "a2v video modulation");
    GPU_OP(r, h3_gpu_adaln_bf16(r->gpu, s->audio_scaled, s->audio_pre,
                                space->ones_audio, weight->audio_cross,
                                space->row_map_audio, audio_rows, AUDIO_DIM,
                                CROSS_SLOTS, 1, 0, BLOCK_EPSILON),
           "a2v audio modulation");
    run_attention(r, &weight->a2v, s->video_branch, s->video_scaled, video_rows,
                  s->audio_scaled, audio_rows, s->video_cross_pe.cos,
                  s->video_cross_pe.sin, s->audio_cross_pe.cos,
                  s->audio_cross_pe.sin, space);
    GPU_OP(r, h3_gpu_gate_bf16(r->gpu, s->video, s->video, s->video_branch,
                               weight->video_gate, space->row_map_video,
                               video_rows, VIDEO_DIM, GATE_SLOTS, 0),
           "a2v residual");
    GPU_OP(r, h3_gpu_adaln_bf16(r->gpu, s->audio_scaled, s->audio_pre,
                                space->ones_audio, weight->audio_cross,
                                space->row_map_audio, audio_rows, AUDIO_DIM,
                                CROSS_SLOTS, 3, 2, BLOCK_EPSILON),
           "v2a audio modulation");
    GPU_OP(r, h3_gpu_adaln_bf16(r->gpu, s->video_scaled, s->video_pre,
                                space->ones_video, weight->video_cross,
                                space->row_map_video, video_rows, VIDEO_DIM,
                                CROSS_SLOTS, 3, 2, BLOCK_EPSILON),
           "v2a video modulation");
    run_attention(r, &weight->v2a, s->audio_branch, s->audio_scaled, audio_rows,
                  s->video_scaled, video_rows, s->audio_cross_pe.cos,
                  s->audio_cross_pe.sin, s->video_cross_pe.cos,
                  s->video_cross_pe.sin, space);
    GPU_OP(r, h3_gpu_gate_bf16(r->gpu, s->audio, s->audio, s->audio_branch,
                               weight->audio_gate, space->row_map_audio,
                               audio_rows, AUDIO_DIM, GATE_SLOTS, 0),
           "v2a residual");
    feed_forward(r, &weight->ff_in, &weight->ff_out, s->video, s->video_scaled,
                 s->video_branch, weight->video_modulation, space->ones_video,
                 space->row_map_video, video_rows, VIDEO_DIM, VIDEO_FF, space);
    feed_forward(r, &weight->audio_ff_in, &weight->audio_ff_out, s->audio,
                 s->audio_scaled, s->audio_branch, weight->audio_modulation,
                 space->ones_audio, space->row_map_audio, audio_rows,
                 AUDIO_DIM, AUDIO_FF, space);
}

/* -------------------------------------------------------------- the head */

/* LayerNorm -- the one norm here that subtracts a mean -- then the two-row
 * table plus the *same* embedded timestep in both rows, then the projection
 * back down to the latent width.
 *
 * The embedded timestep is per token here too: `_process_output` indexes it
 * `embedded_timestep[:, :, None]`, so a conditioning token's head modulation
 * is the one built at timestep zero, not the one the rest of the frame uses. */
static void output_head(run *r, const float *x, const float *table,
                        const float *embedded, const uint32_t *level_of,
                        const float *weight, const float *bias, float *out,
                        uint32_t rows, uint32_t dim) {
    if (r->failed) return;
    float *scratch_row = malloc(dim * sizeof(*scratch_row));
    if (!scratch_row) { oops(r, "cannot allocate a head row"); return; }
    for (uint32_t row = 0; row < rows; row++) {
        const float *source = x + (size_t)row * dim;
        const float *level = embedded + (size_t)level_of[row] * dim;
        double mean = 0.0;
        for (uint32_t index = 0; index < dim; index++) mean += source[index];
        mean /= (double)dim;
        double variance = 0.0;
        for (uint32_t index = 0; index < dim; index++) {
            const double centred = (double)source[index] - mean;
            variance += centred * centred;
        }
        const double inverse = 1.0 / sqrt(variance / (double)dim + 1e-6);
        for (uint32_t index = 0; index < dim; index++) {
            const float shift = table[index] + level[index];
            const float scale = table[dim + index] + level[index];
            scratch_row[index] =
                (float)(((double)source[index] - mean) * inverse) *
                (1.0f + scale) + shift;
        }
        linear_rows(out + (size_t)row * LATENT, scratch_row, weight, bias, 1,
                    dim, LATENT);
    }
    free(scratch_row);
}

/* ----------------------------------------------------------- the sampler */

/* Velocity to denoised prediction, then the Euler step, written the long way
 * round: `(x - (x - v * sigma)) / sigma` is `v` only in exact arithmetic, and
 * this has to match the driver bit for bit.
 *
 * The step runs on the *scalar* schedule sigma rather than a per-token
 * timestep. The two are not interchangeable -- sigma here is never zero, while
 * a held token's timestep always is -- but nothing is held in this path, so
 * every token shares it. */
static void advance(float *latent, const float *velocity, double sigma,
                    double next, size_t count) {
    for (size_t index = 0; index < count; index++) {
        const double denoised = (double)latent[index] -
                                (double)velocity[index] * sigma;
        latent[index] = (float)((double)latent[index] +
            (((double)latent[index] - denoised) / sigma) * (next - sigma));
    }
}

/* The schedule, token-dependent and stretched to the terminal value -- pinned
 * against LTX's own in `test_ltx_sampler.c`. */
static void schedule(float *out, int steps, double tokens) {
    const double slope = (2.05 - 0.95) / (4096.0 - 1024.0);
    const double intercept = 0.95 - slope * 1024.0;
    const double shift = exp(tokens * slope + intercept);
    for (int index = 0; index <= steps; index++) {
        const double linear = 1.0 - (double)index / (double)steps;
        out[index] = linear != 0.0
            ? (float)(shift / (shift + (1.0 / linear - 1.0))) : 0.0f;
    }
    int last = -1;
    for (int index = 0; index <= steps; index++)
        if (out[index] != 0.0f) last = index;
    if (last < 0) return;
    const double scale = (1.0 - (double)out[last]) / (1.0 - 0.1);
    for (int index = 0; index <= steps; index++)
        if (out[index] != 0.0f)
            out[index] = (float)(1.0 - (1.0 - (double)out[index]) / scale);
}

/* ---------------------------------------------------------------- noise */

/* xorshift plus a Box-Muller pair: deterministic from a seed, and self
 * contained so a generation reproduces without depending on libc's RNG. */
typedef struct { uint64_t state; } rng;

static double rng_uniform(rng *r) {
    r->state ^= r->state << 13;
    r->state ^= r->state >> 7;
    r->state ^= r->state << 17;
    /* Open at zero: the log below cannot take it. */
    return ((double)(r->state >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

static void gaussian_fill(rng *r, float *out, size_t count) {
    for (size_t index = 0; index < count; index += 2) {
        const double radius = sqrt(-2.0 * log(rng_uniform(r)));
        const double angle = 2.0 * M_PI * rng_uniform(r);
        out[index] = (float)(radius * cos(angle));
        if (index + 1 < count) out[index + 1] = (float)(radius * sin(angle));
    }
}

/* -------------------------------------------------------------- prefetch */

/* The weights for one block, loaded off the critical path, the way `h3_dit.c`
 * streams H3's. On by default; `H3_LTX_PREFETCH=0` disables it, which is how
 * the A/B below was run.
 *
 * Measured at 512x512 over 4 steps, alternating so both modes see the same
 * machine:
 *
 *     serial    71.7 s blocks + 43.3 s loading = 126.1 s
 *     prefetch  45.8 s blocks +  0.4 s loading =  56.5 s
 *     serial    45.0 s blocks + 28.1 s loading =  82.4 s
 *     prefetch  44.9 s blocks +  0.5 s loading =  55.7 s
 *
 * Block time is unchanged between modes, so the worker costs the GPU nothing;
 * loading essentially disappears. A block is 388 MB of int8 and the file
 * layout scatters it -- the checkpoint is sorted by tensor name, so a block's
 * hundred tensors interleave with every other block's and span the whole
 * 21.5 GB at 1.8% density. There is no sequential read to be had without
 * rewriting the file.
 *
 * A first attempt at this measurement said the opposite, confidently, with a
 * tidy explanation about unified memory. Another generation was running on the
 * machine at the time. **One timing run on a shared machine is not a
 * measurement.** Alternate the configurations.
 *
 * `load_block` only reads the checkpoint and uploads; it never touches the
 * command buffer, so it is safe beside a submit. It gets its own `run` rather
 * than sharing the caller's, because the shared one is being read by the main
 * thread's GPU_OPs at the same time -- and because `h3_gpu_error` is a single
 * slot that two threads must not both be reaching for. */
typedef struct {
    int index;
    const conditioning *cond;
    block_weights weights;
    run r;
    char error[512];
} preload_job;

static void *preload_thread(void *opaque) {
    preload_job *job = opaque;
    load_block(&job->r, job->index, job->cond, &job->weights);
    return NULL;
}

static void start_preload(run *r, preload_job *job, int index,
                          const conditioning *cond) {
    job->index = index;
    job->cond = cond;
    job->error[0] = '\0';
    job->r = *r;
    job->r.error = job->error;
    job->r.error_size = sizeof(job->error);
    job->r.failed = 0;
    job->r.cancelled = 0;
}

static void adopt_preload(run *r, const preload_job *job) {
    if (!job->r.failed || r->failed) return;
    oops(r, "%s", job->error[0] ? job->error :
                  "a prefetched block failed without saying why");
}

/* -------------------------------------------------------- the position grid */

/* The rope reads **pixel** coordinates, not latent ones. `get_pixel_coords`
 * scales each axis by the VAE's compression before `precompute_freqs_cis` ever
 * sees it, and `causal_temporal_positioning` then shifts the time axis so the
 * first latent frame -- which a causal encoder builds from a single pixel
 * frame -- spans [0, 1) rather than [0, 8).
 *
 * This is the seam no component test could reach: the block, top-level and
 * end-to-end anchors all fed the reference the *same* grid the engine used, so
 * latent coordinates agreed with themselves. */
static void video_positions(const run *r, float *grid) {
    static const int SCALE[VIDEO_AXES] = {8, 32, 32};
    const uint32_t rows = r->video_rows;
    const uint32_t plane = r->height * r->width;
    for (uint32_t token = 0; token < rows; token++) {
        const int frame = (int)(token / plane);
        const uint32_t rest = token % plane;
        const int starts[VIDEO_AXES] = {frame, (int)(rest / r->width),
                                        (int)(rest % r->width)};
        for (int axis = 0; axis < VIDEO_AXES; axis++) {
            const size_t at = ((size_t)axis * rows + token) * 2;
            int begin = starts[axis] * SCALE[axis];
            int end = (starts[axis] + 1) * SCALE[axis];
            if (axis == 0) {
                begin = begin + 1 - SCALE[0];
                end = end + 1 - SCALE[0];
                if (begin < 0) begin = 0;
                if (end < 0) end = 0;
            }
            /* And the time axis is in **seconds**, not frames:
             * `create_initial_state` divides axis 0 by the fps right after
             * `get_pixel_coords`. Leaving it in frames scales the whole axis
             * by the frame rate and puts the video on a different footing from
             * the audio, whose bounds are already seconds. The two streams
             * attend to each other through these positions, so they have to
             * share units. */
            grid[at] = axis == 0 ? (float)begin / (float)r->fps : (float)begin;
            grid[at + 1] = axis == 0 ? (float)end / (float)r->fps : (float)end;
        }
    }
}

/* The audio patchifier's bounds are **timestamps in seconds**, not indices:
 * `_get_audio_latent_time_in_sec` multiplies the latent frame by the 4x
 * downsample, applies the same causal offset, and scales by hop over sample
 * rate. */
static void audio_positions(const run *r, float *grid) {
    for (uint32_t token = 0; token < r->audio_rows; token++) {
        int begin = (int)token * AUDIO_DOWNSAMPLE + 1 - AUDIO_DOWNSAMPLE;
        int end = (int)(token + 1) * AUDIO_DOWNSAMPLE + 1 - AUDIO_DOWNSAMPLE;
        if (begin < 0) begin = 0;
        if (end < 0) end = 0;
        grid[(size_t)token * 2] = (float)begin * AUDIO_SECONDS;
        grid[(size_t)token * 2 + 1] = (float)end * AUDIO_SECONDS;
    }
}

/* ------------------------------------------------------------------- entry */

int ltx_dit_sample(h3_gpu *gpu, const h3_weight_store *dit,
                   const ltx_dit_request *request, const float *video_context,
                   const float *audio_context, uint32_t span,
                   float *video_latent, float *audio_latent,
                   ltx_dit_tick tick, void *tick_context,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!gpu || !dit || !request || !video_context || !audio_context ||
        !video_latent || !audio_latent) {
        if (error && error_size)
            snprintf(error, error_size, "ltx_dit_sample wants a gpu, the DiT "
                                        "store, a request, both context "
                                        "streams and both latents");
        return 0;
    }
    if (!request->frames || !request->height || !request->width ||
        !request->audio_rows || request->fps <= 0) {
        if (error && error_size)
            snprintf(error, error_size, "the geometry %ux%ux%u at %d fps with "
                                        "%u audio rows is not renderable",
                     request->frames, request->height, request->width,
                     request->fps, request->audio_rows);
        return 0;
    }
    if (!span || span > MAX_CONTEXT) {
        if (error && error_size)
            snprintf(error, error_size, "the context has %u rows, outside "
                                        "1..%d", span, MAX_CONTEXT);
        return 0;
    }
    /* One step is NaN upstream: the schedule divides by it. */
    if (request->steps < 2 || request->steps > MAX_STEPS) {
        if (error && error_size)
            snprintf(error, error_size, "step count %d is outside 2..%d",
                     request->steps, MAX_STEPS);
        return 0;
    }

    run r = {0};
    r.gpu = gpu;
    r.store = dit;
    r.frames = request->frames;
    r.height = request->height;
    r.width = request->width;
    r.video_rows = request->frames * request->height * request->width;
    r.audio_rows = request->audio_rows;
    r.text_rows = span;
    r.fps = request->fps;
    r.error = error;
    r.error_size = error_size;

    const int steps = request->steps;
    const uint32_t video_rows = r.video_rows, audio_rows = r.audio_rows;
    const size_t video_state = (size_t)video_rows * VIDEO_DIM;
    const size_t audio_state = (size_t)audio_rows * AUDIO_DIM;
    /* The text cross-attention keys and values are `span` rows wide, and a
     * short clip can have fewer video tokens than the prompt has -- 2x8x8 is
     * 128 against a 256-token context. The driver sized this on the video
     * alone, which was safe only because every geometry it ever ran had more
     * video tokens than context. */
    const size_t widest = (size_t)(video_rows > span ? video_rows : span) *
                          VIDEO_DIM;

    float sigmas[MAX_STEPS + 1];
    schedule(sigmas, steps, (double)video_rows);

    rng noise = {request->seed};
    if (!noise.state) noise.state = 1;
    /* Flow matching starts at pure noise, which is what sigma 1 means. */
    gaussian_fill(&noise, video_latent, (size_t)video_rows * LATENT);
    gaussian_fill(&noise, audio_latent, (size_t)audio_rows * LATENT);

    /* ------------------------------------ every module at every timestep */

    /* Both streams denoise, so every token shares one modulation row and the
     * two levels hold the same value; the row map stays at zero. */
    float *modulation[MODULE_COUNT] = {0}, *embedded[MODULE_COUNT] = {0};
    for (int index = 0; index < MODULE_COUNT; index++) {
        double values[MAX_STEPS * LEVELS];
        for (int step = 0; step < steps; step++)
            for (int level = 0; level < LEVELS; level++)
                values[step * LEVELS + level] = sigmas[step];
        run_adaln(&r, &MODULES[index], values, (uint32_t)(steps * LEVELS),
                  TIMESTEP_MULTIPLIER, &modulation[index], &embedded[index]);
    }

    /* ------------------------------------------------- weights held across */

    float *video_patch_weight = weight_of(&r, "patchify_proj.weight",
                                          (size_t)VIDEO_DIM * LATENT);
    float *video_patch_bias = weight_of(&r, "patchify_proj.bias", VIDEO_DIM);
    float *audio_patch_weight = weight_of(&r, "audio_patchify_proj.weight",
                                          (size_t)AUDIO_DIM * LATENT);
    float *audio_patch_bias = weight_of(&r, "audio_patchify_proj.bias",
                                        AUDIO_DIM);
    float *video_head_table = weight_of(&r, "scale_shift_table",
                                        (size_t)HEAD_SLOTS * VIDEO_DIM);
    float *audio_head_table = weight_of(&r, "audio_scale_shift_table",
                                        (size_t)HEAD_SLOTS * AUDIO_DIM);
    float *video_head_weight = weight_of(&r, "proj_out.weight",
                                         (size_t)LATENT * VIDEO_DIM);
    float *video_head_bias = weight_of(&r, "proj_out.bias", LATENT);
    float *audio_head_weight = weight_of(&r, "audio_proj_out.weight",
                                         (size_t)LATENT * AUDIO_DIM);
    float *audio_head_bias = weight_of(&r, "audio_proj_out.bias", LATENT);
    /* Keyframes mark latents covering a single pixel frame. A causal encoder
     * makes the first one such, so that is the marker's target here. */
    float *keyframe_marker = weight_of(&r, "keyframes_abs_pos_embedding",
                                       VIDEO_DIM);

    /* ------------------------------------------------------------- rope */

    state s;
    memset(&s, 0, sizeof(s));
    if (!r.failed) {
        float *video_grid = malloc((size_t)VIDEO_AXES * video_rows * 2 *
                                   sizeof(*video_grid));
        float *audio_grid = malloc((size_t)audio_rows * 2 *
                                   sizeof(*audio_grid));
        if (!video_grid || !audio_grid)
            oops(&r, "cannot allocate the position grids");
        else {
            video_positions(&r, video_grid);
            audio_positions(&r, audio_grid);
            rope_tables(&r, &s.video_pe.cos, &s.video_pe.sin, video_grid,
                        video_rows, VIDEO_DIM, VIDEO_AXES, VIDEO_MAX_POS);
            rope_tables(&r, &s.audio_pe.cos, &s.audio_pe.sin, audio_grid,
                        audio_rows, AUDIO_DIM, AUDIO_AXES, AUDIO_MAX_POS);
            /* The cross-modal attentions run at the audio width in both
             * directions, so each stream needs a second table at that width. */
            rope_tables(&r, &s.video_cross_pe.cos, &s.video_cross_pe.sin,
                        video_grid, video_rows, AUDIO_DIM, VIDEO_AXES,
                        VIDEO_MAX_POS);
            rope_tables(&r, &s.audio_cross_pe.cos, &s.audio_cross_pe.sin,
                        audio_grid, audio_rows, AUDIO_DIM, AUDIO_AXES,
                        AUDIO_MAX_POS);
        }
        free(video_grid);
        free(audio_grid);
    }

    /* ----------------------------------------------------------- scratch */

    scratch space;
    memset(&space, 0, sizeof(space));
    if (!r.failed) {
        space.query = h3_gpu_tensor_new_bf16(gpu, widest);
        space.key = h3_gpu_tensor_new_bf16(gpu, widest);
        space.value = h3_gpu_tensor_new_bf16(gpu, widest);
        space.heads = h3_gpu_tensor_new_bf16(gpu, widest);
        space.logits = h3_gpu_tensor_new_bf16(gpu, (size_t)video_rows * HEADS);
        space.rotated_query = h3_gpu_tensor_new_bf16(gpu, widest);
        space.rotated_kv = h3_gpu_tensor_new_bf16(gpu, widest);
        space.inner = h3_gpu_tensor_new_bf16(gpu,
                                             (size_t)video_rows * VIDEO_FF);
        if (!space.query || !space.key || !space.value || !space.heads ||
            !space.logits || !space.rotated_query || !space.rotated_kv ||
            !space.inner)
            oops(&r, "cannot allocate attention scratch");
    }
    if (!r.failed) {
        uint16_t *ones = malloc(VIDEO_DIM * sizeof(*ones));
        if (!ones) oops(&r, "cannot allocate a norm weight");
        else {
            for (int index = 0; index < VIDEO_DIM; index++) ones[index] = 0x3f80;
            space.ones_video = h3_gpu_tensor_from_bf16(gpu, ones, VIDEO_DIM);
            space.ones_audio = h3_gpu_tensor_from_bf16(gpu, ones, AUDIO_DIM);
            free(ones);
            if (!space.ones_video || !space.ones_audio)
                oops(&r, "cannot upload a norm weight");
        }
    }
    /* Every token takes modulation row zero: nothing is held. The map exists
     * because the kernel indexes through it, and because conditioning -- an
     * image held at timestep zero -- is what would give it a second row. */
    const uint32_t map_rows = video_rows > audio_rows ? video_rows : audio_rows;
    uint32_t *level_of = calloc(map_rows, sizeof(*level_of));
    if (!level_of) oops(&r, "cannot allocate a row map");
    if (!r.failed) {
        space.row_map_video = h3_gpu_tensor_from_u32(gpu, level_of, video_rows);
        space.row_map_audio = h3_gpu_tensor_from_u32(gpu, level_of, audio_rows);
        if (!space.row_map_video || !space.row_map_audio)
            oops(&r, "cannot upload a row map");
    }
    if (!r.failed) {
        s.video_pre = h3_gpu_tensor_new_bf16(gpu, video_state);
        s.audio_pre = h3_gpu_tensor_new_bf16(gpu, audio_state);
        s.video_scaled = h3_gpu_tensor_new_bf16(gpu, video_state);
        s.audio_scaled = h3_gpu_tensor_new_bf16(gpu, audio_state);
        s.video_branch = h3_gpu_tensor_new_bf16(gpu, video_state);
        s.audio_branch = h3_gpu_tensor_new_bf16(gpu, audio_state);
        if (!s.video_pre || !s.audio_pre || !s.video_scaled || !s.audio_scaled ||
            !s.video_branch || !s.audio_branch)
            oops(&r, "cannot allocate block state");
    }

    /* ------------------------------------------------------- the latents */

    float *video_tokens = NULL, *audio_tokens = NULL;
    float *video_out = NULL, *audio_out = NULL;
    float *video_velocity = NULL, *audio_velocity = NULL;
    if (!r.failed) {
        video_tokens = malloc(video_state * sizeof(float));
        audio_tokens = malloc(audio_state * sizeof(float));
        video_out = malloc(video_state * sizeof(float));
        audio_out = malloc(audio_state * sizeof(float));
        video_velocity = malloc((size_t)video_rows * LATENT * sizeof(float));
        audio_velocity = malloc((size_t)audio_rows * LATENT * sizeof(float));
        if (!video_tokens || !audio_tokens || !video_out || !audio_out ||
            !video_velocity || !audio_velocity)
            oops(&r, "cannot allocate the loop state");
    }

    const char *prefetch_env = getenv("H3_LTX_PREFETCH");
    const int prefetch = !prefetch_env || !*prefetch_env || *prefetch_env != '0';

    /* ------------------------------------------------------------ the loop */

    for (int step = 0; step < steps && !r.failed; step++) {
        conditioning cond;
        memset(&cond, 0, sizeof(cond));
        const float *slice[MODULE_COUNT];
        for (int index = 0; index < MODULE_COUNT; index++)
            slice[index] = modulation[index] + (size_t)step * LEVELS *
                           MODULES[index].slots * MODULES[index].width;
        cond.video_timestep = slice[0];
        cond.audio_timestep = slice[1];
        cond.video_prompt = slice[2];
        cond.audio_prompt = slice[3];
        cond.video_cross = slice[4];
        cond.audio_cross = slice[5];
        cond.video_gate = slice[6];
        cond.audio_gate = slice[7];
        cond.video_embedded = embedded[0] + (size_t)step * LEVELS * VIDEO_DIM;
        cond.audio_embedded = embedded[1] + (size_t)step * LEVELS * AUDIO_DIM;
        cond.video_context = video_context;
        cond.audio_context = audio_context;

        linear_rows(video_tokens, video_latent, video_patch_weight,
                    video_patch_bias, video_rows, LATENT, VIDEO_DIM);
        for (uint32_t row = 0; row < r.height * r.width; row++)
            for (int index = 0; index < VIDEO_DIM; index++)
                video_tokens[(size_t)row * VIDEO_DIM + index] +=
                    keyframe_marker[index];
        linear_rows(audio_tokens, audio_latent, audio_patch_weight,
                    audio_patch_bias, audio_rows, LATENT, AUDIO_DIM);
        s.video = upload(&r, video_tokens, video_state);
        s.audio = upload(&r, audio_tokens, audio_state);

        /* Block N + 1 loads on a worker while block N runs on the GPU, which
         * is the shape `h3_dit.c` settled on for H3's SSD streaming.
         *
         * Prefetch stops at the step boundary rather than wrapping as H3's
         * does: `cond` carries this step's modulation, so block 0 of the next
         * step cannot be built until the step advances. That forfeits one
         * block of 48, which the measurement says is not worth chasing --
         * loading already falls under a second. */
        preload_job jobs[2];
        memset(jobs, 0, sizeof(jobs));
        int slot = 0;
        load_block(&r, 0, &cond, &jobs[0].weights);
        for (int index = 0; index < TOTAL_BLOCKS; index++) {
            /* Asked between blocks rather than between steps: a step is a
             * minute at 512 square, and a cancel that takes a minute to land
             * reads as a hang. `step` counts *completed* steps either way, so
             * it never goes backwards. */
            if (tick && !r.failed && !tick(step, steps, tick_context)) stop(&r);
            pthread_t worker;
            int started = 0;
            if (!r.failed && prefetch && index + 1 < TOTAL_BLOCKS) {
                start_preload(&r, &jobs[slot ^ 1], index + 1, &cond);
                if (pthread_create(&worker, NULL, preload_thread,
                                   &jobs[slot ^ 1]) != 0)
                    oops(&r, "cannot start the prefetch for block %d",
                         index + 1);
                else
                    started = 1;
            }
            if (!r.failed) {
                GPU_OP(&r, h3_gpu_begin(gpu), "begin block");
                run_block(&r, &jobs[slot].weights, &s, &space);
                GPU_OP(&r, h3_gpu_submit(gpu), "submit block");
            }
            if (started) {
                /* Joined even when the run has already failed: the worker owns
                 * GPU allocations this thread is about to free. */
                pthread_join(worker, NULL);
                adopt_preload(&r, &jobs[slot ^ 1]);
            } else if (!r.failed && !prefetch && index + 1 < TOTAL_BLOCKS) {
                load_block(&r, index + 1, &cond, &jobs[slot ^ 1].weights);
            }
            free_block(&jobs[slot].weights);
            slot ^= 1;
            if (r.failed) break;
        }
        /* Whatever the far slot holds -- the block loaded but never run when
         * the loop broke early, and nothing at all when it did not. */
        free_block(&jobs[slot].weights);

        read_bf16_as_f32(&r, s.video, video_out, video_state);
        read_bf16_as_f32(&r, s.audio, audio_out, audio_state);
        h3_gpu_tensor_free(s.video); s.video = NULL;
        h3_gpu_tensor_free(s.audio); s.audio = NULL;
        output_head(&r, video_out, video_head_table, cond.video_embedded,
                    level_of, video_head_weight, video_head_bias,
                    video_velocity, video_rows, VIDEO_DIM);
        output_head(&r, audio_out, audio_head_table, cond.audio_embedded,
                    level_of, audio_head_weight, audio_head_bias,
                    audio_velocity, audio_rows, AUDIO_DIM);
        if (r.failed) break;

        advance(video_latent, video_velocity, sigmas[step], sigmas[step + 1],
                (size_t)video_rows * LATENT);
        advance(audio_latent, audio_velocity, sigmas[step], sigmas[step + 1],
                (size_t)audio_rows * LATENT);
        for (size_t index = 0; index < (size_t)video_rows * LATENT; index++)
            if (!isfinite(video_latent[index])) {
                oops(&r, "a video latent is not finite after step %d",
                     step + 1);
                break;
            }
        for (size_t index = 0; index < (size_t)audio_rows * LATENT; index++)
            if (!isfinite(audio_latent[index])) {
                oops(&r, "an audio latent is not finite after step %d",
                     step + 1);
                break;
            }
    }
    if (tick && !r.failed && !tick(steps, steps, tick_context)) stop(&r);

    /* ------------------------------------------------------------ cleanup */

    for (int index = 0; index < MODULE_COUNT; index++) {
        free(modulation[index]);
        free(embedded[index]);
    }
    free(video_patch_weight); free(video_patch_bias);
    free(audio_patch_weight); free(audio_patch_bias);
    free(video_head_table); free(audio_head_table);
    free(video_head_weight); free(video_head_bias);
    free(audio_head_weight); free(audio_head_bias);
    free(keyframe_marker);
    free(video_tokens); free(audio_tokens);
    free(video_out); free(audio_out);
    free(video_velocity); free(audio_velocity);
    free(level_of);
    h3_gpu_tensor_free(s.video); h3_gpu_tensor_free(s.audio);
    h3_gpu_tensor_free(s.video_pre); h3_gpu_tensor_free(s.audio_pre);
    h3_gpu_tensor_free(s.video_scaled); h3_gpu_tensor_free(s.audio_scaled);
    h3_gpu_tensor_free(s.video_branch); h3_gpu_tensor_free(s.audio_branch);
    h3_gpu_tensor_free(s.video_pe.cos); h3_gpu_tensor_free(s.video_pe.sin);
    h3_gpu_tensor_free(s.audio_pe.cos); h3_gpu_tensor_free(s.audio_pe.sin);
    h3_gpu_tensor_free(s.video_cross_pe.cos);
    h3_gpu_tensor_free(s.video_cross_pe.sin);
    h3_gpu_tensor_free(s.audio_cross_pe.cos);
    h3_gpu_tensor_free(s.audio_cross_pe.sin);
    h3_gpu_tensor_free(space.query); h3_gpu_tensor_free(space.key);
    h3_gpu_tensor_free(space.value); h3_gpu_tensor_free(space.heads);
    h3_gpu_tensor_free(space.logits);
    h3_gpu_tensor_free(space.rotated_query);
    h3_gpu_tensor_free(space.rotated_kv); h3_gpu_tensor_free(space.inner);
    h3_gpu_tensor_free(space.ones_video); h3_gpu_tensor_free(space.ones_audio);
    h3_gpu_tensor_free(space.row_map_video);
    h3_gpu_tensor_free(space.row_map_audio);
    return !r.failed;
}
