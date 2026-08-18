/* LTX-2.5's embeddings connector, as a library stage.
 *
 * Lifted from `Vendor/h3.c/tests/test_real_ltx_connector.c`, which keeps the
 * reference comparisons and stays the place to reproduce a disagreement. The
 * arithmetic is unchanged; the exits are error returns, as in the other two.
 *
 * Four conventions worth having at hand, each read off the released weights
 * rather than a library default:
 *
 *   - **the norms are parameter-free.** Every norm in the Gemma tower carries
 *     a learned scale; none of these do, and the checkpoint's silence is the
 *     tell;
 *
 *   - **q_norm and k_norm run over the full inner width, not per head.** The
 *     tower's are per head, and the two disagree by a factor that varies head
 *     to head -- so this is the one place the tower's convention must *not* be
 *     carried across;
 *
 *   - **the gate reads the normed input before the rotation.** It is a small
 *     dense projection, so it never sees a ConvRot-rotated activation, unlike
 *     every quantized projection beside it; and
 *
 *   - **the feed-forward is a plain GELU at 4x, not a gated pair.**
 *     `net.0.proj` widens and `net.2` consumes all of it; there is no w1/w3
 *     here to fuse or to confuse.
 */
#include "ltx_connector.h"

#include "h3_safetensors.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    VIDEO_DIM = LTX_CONNECTOR_VIDEO_DIM,
    VIDEO_HEAD = 128,
    AUDIO_DIM = LTX_CONNECTOR_AUDIO_DIM,
    AUDIO_HEAD = 64,
    HEADS = 32,
    BLOCKS = 8,
    REGISTERS = LTX_CONNECTOR_REGISTERS,
    MAX_POS = 4096,
    CONVROT_GROUP = 256,
    FF_MULTIPLE = 4
};

#define CONNECTOR_EPSILON 1e-6f

static const double THETA = 10000.0;

typedef struct {
    const char *name;
    const char *prefix;
    uint32_t dim;
    uint32_t head_dim;
    uint32_t ff_dim;
} stream;

static const stream STREAMS[2] = {
    {"video", "model.diffusion_model.video_embeddings_connector.",
     VIDEO_DIM, VIDEO_HEAD, VIDEO_DIM * FF_MULTIPLE},
    {"audio", "model.diffusion_model.audio_embeddings_connector.",
     AUDIO_DIM, AUDIO_HEAD, AUDIO_DIM * FF_MULTIPLE}
};

/* ---------------------------------------------------------------- the run */

typedef struct {
    h3_gpu *gpu;
    const h3_weight_store *store;
    char *error;
    size_t error_size;
    int failed;
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
    const uint32_t rounded = bits + 0x7fff + ((bits >> 16) & 1);
    return (uint16_t)(rounded >> 16);
}

/* ----------------------------------------------------------------- loading */

static h3_gpu_tensor *load_bf16(run *r, const char *name, int ndim,
                                uint64_t first, uint64_t second) {
    if (r->failed) return NULL;
    uint64_t shape[2] = {first, second};
    char why[512];
    h3_gpu_tensor *tensor = h3_weight_load_bf16(r->store, r->gpu, name, ndim,
                                                shape, why, sizeof(why));
    if (!tensor) oops(r, "cannot load %s: %s", name, why);
    return tensor;
}

typedef struct {
    h3_gpu_tensor *weight;
    h3_gpu_tensor *scales;
    h3_gpu_tensor *bias;
} projection;

/* Every projection in the connector carries a bias, unlike the DiT's video
 * feed-forward, whose ff_bias is false. */
static void load_projection(run *r, const char *prefix, const char *suffix,
                            uint64_t out_dim, uint64_t in_dim,
                            projection *into) {
    memset(into, 0, sizeof(*into));
    if (r->failed) return;
    char name[224], bias[224], why[512];
    snprintf(name, sizeof(name), "%s%s.weight", prefix, suffix);
    snprintf(bias, sizeof(bias), "%s%s.bias", prefix, suffix);
    if (!h3_weight_load_i8_linear(r->store, r->gpu, name, out_dim, in_dim,
                                  &into->weight, &into->scales, why,
                                  sizeof(why))) {
        oops(r, "cannot load %s: %s", name, why);
        return;
    }
    uint32_t group = 0;
    if (!h3_weight_i8_linear_convrot_group(r->store, name, &group, why,
                                           sizeof(why))) {
        oops(r, "cannot read the ConvRot marker for %s: %s", name, why);
        return;
    }
    if (group != CONVROT_GROUP) {
        oops(r, "%s has ConvRot group %u, expected %d", name, group,
             CONVROT_GROUP);
        return;
    }
    into->bias = load_bf16(r, bias, 1, out_dim, 0);
}

static void free_projection(projection *which) {
    h3_gpu_tensor_free(which->weight);
    h3_gpu_tensor_free(which->scales);
    h3_gpu_tensor_free(which->bias);
    memset(which, 0, sizeof(*which));
}

typedef struct {
    projection query, key, value, output, ff_in, ff_out;
    h3_gpu_tensor *query_norm;
    h3_gpu_tensor *key_norm;
    h3_gpu_tensor *gate_weight;
    h3_gpu_tensor *gate_bias;
} block_weights;

static void load_block(run *r, const stream *which, int index,
                       block_weights *weights) {
    char prefix[224];
    snprintf(prefix, sizeof(prefix), "%stransformer_1d_blocks.%d.",
             which->prefix, index);
    memset(weights, 0, sizeof(*weights));
    load_projection(r, prefix, "attn1.to_q", which->dim, which->dim,
                    &weights->query);
    load_projection(r, prefix, "attn1.to_k", which->dim, which->dim,
                    &weights->key);
    load_projection(r, prefix, "attn1.to_v", which->dim, which->dim,
                    &weights->value);
    load_projection(r, prefix, "attn1.to_out.0", which->dim, which->dim,
                    &weights->output);
    load_projection(r, prefix, "ff.net.0.proj", which->ff_dim, which->dim,
                    &weights->ff_in);
    load_projection(r, prefix, "ff.net.2", which->dim, which->ff_dim,
                    &weights->ff_out);
    char name[224];
    snprintf(name, sizeof(name), "%sattn1.q_norm.weight", prefix);
    weights->query_norm = load_bf16(r, name, 1, which->dim, 0);
    snprintf(name, sizeof(name), "%sattn1.k_norm.weight", prefix);
    weights->key_norm = load_bf16(r, name, 1, which->dim, 0);
    /* One logit per head from the block's *normed input*, taken before any
     * rotation. Not quantized. */
    snprintf(name, sizeof(name), "%sattn1.to_gate_logits.weight", prefix);
    weights->gate_weight = load_bf16(r, name, 2, HEADS, which->dim);
    snprintf(name, sizeof(name), "%sattn1.to_gate_logits.bias", prefix);
    weights->gate_bias = load_bf16(r, name, 1, HEADS, 0);
}

static void free_block(block_weights *weights) {
    free_projection(&weights->query);
    free_projection(&weights->key);
    free_projection(&weights->value);
    free_projection(&weights->output);
    free_projection(&weights->ff_in);
    free_projection(&weights->ff_out);
    h3_gpu_tensor_free(weights->query_norm);
    h3_gpu_tensor_free(weights->key_norm);
    h3_gpu_tensor_free(weights->gate_weight);
    h3_gpu_tensor_free(weights->gate_bias);
    memset(weights, 0, sizeof(*weights));
}

/* -------------------------------------------------------------------- rope */

/* LTX's split rope over one axis, which is not the tower's construction at any
 * point. Frequencies spread geometrically from 1 to theta -- theta^(i/(n-1)) --
 * scaled by pi/2, and positions map onto [-1, 1] through max_pos rather than
 * being used directly. The result is split across heads, so head h owns
 * frequencies h*half through h*half+half-1 and no two heads rotate alike.
 *
 * Laid out [head][token][half] for the kernel's head stride, and computed in
 * double because the released config asks for float64 frequencies. */
static void rope_tables(run *r, h3_gpu_tensor **cosine, h3_gpu_tensor **sine,
                        uint32_t span, uint32_t dim, uint32_t head_dim) {
    if (r->failed) return;
    const uint32_t half = head_dim / 2;
    const uint32_t count = dim / 2;
    if (count != HEADS * half) {
        oops(r, "the frequency count %u does not fill %d heads of %u", count,
             HEADS, half);
        return;
    }
    float *cos_table = malloc((size_t)count * span * sizeof(*cos_table));
    float *sin_table = malloc((size_t)count * span * sizeof(*sin_table));
    if (!cos_table || !sin_table) {
        oops(r, "cannot allocate the rotary tables");
        free(cos_table); free(sin_table);
        return;
    }
    for (uint32_t index = 0; index < count; index++) {
        const double exponent = count > 1 ?
            (double)index / (double)(count - 1) : 0.0;
        const double frequency = pow(THETA, exponent) * M_PI / 2.0;
        const uint32_t head = index / half;
        const uint32_t slot = index % half;
        for (uint32_t token = 0; token < span; token++) {
            const double fractional = (double)token / (double)MAX_POS;
            const double angle = frequency * (fractional * 2.0 - 1.0);
            const size_t at = (size_t)head * span * half +
                              (size_t)token * half + slot;
            cos_table[at] = (float)cos(angle);
            sin_table[at] = (float)sin(angle);
        }
    }
    *cosine = h3_gpu_tensor_from_f32(r->gpu, cos_table, (size_t)count * span);
    *sine = h3_gpu_tensor_from_f32(r->gpu, sin_table, (size_t)count * span);
    if (!*cosine || !*sine) oops(r, "cannot upload the rotary tables");
    free(cos_table);
    free(sin_table);
}

/* ------------------------------------------------------------------- block */

typedef struct {
    h3_gpu_tensor *normed, *query, *key, *value, *heads, *logits, *inner;
    h3_gpu_tensor *ones, *cosine, *sine;
} scratch;

static int linear_i8(run *r, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                     const projection *weight, uint32_t rows, uint32_t in_dim,
                     uint32_t out_dim) {
    return h3_gpu_linear_i8_weight_bf16(r->gpu, output, input, weight->weight,
                                        weight->scales, weight->bias, rows,
                                        in_dim, out_dim);
}

static void run_block(run *r, const stream *which, const block_weights *weight,
                      scratch *space, h3_gpu_tensor *hidden, uint32_t span) {
    const uint32_t dim = which->dim;
    const uint32_t head_dim = which->head_dim;
    const uint32_t stride = span * (head_dim / 2);

    GPU_OP(r, h3_gpu_rms_norm_bf16(r->gpu, space->normed, hidden, space->ones,
                                   span, dim, CONNECTOR_EPSILON),
           "attention norm");
    GPU_OP(r, h3_gpu_linear_bf16(r->gpu, space->logits, space->normed,
                                 weight->gate_weight, weight->gate_bias, span,
                                 dim, HEADS), "gate logits");
    GPU_OP(r, h3_gpu_convrot_bf16(r->gpu, space->normed, space->normed, span,
                                  dim, CONVROT_GROUP), "Q/K/V ConvRot");
    GPU_OP(r, linear_i8(r, space->query, space->normed, &weight->query, span,
                        dim, dim), "query projection");
    GPU_OP(r, linear_i8(r, space->key, space->normed, &weight->key, span, dim,
                        dim), "key projection");
    GPU_OP(r, linear_i8(r, space->value, space->normed, &weight->value, span,
                        dim, dim), "value projection");
    GPU_OP(r, h3_gpu_rms_norm_bf16(r->gpu, space->query, space->query,
                                   weight->query_norm, span, dim,
                                   CONNECTOR_EPSILON), "query norm");
    GPU_OP(r, h3_gpu_rms_norm_bf16(r->gpu, space->key, space->key,
                                   weight->key_norm, span, dim,
                                   CONNECTOR_EPSILON), "key norm");
    GPU_OP(r, h3_gpu_rope_rows_bf16(r->gpu, space->query, space->cosine,
                                    space->sine, span, HEADS, head_dim, stride),
           "query rope");
    GPU_OP(r, h3_gpu_rope_rows_bf16(r->gpu, space->key, space->cosine,
                                    space->sine, span, HEADS, head_dim, stride),
           "key rope");
    /* Bidirectional over the whole span: the registers have already taken the
     * padded slots and the reference zeroes its mask when it does that, so
     * there is nothing left to mask. The value is neither normed nor rotated. */
    GPU_OP(r, h3_gpu_sdpa_bf16(r->gpu, space->heads, space->query, space->key,
                               space->value, span, HEADS, head_dim,
                               1.0f / sqrtf((float)head_dim)), "attention");
    GPU_OP(r, h3_gpu_head_gate_bf16(r->gpu, space->heads, space->logits, span,
                                    HEADS, head_dim), "head gate");
    GPU_OP(r, h3_gpu_convrot_bf16(r->gpu, space->heads, space->heads, span, dim,
                                  CONVROT_GROUP), "output ConvRot");
    GPU_OP(r, linear_i8(r, space->normed, space->heads, &weight->output, span,
                        dim, dim), "output projection");
    GPU_OP(r, h3_gpu_add_bf16(r->gpu, hidden, hidden, space->normed,
                              span * dim), "attention residual");

    GPU_OP(r, h3_gpu_rms_norm_bf16(r->gpu, space->normed, hidden, space->ones,
                                   span, dim, CONNECTOR_EPSILON),
           "feed-forward norm");
    GPU_OP(r, h3_gpu_convrot_bf16(r->gpu, space->normed, space->normed, span,
                                  dim, CONVROT_GROUP), "feed-forward ConvRot");
    GPU_OP(r, linear_i8(r, space->inner, space->normed, &weight->ff_in, span,
                        dim, which->ff_dim), "feed-forward in");
    GPU_OP(r, h3_gpu_gelu_bf16(r->gpu, space->inner, space->inner,
                               span * which->ff_dim, 1), "GELU");
    GPU_OP(r, h3_gpu_convrot_bf16(r->gpu, space->inner, space->inner, span,
                                  which->ff_dim, CONVROT_GROUP),
           "feed-forward out ConvRot");
    GPU_OP(r, linear_i8(r, space->normed, space->inner, &weight->ff_out, span,
                        which->ff_dim, dim), "feed-forward out");
    GPU_OP(r, h3_gpu_add_bf16(r->gpu, hidden, hidden, space->normed,
                              span * dim), "feed-forward residual");
}

/* ------------------------------------------------------------------ stream */

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

/* Padded slots take the registers, tiled to cover the span -- the reference
 * repeats the table span/128 times, so slot s takes register s % 128. Valid
 * slots keep their aggregated features. */
static void run_stream(run *r, const stream *which, const float *features,
                       uint32_t tokens, uint32_t span, float *out) {
    const uint32_t dim = which->dim;
    const size_t state = (size_t)span * dim;

    char name[224];
    snprintf(name, sizeof(name), "%slearnable_registers", which->prefix);
    h3_gpu_tensor *registers = load_bf16(r, name, 2, REGISTERS, dim);
    float *host_registers = NULL;
    if (!r->failed) {
        host_registers = malloc((size_t)REGISTERS * dim *
                                sizeof(*host_registers));
        if (!host_registers) oops(r, "cannot allocate the registers");
    }
    read_bf16_as_f32(r, registers, host_registers, (size_t)REGISTERS * dim);
    h3_gpu_tensor_free(registers);

    uint16_t *staged = NULL;
    if (!r->failed) {
        staged = malloc(state * sizeof(*staged));
        if (!staged) oops(r, "cannot allocate the connector input");
    }
    if (!r->failed)
        for (uint32_t token = 0; token < span; token++) {
            const float *source = token < tokens ?
                features + (size_t)token * dim :
                host_registers + (size_t)(token % REGISTERS) * dim;
            for (uint32_t index = 0; index < dim; index++)
                staged[(size_t)token * dim + index] = to_bf16(source[index]);
        }
    free(host_registers);
    h3_gpu_tensor *hidden = NULL;
    if (!r->failed) {
        hidden = h3_gpu_tensor_from_bf16(r->gpu, staged, state);
        if (!hidden) oops(r, "cannot upload the connector input");
    }
    free(staged);

    scratch space;
    memset(&space, 0, sizeof(space));
    if (!r->failed) {
        space.normed = h3_gpu_tensor_new_bf16(r->gpu, state);
        space.query = h3_gpu_tensor_new_bf16(r->gpu, state);
        space.key = h3_gpu_tensor_new_bf16(r->gpu, state);
        space.value = h3_gpu_tensor_new_bf16(r->gpu, state);
        space.heads = h3_gpu_tensor_new_bf16(r->gpu, state);
        space.logits = h3_gpu_tensor_new_bf16(r->gpu, (size_t)span * HEADS);
        space.inner = h3_gpu_tensor_new_bf16(r->gpu,
                                             (size_t)span * which->ff_dim);
        if (!space.normed || !space.query || !space.key || !space.value ||
            !space.heads || !space.logits || !space.inner)
            oops(r, "cannot allocate connector scratch");
    }
    if (!r->failed) {
        uint16_t *ones = malloc(dim * sizeof(*ones));
        if (!ones) oops(r, "cannot allocate the norm weight");
        else {
            for (uint32_t index = 0; index < dim; index++) ones[index] = 0x3f80;
            space.ones = h3_gpu_tensor_from_bf16(r->gpu, ones, dim);
            if (!space.ones) oops(r, "cannot upload the norm weight");
            free(ones);
        }
    }
    rope_tables(r, &space.cosine, &space.sine, span, dim, which->head_dim);

    for (int index = 0; index < BLOCKS && !r->failed; index++) {
        block_weights weights;
        load_block(r, which, index, &weights);
        GPU_OP(r, h3_gpu_begin(r->gpu), "begin connector block");
        run_block(r, which, &weights, &space, hidden, span);
        GPU_OP(r, h3_gpu_submit(r->gpu), "submit connector block");
        free_block(&weights);
    }
    /* connector_norm_output: one more parameter-free norm on the way out. */
    GPU_OP(r, h3_gpu_begin(r->gpu), "begin output norm");
    GPU_OP(r, h3_gpu_rms_norm_bf16(r->gpu, space.normed, hidden, space.ones,
                                   span, dim, CONNECTOR_EPSILON),
           "output norm");
    GPU_OP(r, h3_gpu_submit(r->gpu), "submit output norm");
    read_bf16_as_f32(r, space.normed, out, state);

    if (!r->failed)
        for (size_t index = 0; index < state; index++)
            if (!isfinite(out[index])) {
                oops(r, "a %s connector output is not finite", which->name);
                break;
            }

    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(space.normed); h3_gpu_tensor_free(space.query);
    h3_gpu_tensor_free(space.key); h3_gpu_tensor_free(space.value);
    h3_gpu_tensor_free(space.heads); h3_gpu_tensor_free(space.logits);
    h3_gpu_tensor_free(space.inner); h3_gpu_tensor_free(space.ones);
    h3_gpu_tensor_free(space.cosine); h3_gpu_tensor_free(space.sine);
}

int ltx_connector_run(h3_gpu *gpu, const h3_weight_store *dit,
                      const float *video_features, const float *audio_features,
                      uint32_t tokens, uint32_t span, float *video_context,
                      float *audio_context, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!gpu || !dit || !video_features || !audio_features || !video_context ||
        !audio_context || !tokens || !span || tokens > span) {
        if (error && error_size)
            snprintf(error, error_size, "ltx_connector_run wants a gpu, the "
                                        "DiT store, both feature streams and "
                                        "0 < tokens <= span");
        return 0;
    }
    run r = {0};
    r.gpu = gpu; r.store = dit; r.error = error; r.error_size = error_size;
    run_stream(&r, &STREAMS[0], video_features, tokens, span, video_context);
    run_stream(&r, &STREAMS[1], audio_features, tokens, span, audio_context);
    return !r.failed;
}
