/* LTX-2.5's Gemma 4 tower and feature aggregation, as a library stage.
 *
 * Lifted from `Vendor/h3.c/tests/test_real_ltx_text.c`, which keeps the
 * reference comparisons and stays the place to reproduce a disagreement.
 *
 * Conventions worth having at hand, each read off the released weights against
 * a plausible library default:
 *
 *   - **layer types come from the checkpoint's shapes**, not the config's
 *     `layer_types` list, and the two are cross-checked: full attention at
 *     every sixth index. `resolve_all` asserts that rather than inheriting it;
 *
 *   - **the global layers rotate a quarter of their 512-wide head** and leave
 *     the rest at an identity rotation, with the exponent divided by the
 *     *full* head dim rather than the rotated width. Dividing by the rotated
 *     width is the obvious misreading and gives entirely different frequencies
 *     while producing perfectly well-formed conditioning;
 *
 *   - **`attention_k_eq_v`**: a global layer's value is the raw key
 *     projection, taken before k_norm and before the rotation, and the
 *     checkpoint carries no v_proj for those layers at all;
 *
 *   - **q_norm and k_norm are per head.** The connector's are over the full
 *     inner width, and the two disagree by a factor that varies head to head;
 *
 *   - **the embedding scale is BF16 62.0, not sqrt(3840) = 61.96773.** The
 *     reference multiplies by a BF16 copy, so rounding here keeps this path on
 *     its arithmetic -- and held against an F32 reference alone the exact
 *     value scores *better*, which is how the wrong one hides;
 *
 *   - **GeGLU, not SwiGLU**, no 1/sqrt(head_dim) anywhere -- Gemma 4 sets the
 *     softmax scale to 1.0 -- and a trained scalar on the way out of each
 *     layer; and
 *
 *   - **the aggregation lays the 49 states out layer-minor**,
 *     `normed[token][d * 49 + l]`, which is what `reshape(B, T, D * L)`
 *     produces from a `[B, T, D, L]` stack. Its rescale is
 *     `sqrt(out_dim / 3840)`, so it is not the same number for the 4096-wide
 *     video embedding as for the 2048-wide audio one; reusing one for the
 *     other is wrong by sqrt(2) and entirely plausible.
 */
#include "ltx_text.h"

#include "h3_safetensors.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    HIDDEN = 3840,
    INTERMEDIATE = 15360,
    LAYERS = LTX_TEXT_LAYERS,
    VOCAB = 262144,
    QUERY_HEADS = 16,
    SLIDING_HEAD = 256,
    SLIDING_KV_HEADS = 8,
    GLOBAL_HEAD = 512,
    GLOBAL_KV_HEADS = 1,
    CONVROT_GROUP = 256,
    HIDDEN_STATES = LTX_TEXT_STATES,
    VIDEO_DIM = LTX_TEXT_VIDEO_DIM,
    AUDIO_DIM = LTX_TEXT_AUDIO_DIM,
    AGGREGATE_INPUT = HIDDEN * HIDDEN_STATES
};

#define RMS_EPSILON 1e-6f
#define AGGREGATE_EPSILON 1e-6f

static const double SLIDING_THETA = 10000.0;
static const double GLOBAL_THETA = 1000000.0;
static const double GLOBAL_PARTIAL = 0.25;

/* ---------------------------------------------------------------- the run */

typedef struct {
    h3_gpu *gpu;
    const h3_weight_store *store;
    char *error;
    size_t error_size;
    int failed;
    int cancelled;
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

static float round_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounded = bits + 0x7fff + ((bits >> 16) & 1);
    const uint32_t truncated = rounded & 0xffff0000u;
    float result;
    memcpy(&result, &truncated, sizeof(result));
    return result;
}

/* -------------------------------------------------------------- resolving */

static const h3_st_tensor *expect(run *r, const char *name, h3_dtype dtype,
                                  int ndim, uint64_t first, uint64_t second) {
    if (r->failed) return NULL;
    const h3_st_tensor *tensor = h3_weight_find(r->store, name, NULL);
    if (!tensor) { oops(r, "the checkpoint has no tensor %s", name); return NULL; }
    if (tensor->dtype != dtype) {
        oops(r, "%s is %s, expected %s", name, h3_dtype_name(tensor->dtype),
             h3_dtype_name(dtype));
        return NULL;
    }
    if (tensor->ndim != ndim || tensor->shape[0] != first ||
        (ndim > 1 && tensor->shape[1] != second)) {
        oops(r, "%s has the wrong shape for [%llu, %llu]", name,
             (unsigned long long)first, (unsigned long long)second);
        return NULL;
    }
    return tensor;
}

static void expect_projection(run *r, const char *prefix, const char *suffix,
                              uint64_t out_dim, uint64_t in_dim,
                              uint32_t *group) {
    char name[224], scales[224], why[512];
    snprintf(name, sizeof(name), "%s%s.weight", prefix, suffix);
    expect(r, name, H3_DTYPE_I8, 2, out_dim, in_dim);
    snprintf(scales, sizeof(scales), "%s%s.weight_scale", prefix, suffix);
    expect(r, scales, H3_DTYPE_F32, 2, out_dim, 1);
    if (r->failed) return;
    if (!h3_weight_i8_linear_convrot_group(r->store, name, group, why,
                                           sizeof(why)))
        oops(r, "cannot read the ConvRot marker for %s: %s", name, why);
    else if (*group != CONVROT_GROUP)
        oops(r, "%s has ConvRot group %u, expected %d", name, *group,
             CONVROT_GROUP);
}

/* Layer types from the checkpoint's own shapes, cross-checked against the
 * published pattern, and the ConvRot groups checked for agreement: the
 * rotation is applied to the *activation*, so every projection reading the
 * same activation must agree on it. Rotating once when they disagreed would
 * quietly corrupt the others rather than fail. */
static void resolve_all(run *r) {
    for (int index = 0; index < LAYERS && !r->failed; index++) {
        char prefix[96], name[224];
        snprintf(prefix, sizeof(prefix), "model.layers.%d.", index);
        snprintf(name, sizeof(name), "%sself_attn.q_proj.weight", prefix);
        const h3_st_tensor *tensor = h3_weight_find(r->store, name, NULL);
        if (!tensor) { oops(r, "the checkpoint has no layer %d", index); return; }
        int global;
        if (tensor->shape[0] == (uint64_t)QUERY_HEADS * SLIDING_HEAD) global = 0;
        else if (tensor->shape[0] == (uint64_t)QUERY_HEADS * GLOBAL_HEAD) global = 1;
        else {
            oops(r, "layer %d has a query projection of %llu rows, neither %d "
                 "sliding heads nor %d global ones", index,
                 (unsigned long long)tensor->shape[0], QUERY_HEADS, QUERY_HEADS);
            return;
        }
        snprintf(name, sizeof(name), "%sself_attn.v_proj.weight", prefix);
        if ((h3_weight_find(r->store, name, NULL) != NULL) == global) {
            oops(r, "layer %d looks %s by shape but %s a value projection",
                 index, global ? "global" : "sliding", global ? "has" : "lacks");
            return;
        }
        if (global != (index % 6 == 5)) {
            oops(r, "layer %d is %s, but the published layer_types put "
                 "full_attention at every sixth index", index,
                 global ? "global" : "sliding");
            return;
        }
        const uint32_t head = global ? GLOBAL_HEAD : SLIDING_HEAD;
        const uint64_t query_dim = (uint64_t)QUERY_HEADS * head;
        const uint64_t kv_dim = (uint64_t)(global ? GLOBAL_KV_HEADS
                                                  : SLIDING_KV_HEADS) * head;
        uint32_t q = 0, k = 0, v = 0, o = 0, gate = 0, up = 0, down = 0;
        expect_projection(r, prefix, "self_attn.q_proj", query_dim, HIDDEN, &q);
        expect_projection(r, prefix, "self_attn.k_proj", kv_dim, HIDDEN, &k);
        if (!global)
            expect_projection(r, prefix, "self_attn.v_proj", kv_dim, HIDDEN, &v);
        expect_projection(r, prefix, "self_attn.o_proj", HIDDEN, query_dim, &o);
        expect_projection(r, prefix, "mlp.gate_proj", INTERMEDIATE, HIDDEN, &gate);
        expect_projection(r, prefix, "mlp.up_proj", INTERMEDIATE, HIDDEN, &up);
        expect_projection(r, prefix, "mlp.down_proj", HIDDEN, INTERMEDIATE, &down);
        if (!r->failed && (q != k || (!global && q != v)))
            oops(r, "layer %d Q/K%s ConvRot groups disagree: %u, %u, %u", index,
                 global ? "" : "/V", q, k, v);
        if (!r->failed && gate != up)
            oops(r, "layer %d gate/up ConvRot groups disagree: %u, %u", index,
                 gate, up);
        (void)o; (void)down;
    }
}

/* ----------------------------------------------------------------- loading */

typedef struct { h3_gpu_tensor *weight, *scales; } projection;

typedef struct {
    int global;
    uint32_t head_dim, kv_heads, query_dim, kv_dim;
    float scalar;
    h3_gpu_tensor *input_norm, *post_attention_norm;
    h3_gpu_tensor *pre_ffn_norm, *post_ffn_norm;
    h3_gpu_tensor *query_norm, *key_norm;
    projection query, key, value, output, gate, up, down;
} layer_weights;

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

static void load_projection(run *r, const char *prefix, const char *suffix,
                            uint64_t out_dim, uint64_t in_dim,
                            projection *into) {
    memset(into, 0, sizeof(*into));
    if (r->failed) return;
    char name[224], why[512];
    snprintf(name, sizeof(name), "%s%s.weight", prefix, suffix);
    if (!h3_weight_load_i8_linear(r->store, r->gpu, name, out_dim, in_dim,
                                  &into->weight, &into->scales, why,
                                  sizeof(why)))
        oops(r, "cannot load %s: %s", name, why);
}

static void free_projection(projection *which) {
    h3_gpu_tensor_free(which->weight);
    h3_gpu_tensor_free(which->scales);
    memset(which, 0, sizeof(*which));
}

static float read_scalar(run *r, const char *name) {
    if (r->failed) return 1.0f;
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(r->store, name, &header);
    if (!tensor) { oops(r, "the checkpoint has no %s", name); return 1.0f; }
    uint16_t raw = 0;
    char why[512];
    if (!h3_st_read_data(header, tensor, &raw, sizeof(raw), why, sizeof(why))) {
        oops(r, "cannot read %s: %s", name, why);
        return 1.0f;
    }
    return from_bf16(raw);
}

static void load_layer(run *r, int index, layer_weights *weights) {
    char prefix[96];
    snprintf(prefix, sizeof(prefix), "model.layers.%d.", index);
    memset(weights, 0, sizeof(*weights));
    weights->global = index % 6 == 5;
    weights->head_dim = weights->global ? GLOBAL_HEAD : SLIDING_HEAD;
    weights->kv_heads = weights->global ? GLOBAL_KV_HEADS : SLIDING_KV_HEADS;
    weights->query_dim = QUERY_HEADS * weights->head_dim;
    weights->kv_dim = weights->kv_heads * weights->head_dim;

    char name[224];
#define NORM(field, suffix, width) do {                                       \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                     \
    weights->field = load_bf16(r, name, 1, (width), 0);                       \
} while (0)
    NORM(input_norm, "input_layernorm.weight", HIDDEN);
    NORM(post_attention_norm, "post_attention_layernorm.weight", HIDDEN);
    NORM(pre_ffn_norm, "pre_feedforward_layernorm.weight", HIDDEN);
    NORM(post_ffn_norm, "post_feedforward_layernorm.weight", HIDDEN);
    NORM(query_norm, "self_attn.q_norm.weight", weights->head_dim);
    NORM(key_norm, "self_attn.k_norm.weight", weights->head_dim);
#undef NORM
    snprintf(name, sizeof(name), "%slayer_scalar", prefix);
    weights->scalar = read_scalar(r, name);

    load_projection(r, prefix, "self_attn.q_proj", weights->query_dim, HIDDEN,
                    &weights->query);
    load_projection(r, prefix, "self_attn.k_proj", weights->kv_dim, HIDDEN,
                    &weights->key);
    if (!weights->global)
        load_projection(r, prefix, "self_attn.v_proj", weights->kv_dim, HIDDEN,
                        &weights->value);
    load_projection(r, prefix, "self_attn.o_proj", HIDDEN, weights->query_dim,
                    &weights->output);
    load_projection(r, prefix, "mlp.gate_proj", INTERMEDIATE, HIDDEN,
                    &weights->gate);
    load_projection(r, prefix, "mlp.up_proj", INTERMEDIATE, HIDDEN,
                    &weights->up);
    load_projection(r, prefix, "mlp.down_proj", HIDDEN, INTERMEDIATE,
                    &weights->down);
}

static void free_layer(layer_weights *weights) {
    h3_gpu_tensor_free(weights->input_norm);
    h3_gpu_tensor_free(weights->post_attention_norm);
    h3_gpu_tensor_free(weights->pre_ffn_norm);
    h3_gpu_tensor_free(weights->post_ffn_norm);
    h3_gpu_tensor_free(weights->query_norm);
    h3_gpu_tensor_free(weights->key_norm);
    free_projection(&weights->query);
    free_projection(&weights->key);
    if (!weights->global) free_projection(&weights->value);
    free_projection(&weights->output);
    free_projection(&weights->gate);
    free_projection(&weights->up);
    free_projection(&weights->down);
    memset(weights, 0, sizeof(*weights));
}

/* --------------------------------------------------------------- the rope */

/* One compact [tokens, head_dim / 2] pair, shared across heads. The global
 * layers rotate 64 of their 256 frequency pairs and leave the rest at zero --
 * an identity rotation -- with the exponent over the full head dim, which is
 * what `proportional` names. */
static void rope_tables(run *r, h3_gpu_tensor **cosine, h3_gpu_tensor **sine,
                        uint32_t tokens, uint32_t head_dim, double theta,
                        double partial) {
    if (r->failed) return;
    const uint32_t half = head_dim / 2;
    const uint32_t rotated = (uint32_t)(partial * (double)head_dim / 2.0);
    float *cos_table = calloc((size_t)tokens * half, sizeof(*cos_table));
    float *sin_table = calloc((size_t)tokens * half, sizeof(*sin_table));
    if (!cos_table || !sin_table) {
        oops(r, "cannot allocate rotary tables");
        free(cos_table); free(sin_table);
        return;
    }
    for (uint32_t position = 0; position < tokens; position++)
        for (uint32_t index = 0; index < half; index++) {
            const double inverse = index < rotated ?
                1.0 / pow(theta, (double)(2 * index) / (double)head_dim) : 0.0;
            const double angle = (double)position * inverse;
            cos_table[position * half + index] = (float)cos(angle);
            sin_table[position * half + index] = (float)sin(angle);
        }
    *cosine = h3_gpu_tensor_from_f32(r->gpu, cos_table, (size_t)tokens * half);
    *sine = h3_gpu_tensor_from_f32(r->gpu, sin_table, (size_t)tokens * half);
    if (!*cosine || !*sine) oops(r, "cannot upload rotary tables");
    free(cos_table);
    free(sin_table);
}

/* ------------------------------------------------------------------ layer */

typedef struct {
    h3_gpu_tensor *normed, *query, *key, *value, *heads, *projected;
    h3_gpu_tensor *gate, *up, *ones;
    h3_gpu_tensor *sliding_cos, *sliding_sin, *global_cos, *global_sin;
} scratch;

static void scratch_create(run *r, scratch *space, uint32_t tokens) {
    memset(space, 0, sizeof(*space));
    if (r->failed) return;
    const size_t widest_query = (size_t)tokens * QUERY_HEADS * GLOBAL_HEAD;
    const size_t widest_kv = (size_t)tokens * SLIDING_KV_HEADS * SLIDING_HEAD;
    space->normed = h3_gpu_tensor_new_bf16(r->gpu, (size_t)tokens * HIDDEN);
    space->query = h3_gpu_tensor_new_bf16(r->gpu, widest_query);
    space->key = h3_gpu_tensor_new_bf16(r->gpu, widest_kv);
    space->value = h3_gpu_tensor_new_bf16(r->gpu, widest_kv);
    space->heads = h3_gpu_tensor_new_bf16(r->gpu, widest_query);
    space->projected = h3_gpu_tensor_new_bf16(r->gpu, (size_t)tokens * HIDDEN);
    space->gate = h3_gpu_tensor_new_bf16(r->gpu, (size_t)tokens * INTERMEDIATE);
    space->up = h3_gpu_tensor_new_bf16(r->gpu, (size_t)tokens * INTERMEDIATE);
    if (!space->normed || !space->query || !space->key || !space->value ||
        !space->heads || !space->projected || !space->gate || !space->up) {
        oops(r, "cannot allocate tower scratch");
        return;
    }
    /* The value norm carries no learnable scale, so it is driven with ones
     * rather than a second kernel. 0x3f80 is 1.0f in BF16. */
    uint16_t *ones = malloc(GLOBAL_HEAD * sizeof(*ones));
    if (!ones) { oops(r, "cannot allocate the value norm weight"); return; }
    for (int index = 0; index < GLOBAL_HEAD; index++) ones[index] = 0x3f80;
    space->ones = h3_gpu_tensor_from_bf16(r->gpu, ones, GLOBAL_HEAD);
    free(ones);
    if (!space->ones) { oops(r, "cannot upload the value norm weight"); return; }

    rope_tables(r, &space->sliding_cos, &space->sliding_sin, tokens,
                SLIDING_HEAD, SLIDING_THETA, 1.0);
    rope_tables(r, &space->global_cos, &space->global_sin, tokens,
                GLOBAL_HEAD, GLOBAL_THETA, GLOBAL_PARTIAL);
}

static void scratch_free(scratch *space) {
    h3_gpu_tensor_free(space->normed);
    h3_gpu_tensor_free(space->query);
    h3_gpu_tensor_free(space->key);
    h3_gpu_tensor_free(space->value);
    h3_gpu_tensor_free(space->heads);
    h3_gpu_tensor_free(space->projected);
    h3_gpu_tensor_free(space->gate);
    h3_gpu_tensor_free(space->up);
    h3_gpu_tensor_free(space->ones);
    h3_gpu_tensor_free(space->sliding_cos);
    h3_gpu_tensor_free(space->sliding_sin);
    h3_gpu_tensor_free(space->global_cos);
    h3_gpu_tensor_free(space->global_sin);
    memset(space, 0, sizeof(*space));
}

static int linear_i8(run *r, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                     const projection *weight, uint32_t rows, uint32_t in_dim,
                     uint32_t out_dim) {
    return h3_gpu_linear_i8_weight_bf16(r->gpu, output, input, weight->weight,
                                        weight->scales, NULL, rows, in_dim,
                                        out_dim);
}

/* One decoder layer, in the order Gemma4UnifiedTextDecoderLayer runs it: a norm
 * sandwich around each residual, GeGLU rather than SwiGLU, no 1/sqrt(head_dim)
 * anywhere, and a trained scalar on the way out. */
static void run_layer(run *r, const layer_weights *weight, scratch *space,
                      h3_gpu_tensor *hidden, uint32_t tokens) {
    const uint32_t head_dim = weight->head_dim;
    const uint32_t kv_heads = weight->kv_heads;
    const uint32_t query_dim = weight->query_dim;
    const uint32_t kv_dim = weight->kv_dim;
    const h3_gpu_tensor *cosine = weight->global ? space->global_cos
                                                 : space->sliding_cos;
    const h3_gpu_tensor *sine = weight->global ? space->global_sin
                                               : space->sliding_sin;

    GPU_OP(r, h3_gpu_rms_norm_bf16(r->gpu, space->normed, hidden,
                                   weight->input_norm, tokens, HIDDEN,
                                   RMS_EPSILON), "input norm");
    GPU_OP(r, h3_gpu_convrot_bf16(r->gpu, space->normed, space->normed, tokens,
                                  HIDDEN, CONVROT_GROUP), "Q/K/V ConvRot");
    GPU_OP(r, linear_i8(r, space->query, space->normed, &weight->query, tokens,
                        HIDDEN, query_dim), "query projection");
    GPU_OP(r, linear_i8(r, space->key, space->normed, &weight->key, tokens,
                        HIDDEN, kv_dim), "key projection");
    if (weight->global) {
        /* attention_k_eq_v: the value is the raw key projection, before k_norm
         * and before the rotation. */
        GPU_OP(r, h3_gpu_copy_bf16(r->gpu, space->value, 0, space->key, 0,
                                   (size_t)tokens * kv_dim), "value from key");
    } else {
        GPU_OP(r, linear_i8(r, space->value, space->normed, &weight->value,
                            tokens, HIDDEN, kv_dim), "value projection");
    }
    GPU_OP(r, h3_gpu_head_rms_norm_bf16(r->gpu, space->query,
                                        weight->query_norm, tokens,
                                        QUERY_HEADS, head_dim, RMS_EPSILON),
           "query norm");
    GPU_OP(r, h3_gpu_head_rms_norm_bf16(r->gpu, space->key, weight->key_norm,
                                        tokens, kv_heads, head_dim,
                                        RMS_EPSILON), "key norm");
    GPU_OP(r, h3_gpu_rope_text_bf16(r->gpu, space->query, space->key, cosine,
                                    sine, tokens, QUERY_HEADS, kv_heads,
                                    head_dim, 0), "rotary");
    GPU_OP(r, h3_gpu_head_rms_norm_bf16(r->gpu, space->value, space->ones,
                                        tokens, kv_heads, head_dim,
                                        RMS_EPSILON), "value norm");
    GPU_OP(r, h3_gpu_gqa_causal_bf16(r->gpu, space->heads, space->query,
                                     space->key, space->value, tokens,
                                     QUERY_HEADS, kv_heads, head_dim, 1.0f),
           "causal attention");
    GPU_OP(r, h3_gpu_convrot_bf16(r->gpu, space->heads, space->heads, tokens,
                                  query_dim, CONVROT_GROUP), "output ConvRot");
    GPU_OP(r, linear_i8(r, space->projected, space->heads, &weight->output,
                        tokens, query_dim, HIDDEN), "output projection");
    GPU_OP(r, h3_gpu_rms_norm_bf16(r->gpu, space->projected, space->projected,
                                   weight->post_attention_norm, tokens, HIDDEN,
                                   RMS_EPSILON), "post-attention norm");
    GPU_OP(r, h3_gpu_add_bf16(r->gpu, hidden, hidden, space->projected,
                              tokens * HIDDEN), "attention residual");

    GPU_OP(r, h3_gpu_rms_norm_bf16(r->gpu, space->normed, hidden,
                                   weight->pre_ffn_norm, tokens, HIDDEN,
                                   RMS_EPSILON), "pre-FFN norm");
    GPU_OP(r, h3_gpu_convrot_bf16(r->gpu, space->normed, space->normed, tokens,
                                  HIDDEN, CONVROT_GROUP), "gate/up ConvRot");
    GPU_OP(r, linear_i8(r, space->gate, space->normed, &weight->gate, tokens,
                        HIDDEN, INTERMEDIATE), "MLP gate");
    GPU_OP(r, linear_i8(r, space->up, space->normed, &weight->up, tokens,
                        HIDDEN, INTERMEDIATE), "MLP up");
    GPU_OP(r, h3_gpu_gelu_mul_bf16(r->gpu, space->gate, space->gate, space->up,
                                   tokens * INTERMEDIATE), "GeGLU");
    GPU_OP(r, h3_gpu_convrot_bf16(r->gpu, space->gate, space->gate, tokens,
                                  INTERMEDIATE, CONVROT_GROUP), "down ConvRot");
    GPU_OP(r, linear_i8(r, space->projected, space->gate, &weight->down,
                        tokens, INTERMEDIATE, HIDDEN), "MLP down");
    GPU_OP(r, h3_gpu_rms_norm_bf16(r->gpu, space->projected, space->projected,
                                   weight->post_ffn_norm, tokens, HIDDEN,
                                   RMS_EPSILON), "post-FFN norm");
    GPU_OP(r, h3_gpu_add_bf16(r->gpu, hidden, hidden, space->projected,
                              tokens * HIDDEN), "MLP residual");
    GPU_OP(r, h3_gpu_scale_bf16(r->gpu, hidden, hidden, tokens * HIDDEN,
                                weight->scalar), "layer scalar");
}

/* ---------------------------------------------------------------- reading */

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

static h3_gpu_tensor *embed(run *r, const int32_t *ids, uint32_t tokens) {
    h3_gpu_tensor *table = load_bf16(r, "model.embed_tokens.weight", 2, VOCAB,
                                     HIDDEN);
    uint32_t *raw = NULL;
    if (!r->failed) {
        raw = malloc((size_t)tokens * sizeof(*raw));
        if (!raw) oops(r, "cannot allocate the token ids");
    }
    if (!r->failed)
        for (uint32_t index = 0; index < tokens; index++) {
            if (ids[index] < 0 || ids[index] >= VOCAB) {
                oops(r, "token id %d at %u is outside the vocabulary",
                     ids[index], index);
                break;
            }
            raw[index] = (uint32_t)ids[index];
        }
    h3_gpu_tensor *token_ids = NULL;
    if (!r->failed) {
        token_ids = h3_gpu_tensor_from_u32(r->gpu, raw, tokens);
        if (!token_ids) oops(r, "cannot upload the token ids");
    }
    free(raw);

    h3_gpu_tensor *hidden = NULL;
    if (!r->failed) {
        hidden = h3_gpu_tensor_new_bf16(r->gpu, (size_t)tokens * HIDDEN);
        if (!hidden) oops(r, "cannot allocate the hidden state");
    }
    GPU_OP(r, h3_gpu_begin(r->gpu), "begin embedding");
    GPU_OP(r, h3_gpu_embedding_bf16(r->gpu, hidden, table, token_ids, tokens,
                                    VOCAB, HIDDEN), "embedding lookup");
    /* sqrt(3840) is 61.96773, but the reference multiplies by a BF16 copy of
     * it, so the scale that actually runs is 62.0. */
    GPU_OP(r, h3_gpu_scale_bf16(r->gpu, hidden, hidden, tokens * HIDDEN,
                                round_bf16(sqrtf((float)HIDDEN))),
           "embedding scale");
    GPU_OP(r, h3_gpu_submit(r->gpu), "submit embedding");

    h3_gpu_tensor_free(table);
    h3_gpu_tensor_free(token_ids);
    return hidden;
}

/* -------------------------------------------------------- the aggregation */

static void aggregate_stream(run *r, const float *normed, uint32_t tokens,
                             const char *name, uint32_t out_dim, float *out) {
    if (r->failed) return;
    const float rescale = sqrtf((float)out_dim / (float)HIDDEN);
    const size_t count = (size_t)tokens * AGGREGATE_INPUT;
    uint16_t *staged = malloc(count * sizeof(*staged));
    if (!staged) { oops(r, "cannot allocate the aggregation input"); return; }
    for (size_t index = 0; index < count; index++) {
        const float value = round_bf16(normed[index] * rescale);
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        staged[index] = (uint16_t)(bits >> 16);
    }
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(r->gpu, staged, count);
    free(staged);
    if (!input) { oops(r, "cannot upload the aggregation input"); return; }

    char weight_name[224], bias_name[224];
    snprintf(weight_name, sizeof(weight_name),
             "text_embedding_projection.%s.weight", name);
    snprintf(bias_name, sizeof(bias_name),
             "text_embedding_projection.%s.bias", name);
    h3_gpu_tensor *weight = load_bf16(r, weight_name, 2, out_dim,
                                      AGGREGATE_INPUT);
    h3_gpu_tensor *bias = load_bf16(r, bias_name, 1, out_dim, 0);
    h3_gpu_tensor *result = NULL;
    if (!r->failed) {
        result = h3_gpu_tensor_new_bf16(r->gpu, (size_t)tokens * out_dim);
        if (!result) oops(r, "cannot allocate the aggregation output");
    }
    GPU_OP(r, h3_gpu_begin(r->gpu), "begin aggregation");
    GPU_OP(r, h3_gpu_linear_bf16(r->gpu, result, input, weight, bias, tokens,
                                 AGGREGATE_INPUT, out_dim), "aggregation");
    GPU_OP(r, h3_gpu_submit(r->gpu), "submit aggregation");
    read_bf16_as_f32(r, result, out, (size_t)tokens * out_dim);

    if (!r->failed)
        for (size_t index = 0; index < (size_t)tokens * out_dim; index++)
            if (!isfinite(out[index])) {
                oops(r, "an aggregated %s feature is not finite", name);
                break;
            }
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(bias);
    h3_gpu_tensor_free(result);
}

/* Each state normalized per token over its own 3840 channels, the 49 laid out
 * **layer-minor** -- normed[token][d * 49 + l] -- then rescaled and projected. */
static void aggregate(run *r, float *const *states, uint32_t tokens,
                      float *video, float *audio) {
    if (r->failed) return;
    const size_t count = (size_t)tokens * AGGREGATE_INPUT;
    float *normed = malloc(count * sizeof(*normed));
    if (!normed) { oops(r, "cannot allocate the normalized states"); return; }
    for (uint32_t token = 0; token < tokens; token++)
        for (int layer = 0; layer < HIDDEN_STATES; layer++) {
            const float *row = states[layer] + (size_t)token * HIDDEN;
            double sum = 0.0;
            for (int index = 0; index < HIDDEN; index++)
                sum += (double)row[index] * (double)row[index];
            const float inverse = 1.0f /
                sqrtf((float)(sum / (double)HIDDEN) + AGGREGATE_EPSILON);
            float *out = normed + (size_t)token * AGGREGATE_INPUT;
            for (int index = 0; index < HIDDEN; index++)
                out[(size_t)index * HIDDEN_STATES + layer] = row[index] * inverse;
        }
    aggregate_stream(r, normed, tokens, "video_aggregate_embed", VIDEO_DIM,
                     video);
    aggregate_stream(r, normed, tokens, "audio_aggregate_embed", AUDIO_DIM,
                     audio);
    free(normed);
}

/* ------------------------------------------------------------------ encode */

int ltx_text_encode(h3_gpu *gpu, const h3_weight_store *encoder,
                    const int32_t *ids, uint32_t tokens, float *video_features,
                    float *audio_features, ltx_text_tick tick,
                    void *tick_context, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!gpu || !encoder || !ids || !tokens || !video_features ||
        !audio_features) {
        if (error && error_size)
            snprintf(error, error_size, "ltx_text_encode wants a gpu, the "
                                        "encoder store, token ids and "
                                        "somewhere to put both streams");
        return 0;
    }
    run r = {0};
    r.gpu = gpu; r.store = encoder; r.error = error; r.error_size = error_size;

    resolve_all(&r);

    const size_t state = (size_t)tokens * HIDDEN;
    float **states = calloc(HIDDEN_STATES, sizeof(*states));
    if (!states) { oops(&r, "cannot allocate the hidden state table"); return 0; }
    for (int index = 0; index < HIDDEN_STATES && !r.failed; index++) {
        states[index] = malloc(state * sizeof(**states));
        if (!states[index]) oops(&r, "cannot allocate hidden state %d", index);
    }

    scratch space;
    scratch_create(&r, &space, tokens);
    h3_gpu_tensor *hidden = embed(&r, ids, tokens);
    read_bf16_as_f32(&r, hidden, states[0], state);

    for (int index = 0; index < LAYERS && !r.failed; index++) {
        layer_weights weights;
        load_layer(&r, index, &weights);
        GPU_OP(&r, h3_gpu_begin(gpu), "begin layer");
        run_layer(&r, &weights, &space, hidden, tokens);
        GPU_OP(&r, h3_gpu_submit(gpu), "submit layer");
        free_layer(&weights);
        /* State i + 1 is layer i's output, for every layer but the last: the
         * reference records the stream *entering* each layer, so layer 47's
         * raw output is never a state and only its normed form is. */
        if (index + 1 < LAYERS)
            read_bf16_as_f32(&r, hidden, states[index + 1], state);
        if (tick && !tick(index + 1, LAYERS, tick_context)) {
            r.cancelled = 1;
            break;
        }
    }
    if (!r.failed && !r.cancelled) {
        h3_gpu_tensor *final_norm = load_bf16(&r, "model.norm.weight", 1,
                                              HIDDEN, 0);
        h3_gpu_tensor *normed = NULL;
        if (!r.failed) {
            normed = h3_gpu_tensor_new_bf16(gpu, state);
            if (!normed) oops(&r, "cannot allocate the final normed state");
        }
        GPU_OP(&r, h3_gpu_begin(gpu), "begin final norm");
        GPU_OP(&r, h3_gpu_rms_norm_bf16(gpu, normed, hidden, final_norm, tokens,
                                        HIDDEN, RMS_EPSILON), "final norm");
        GPU_OP(&r, h3_gpu_submit(gpu), "submit final norm");
        read_bf16_as_f32(&r, normed, states[LAYERS], state);
        h3_gpu_tensor_free(final_norm);
        h3_gpu_tensor_free(normed);
    }
    if (!r.failed && !r.cancelled)
        aggregate(&r, states, tokens, video_features, audio_features);

    h3_gpu_tensor_free(hidden);
    scratch_free(&space);
    for (int index = 0; index < HIDDEN_STATES; index++) free(states[index]);
    free(states);
    return !r.failed && !r.cancelled;
}
