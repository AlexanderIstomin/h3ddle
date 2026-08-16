#include "qwen_gpu.h"

#include "qwen_block.h"
#include "qwen_predictor.h"
#include "qwen_talker.h"
#include "qwen_weights.h"

#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH    QWEN_TALKER_WIDTH
#define HEADS    QWEN_TALKER_HEADS
#define KV_HEADS QWEN_TALKER_KV_HEADS
#define HEAD_DIM QWEN_TALKER_HEAD_DIM
#define FFN      QWEN_TALKER_FFN
#define KV_WIDTH (KV_HEADS * HEAD_DIM)   /* 1024 */
#define Q_WIDTH  (HEADS * HEAD_DIM)      /* 2048 */
#define ROPE_HALF (HEAD_DIM / 2)         /* 64   */
#define RMS_EPSILON 1e-6f
#define ROPE_THETA  1000000.0
#define PREDICTOR_STEPS (QWEN_CODE_GROUPS - 1)

/* The talker and the code predictor are the same Qwen3 block in different
 * quantities — 28 layers against 5 — at identical width, head counts, head
 * dimension and rope theta. Everything from here to forward_block is shared. */

typedef struct {
    h3_gpu_tensor *input_layernorm, *post_attention_layernorm;
    h3_gpu_tensor *q_proj, *k_proj, *v_proj, *o_proj;
    h3_gpu_tensor *q_norm, *k_norm;
    h3_gpu_tensor *gate_proj, *up_proj, *down_proj;
    h3_gpu_tensor *keys, *values;   /* [max_tokens][KV_WIDTH] */
} qwen_gpu_layer;

typedef struct {
    h3_gpu_tensor *hidden, *normed, *branch;
    h3_gpu_tensor *query, *key, *value, *attention;
    h3_gpu_tensor *gate, *up;
    /* The full rotation table stays resident and a window of it is copied to
     * the front for each call, because h3_gpu_rope_text_bf16 indexes rows from
     * zero and a forward starts wherever the cache left off. */
    h3_gpu_tensor *rope_cos, *rope_sin;
    h3_gpu_tensor *rope_window_cos, *rope_window_sin;
    int max_tokens;
} qwen_gpu_workspace;

static int fail(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
    return 0;
}

static int fail_gpu(h3_gpu *gpu, char *error, size_t error_size,
                    const char *operation) {
    if (error && error_size)
        snprintf(error, error_size, "%s failed: %s", operation,
                 h3_gpu_error(gpu));
    return 0;
}

/* bf16 is the top 16 bits of an f32, so both directions are shifts. Rounding is
 * round-to-nearest-even, matching h3.c's own conversion. */
static uint16_t narrow(float value) {
    union { uint32_t bits; float number; } cast;
    cast.number = value;
    cast.bits += 0x7fffu + ((cast.bits >> 16) & 1u);
    return (uint16_t)(cast.bits >> 16);
}

static h3_gpu_tensor *upload(h3_gpu *gpu, const qwen_weights *weights,
                             const char *name, int ndim, const int64_t *shape,
                             size_t elements, char *error, size_t error_size) {
    const uint16_t *values =
        qwen_weights_bf16(weights, name, ndim, shape, error, error_size);
    if (!values) return NULL;
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(gpu, values, elements);
    if (!tensor)
        snprintf(error, error_size, "cannot upload %s: %s", name,
                 h3_gpu_error(gpu));
    return tensor;
}

static h3_gpu_tensor *upload_at(h3_gpu *gpu, const qwen_weights *weights,
                                const char *pattern, int index, int ndim,
                                const int64_t *shape, size_t elements,
                                char *error, size_t error_size) {
    const uint16_t *values = qwen_weights_bf16_at(weights, pattern, index, ndim,
                                                  shape, error, error_size);
    if (!values) return NULL;
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(gpu, values, elements);
    if (!tensor)
        snprintf(error, error_size, "cannot upload %s (layer %d): %s", pattern,
                 index, h3_gpu_error(gpu));
    return tensor;
}

#define UPLOAD_AT(target, pattern, index, elements, ndim, ...) do {             \
        const int64_t expected[] = __VA_ARGS__;                                \
        (target) = upload_at(gpu, weights, (pattern), (index), (ndim),          \
                             expected, (elements), error, error_size);          \
        if (!(target)) return 0;                                                \
    } while (0)

static int load_layer(h3_gpu *gpu, const qwen_weights *weights,
                      qwen_gpu_layer *layer, int index, int max_tokens,
                      char *error, size_t error_size) {
    UPLOAD_AT(layer->input_layernorm, "model.layers.%d.input_layernorm.weight",
              index, WIDTH, 1, {WIDTH});
    UPLOAD_AT(layer->post_attention_layernorm,
              "model.layers.%d.post_attention_layernorm.weight", index, WIDTH, 1,
              {WIDTH});
    UPLOAD_AT(layer->q_proj, "model.layers.%d.self_attn.q_proj.weight", index,
              (size_t)Q_WIDTH * WIDTH, 2, {Q_WIDTH, WIDTH});
    UPLOAD_AT(layer->k_proj, "model.layers.%d.self_attn.k_proj.weight", index,
              (size_t)KV_WIDTH * WIDTH, 2, {KV_WIDTH, WIDTH});
    UPLOAD_AT(layer->v_proj, "model.layers.%d.self_attn.v_proj.weight", index,
              (size_t)KV_WIDTH * WIDTH, 2, {KV_WIDTH, WIDTH});
    UPLOAD_AT(layer->o_proj, "model.layers.%d.self_attn.o_proj.weight", index,
              (size_t)WIDTH * Q_WIDTH, 2, {WIDTH, Q_WIDTH});
    UPLOAD_AT(layer->q_norm, "model.layers.%d.self_attn.q_norm.weight", index,
              HEAD_DIM, 1, {HEAD_DIM});
    UPLOAD_AT(layer->k_norm, "model.layers.%d.self_attn.k_norm.weight", index,
              HEAD_DIM, 1, {HEAD_DIM});
    UPLOAD_AT(layer->gate_proj, "model.layers.%d.mlp.gate_proj.weight", index,
              (size_t)FFN * WIDTH, 2, {FFN, WIDTH});
    UPLOAD_AT(layer->up_proj, "model.layers.%d.mlp.up_proj.weight", index,
              (size_t)FFN * WIDTH, 2, {FFN, WIDTH});
    UPLOAD_AT(layer->down_proj, "model.layers.%d.mlp.down_proj.weight", index,
              (size_t)WIDTH * FFN, 2, {WIDTH, FFN});

    const size_t cache = (size_t)max_tokens * KV_WIDTH;
    layer->keys = h3_gpu_tensor_new_bf16(gpu, cache);
    layer->values = h3_gpu_tensor_new_bf16(gpu, cache);
    if (!layer->keys || !layer->values)
        return fail(error, error_size, "cannot allocate the key/value cache");
    return 1;
}
#undef UPLOAD_AT

static void free_layer(qwen_gpu_layer *layer) {
    h3_gpu_tensor_free(layer->input_layernorm);
    h3_gpu_tensor_free(layer->post_attention_layernorm);
    h3_gpu_tensor_free(layer->q_proj);
    h3_gpu_tensor_free(layer->k_proj);
    h3_gpu_tensor_free(layer->v_proj);
    h3_gpu_tensor_free(layer->o_proj);
    h3_gpu_tensor_free(layer->q_norm);
    h3_gpu_tensor_free(layer->k_norm);
    h3_gpu_tensor_free(layer->gate_proj);
    h3_gpu_tensor_free(layer->up_proj);
    h3_gpu_tensor_free(layer->down_proj);
    h3_gpu_tensor_free(layer->keys);
    h3_gpu_tensor_free(layer->values);
    memset(layer, 0, sizeof(*layer));
}

/* The same table qwen_rope_init builds, formed in double so the float it stores
 * is right to the last bit, and left in f32 because h3_gpu_rope_text_bf16 reads
 * f32 tables — quantizing the angles to bf16 would compound into the rotation. */
static int workspace_init(qwen_gpu_workspace *work, h3_gpu *gpu, int max_tokens,
                          char *error, size_t error_size) {
    memset(work, 0, sizeof(*work));
    work->max_tokens = max_tokens;
    const size_t rows = (size_t)max_tokens;
    work->hidden = h3_gpu_tensor_new_bf16(gpu, rows * WIDTH);
    work->normed = h3_gpu_tensor_new_bf16(gpu, rows * WIDTH);
    work->branch = h3_gpu_tensor_new_bf16(gpu, rows * WIDTH);
    work->query = h3_gpu_tensor_new_bf16(gpu, rows * Q_WIDTH);
    work->key = h3_gpu_tensor_new_bf16(gpu, rows * KV_WIDTH);
    work->value = h3_gpu_tensor_new_bf16(gpu, rows * KV_WIDTH);
    work->attention = h3_gpu_tensor_new_bf16(gpu, rows * Q_WIDTH);
    work->gate = h3_gpu_tensor_new_bf16(gpu, rows * FFN);
    work->up = h3_gpu_tensor_new_bf16(gpu, rows * FFN);
    if (!work->hidden || !work->normed || !work->branch || !work->query ||
        !work->key || !work->value || !work->attention || !work->gate ||
        !work->up)
        return fail(error, error_size, "cannot allocate the working set");

    const size_t count = rows * ROPE_HALF;
    float *cosines = malloc(count * sizeof(float));
    float *sines = malloc(count * sizeof(float));
    if (!cosines || !sines) {
        free(cosines);
        free(sines);
        return fail(error, error_size, "out of memory");
    }
    for (int index = 0; index < ROPE_HALF; index++) {
        const double inverse =
            pow(ROPE_THETA, -(double)(2 * index) / (double)HEAD_DIM);
        for (int position = 0; position < max_tokens; position++) {
            const double angle = (double)position * inverse;
            cosines[(size_t)position * ROPE_HALF + index] = (float)cos(angle);
            sines[(size_t)position * ROPE_HALF + index] = (float)sin(angle);
        }
    }
    work->rope_cos = h3_gpu_tensor_from_f32(gpu, cosines, count);
    work->rope_sin = h3_gpu_tensor_from_f32(gpu, sines, count);
    work->rope_window_cos = h3_gpu_tensor_new_f32(gpu, count);
    work->rope_window_sin = h3_gpu_tensor_new_f32(gpu, count);
    free(cosines);
    free(sines);
    if (!work->rope_cos || !work->rope_sin || !work->rope_window_cos ||
        !work->rope_window_sin)
        return fail(error, error_size, "cannot allocate the rotation table");
    return 1;
}

static void workspace_release(qwen_gpu_workspace *work) {
    h3_gpu_tensor_free(work->hidden);
    h3_gpu_tensor_free(work->normed);
    h3_gpu_tensor_free(work->branch);
    h3_gpu_tensor_free(work->query);
    h3_gpu_tensor_free(work->key);
    h3_gpu_tensor_free(work->value);
    h3_gpu_tensor_free(work->attention);
    h3_gpu_tensor_free(work->gate);
    h3_gpu_tensor_free(work->up);
    h3_gpu_tensor_free(work->rope_cos);
    h3_gpu_tensor_free(work->rope_sin);
    h3_gpu_tensor_free(work->rope_window_cos);
    h3_gpu_tensor_free(work->rope_window_sin);
    memset(work, 0, sizeof(*work));
}

/* Rows base..base+tokens-1 of the rotation table, moved to the front, because
 * the kernel indexes it from zero. Once per forward, not once per layer. */
static int select_rope_window(h3_gpu *gpu, const qwen_gpu_workspace *work,
                              int tokens, int base, char *error,
                              size_t error_size) {
    const size_t offset = (size_t)base * ROPE_HALF;
    const size_t span = (size_t)tokens * ROPE_HALF;
    if (!h3_gpu_copy_f32(gpu, work->rope_window_cos, 0, work->rope_cos, offset,
                         span) ||
        !h3_gpu_copy_f32(gpu, work->rope_window_sin, 0, work->rope_sin, offset,
                         span))
        return fail_gpu(gpu, error, error_size, "rotation window");
    return 1;
}

/* One predictor step's arithmetic, from the hidden state to its logits: five
 * blocks, the last position, the final norm, and the head for this step. Shared
 * by the two run paths below, which differ only in how the code is chosen. */
static int predictor_step(struct qwen_gpu_predictor *predictor, int step,
                          int tokens, int base, char *error, size_t error_size);

static int forward_block(h3_gpu *gpu, const qwen_gpu_layer *layer,
                         const qwen_gpu_workspace *work, int tokens, int base,
                         char *error, size_t error_size) {
    const uint32_t rows = (uint32_t)tokens;
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);

#define OP(call, label) do {                                                    \
        if (!(call)) return fail_gpu(gpu, error, error_size, (label));          \
    } while (0)

    OP(h3_gpu_rms_norm_bf16(gpu, work->normed, work->hidden,
                            layer->input_layernorm, rows, WIDTH, RMS_EPSILON),
       "input RMSNorm");
    OP(h3_gpu_linear_bf16(gpu, work->query, work->normed, layer->q_proj, NULL,
                          rows, WIDTH, Q_WIDTH), "query projection");
    OP(h3_gpu_linear_bf16(gpu, work->key, work->normed, layer->k_proj, NULL,
                          rows, WIDTH, KV_WIDTH), "key projection");
    OP(h3_gpu_linear_bf16(gpu, work->value, work->normed, layer->v_proj, NULL,
                          rows, WIDTH, KV_WIDTH), "value projection");

    /* Per head, over head_dim alone, after projection and before the rotation. */
    OP(h3_gpu_head_rms_norm_coop_bf16(gpu, work->query, layer->q_norm, rows, HEADS,
                                 HEAD_DIM, RMS_EPSILON), "query head RMSNorm");
    OP(h3_gpu_head_rms_norm_coop_bf16(gpu, work->key, layer->k_norm, rows, KV_HEADS,
                                 HEAD_DIM, RMS_EPSILON), "key head RMSNorm");
    OP(h3_gpu_rope_text_bf16(gpu, work->query, work->key, work->rope_window_cos,
                             work->rope_window_sin, rows, HEADS, KV_HEADS,
                             HEAD_DIM), "RoPE");

    /* Cached keys are normalised and rotated, so a later step only has to dot
     * against them. */
    const size_t offset = (size_t)base * KV_WIDTH;
    const size_t span = (size_t)tokens * KV_WIDTH;
    OP(h3_gpu_copy_bf16(gpu, layer->keys, offset, work->key, 0, span),
       "key cache append");
    OP(h3_gpu_copy_bf16(gpu, layer->values, offset, work->value, 0, span),
       "value cache append");

    OP(h3_gpu_gqa_causal_cache_bf16(gpu, work->attention, work->query,
                                    layer->keys, layer->values, rows,
                                    (uint32_t)base, HEADS, KV_HEADS, HEAD_DIM,
                                    scale), "grouped-query attention");
    OP(h3_gpu_linear_bf16(gpu, work->branch, work->attention, layer->o_proj,
                          NULL, rows, Q_WIDTH, WIDTH),
       "attention output projection");
    OP(h3_gpu_add_bf16(gpu, work->hidden, work->hidden, work->branch,
                       rows * WIDTH), "attention residual");

    OP(h3_gpu_rms_norm_bf16(gpu, work->normed, work->hidden,
                            layer->post_attention_layernorm, rows, WIDTH,
                            RMS_EPSILON), "post-attention RMSNorm");
    OP(h3_gpu_linear_bf16(gpu, work->gate, work->normed, layer->gate_proj, NULL,
                          rows, WIDTH, FFN), "MLP gate");
    OP(h3_gpu_linear_bf16(gpu, work->up, work->normed, layer->up_proj, NULL,
                          rows, WIDTH, FFN), "MLP up");
    OP(h3_gpu_silu_mul_bf16(gpu, work->gate, work->gate, work->up, rows * FFN),
       "SwiGLU");
    OP(h3_gpu_linear_bf16(gpu, work->branch, work->gate, layer->down_proj, NULL,
                          rows, FFN, WIDTH), "MLP down");
    OP(h3_gpu_add_bf16(gpu, work->hidden, work->hidden, work->branch,
                       rows * WIDTH), "MLP residual");
#undef OP
    return 1;
}

/* ---------------------------------------------------------------- talker -- */

struct qwen_gpu_talker {
    h3_gpu *gpu;
    qwen_weights *weights;
    qwen_gpu_layer layers[QWEN_TALKER_LAYERS];
    h3_gpu_tensor *final_norm, *codec_head, *logits;
    qwen_gpu_workspace work;
    uint16_t *staging;   /* f32 <-> bf16 at the boundary */
    int max_tokens;
    int cached;
};

qwen_gpu_talker *qwen_gpu_talker_load(const char *path, const char *shader_path,
                                      int max_tokens, char *error,
                                      size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (max_tokens < 1) max_tokens = 1;

    qwen_gpu_talker *talker = calloc(1, sizeof(*talker));
    if (!talker) {
        fail(error, error_size, "out of memory");
        return NULL;
    }
    talker->max_tokens = max_tokens;
    talker->gpu = h3_gpu_create(shader_path, error, error_size);
    if (!talker->gpu) {
        free(talker);
        return NULL;
    }
    talker->weights = qwen_weights_open(path, error, error_size);
    if (!talker->weights) {
        qwen_gpu_talker_free(talker);
        return NULL;
    }

    h3_gpu *gpu = talker->gpu;
    const qwen_weights *weights = talker->weights;
    talker->final_norm = upload(gpu, weights, "model.norm.weight", 1,
                                (const int64_t[]){WIDTH}, WIDTH, error,
                                error_size);
    talker->codec_head = upload(gpu, weights, "codec_head.weight", 2,
                                (const int64_t[]){QWEN_TALKER_VOCAB, WIDTH},
                                (size_t)QWEN_TALKER_VOCAB * WIDTH, error,
                                error_size);
    if (!talker->final_norm || !talker->codec_head) {
        qwen_gpu_talker_free(talker);
        return NULL;
    }
    for (int index = 0; index < QWEN_TALKER_LAYERS; index++) {
        if (!load_layer(gpu, weights, &talker->layers[index], index, max_tokens,
                        error, error_size)) {
            qwen_gpu_talker_free(talker);
            return NULL;
        }
    }

    talker->logits =
        h3_gpu_tensor_new_bf16(gpu, (size_t)max_tokens * QWEN_TALKER_VOCAB);
    talker->staging =
        malloc((size_t)max_tokens * QWEN_TALKER_VOCAB * sizeof(uint16_t));
    if (!talker->logits || !talker->staging ||
        !workspace_init(&talker->work, gpu, max_tokens, error, error_size)) {
        if (!error[0]) fail(error, error_size, "cannot allocate the talker");
        qwen_gpu_talker_free(talker);
        return NULL;
    }
    return talker;
}

void qwen_gpu_talker_free(qwen_gpu_talker *talker) {
    if (!talker) return;
    for (int index = 0; index < QWEN_TALKER_LAYERS; index++)
        free_layer(&talker->layers[index]);
    workspace_release(&talker->work);
    h3_gpu_tensor_free(talker->final_norm);
    h3_gpu_tensor_free(talker->codec_head);
    h3_gpu_tensor_free(talker->logits);
    free(talker->staging);
    qwen_weights_close(talker->weights);
    if (talker->gpu) h3_gpu_free(talker->gpu);
    free(talker);
}

void qwen_gpu_talker_reset(qwen_gpu_talker *talker) {
    if (talker) talker->cached = 0;
}

int qwen_gpu_talker_cached(const qwen_gpu_talker *talker) {
    return talker ? talker->cached : 0;
}

int qwen_gpu_talker_stats(const qwen_gpu_talker *talker, uint64_t *dispatches,
                          uint64_t *mps_dispatches, uint64_t *submissions,
                          double *encode_seconds, double *gpu_seconds) {
    h3_gpu_stats stats;
    if (!talker || !h3_gpu_get_stats(talker->gpu, &stats)) return 0;
    if (dispatches) *dispatches = stats.direct_dispatches;
    if (mps_dispatches) *mps_dispatches = stats.mps_linear_dispatches;
    if (submissions) *submissions = stats.submissions;
    if (encode_seconds) *encode_seconds = stats.command_encode_seconds;
    if (gpu_seconds) *gpu_seconds = stats.gpu_seconds;
    return 1;
}

int qwen_gpu_talker_forward(qwen_gpu_talker *talker, const float *embeddings,
                            int tokens, float *hidden, float *logits,
                            char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!talker || !embeddings || tokens < 1)
        return fail(error, error_size, "invalid arguments");
    if (talker->cached + tokens > talker->max_tokens) {
        if (error && error_size)
            snprintf(error, error_size,
                     "the utterance needs %d tokens but the cache holds %d",
                     talker->cached + tokens, talker->max_tokens);
        return 0;
    }

    h3_gpu *gpu = talker->gpu;
    const qwen_gpu_workspace *work = &talker->work;
    const int base = talker->cached;
    const uint32_t rows = (uint32_t)tokens;

    for (size_t index = 0; index < (size_t)tokens * WIDTH; index++)
        talker->staging[index] = narrow(embeddings[index]);
    if (!h3_gpu_tensor_write_bf16(work->hidden, talker->staging,
                                  (size_t)tokens * WIDTH))
        return fail(error, error_size, "cannot upload the talker's input");

    if (!h3_gpu_begin(gpu))
        return fail_gpu(gpu, error, error_size, "command stream");
    if (!select_rope_window(gpu, work, tokens, base, error, error_size))
        return 0;
    for (int index = 0; index < QWEN_TALKER_LAYERS; index++) {
        if (!forward_block(gpu, &talker->layers[index], work, tokens, base,
                           error, error_size))
            return 0;
    }
    if (hidden || logits) {
        if (!h3_gpu_rms_norm_bf16(gpu, work->normed, work->hidden,
                                  talker->final_norm, rows, WIDTH, RMS_EPSILON))
            return fail_gpu(gpu, error, error_size, "final RMSNorm");
        if (logits &&
            !h3_gpu_linear_bf16(gpu, talker->logits, work->normed,
                                talker->codec_head, NULL, rows, WIDTH,
                                QWEN_TALKER_VOCAB))
            return fail_gpu(gpu, error, error_size, "codec head");
    }
    if (!h3_gpu_submit(gpu))
        return fail_gpu(gpu, error, error_size, "submit");
    talker->cached = base + tokens;

    if (hidden) {
        const size_t count = (size_t)tokens * WIDTH;
        if (!h3_gpu_tensor_read_bf16(work->normed, talker->staging, count))
            return fail(error, error_size, "cannot read the talker's state");
        for (size_t index = 0; index < count; index++)
            hidden[index] = qwen_widen(talker->staging[index]);
    }
    if (logits) {
        const size_t count = (size_t)tokens * QWEN_TALKER_VOCAB;
        if (!h3_gpu_tensor_read_bf16(talker->logits, talker->staging, count))
            return fail(error, error_size, "cannot read the talker's logits");
        for (size_t index = 0; index < count; index++)
            logits[index] = qwen_widen(talker->staging[index]);
    }
    return 1;
}

/* ------------------------------------------------------------- predictor -- */

struct qwen_gpu_predictor {
    h3_gpu *gpu;
    qwen_weights *weights;
    int owns_gpu;
    qwen_gpu_layer layers[QWEN_PREDICTOR_LAYERS];
    h3_gpu_tensor *embeddings[PREDICTOR_STEPS];  /* codec_embedding[i]: group i+1 */
    h3_gpu_tensor *heads[PREDICTOR_STEPS];       /* lm_head[i]:         group i+1 */
    h3_gpu_tensor *final_norm, *logits;
    /* Slot 0 is the code the next step embeds; slots 1..15 are the frame's
     * codes, read back once at the end. Two writes rather than one because
     * h3_gpu_embedding_bf16 reads its id from index zero. */
    h3_gpu_tensor *code_slots;
    qwen_gpu_workspace work;
    uint16_t *staging;
    float *host_logits;
    qwen_gpu_predictor_probe probe;
    void *probe_opaque;
};

void qwen_gpu_predictor_set_probe(qwen_gpu_predictor *predictor,
                                  qwen_gpu_predictor_probe probe,
                                  void *opaque) {
    if (!predictor) return;
    predictor->probe = probe;
    predictor->probe_opaque = opaque;
}

qwen_gpu_predictor *qwen_gpu_predictor_load(const char *path,
                                            const char *shader_path,
                                            char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    qwen_gpu_predictor *predictor = calloc(1, sizeof(*predictor));
    if (!predictor) {
        fail(error, error_size, "out of memory");
        return NULL;
    }
    predictor->gpu = h3_gpu_create(shader_path, error, error_size);
    if (!predictor->gpu) {
        free(predictor);
        return NULL;
    }
    predictor->owns_gpu = 1;
    predictor->weights = qwen_weights_open(path, error, error_size);
    if (!predictor->weights) {
        qwen_gpu_predictor_free(predictor);
        return NULL;
    }

    h3_gpu *gpu = predictor->gpu;
    const qwen_weights *weights = predictor->weights;
    predictor->final_norm = upload(gpu, weights, "model.norm.weight", 1,
                                   (const int64_t[]){WIDTH}, WIDTH, error,
                                   error_size);
    if (!predictor->final_norm) {
        qwen_gpu_predictor_free(predictor);
        return NULL;
    }
    for (int index = 0; index < QWEN_PREDICTOR_LAYERS; index++) {
        if (!load_layer(gpu, weights, &predictor->layers[index], index,
                        QWEN_CODE_GROUPS, error, error_size)) {
            qwen_gpu_predictor_free(predictor);
            return NULL;
        }
    }
    for (int index = 0; index < PREDICTOR_STEPS; index++) {
        const int64_t shape[] = {QWEN_PREDICTOR_VOCAB, WIDTH};
        const size_t elements = (size_t)QWEN_PREDICTOR_VOCAB * WIDTH;
        predictor->embeddings[index] =
            upload_at(gpu, weights, "model.codec_embedding.%d.weight", index, 2,
                      shape, elements, error, error_size);
        predictor->heads[index] = upload_at(gpu, weights, "lm_head.%d.weight",
                                            index, 2, shape, elements, error,
                                            error_size);
        if (!predictor->embeddings[index] || !predictor->heads[index]) {
            qwen_gpu_predictor_free(predictor);
            return NULL;
        }
    }

    predictor->logits = h3_gpu_tensor_new_bf16(gpu, QWEN_PREDICTOR_VOCAB);
    predictor->code_slots = h3_gpu_tensor_new_u32(gpu, QWEN_CODE_GROUPS);
    predictor->staging =
        malloc((size_t)QWEN_CODE_GROUPS * WIDTH * sizeof(uint16_t));
    predictor->host_logits = malloc(QWEN_PREDICTOR_VOCAB * sizeof(float));
    if (!predictor->logits || !predictor->code_slots || !predictor->staging ||
        !predictor->host_logits ||
        !workspace_init(&predictor->work, gpu, QWEN_CODE_GROUPS, error,
                        error_size)) {
        if (!error[0]) fail(error, error_size, "cannot allocate the predictor");
        qwen_gpu_predictor_free(predictor);
        return NULL;
    }
    return predictor;
}

void qwen_gpu_predictor_free(qwen_gpu_predictor *predictor) {
    if (!predictor) return;
    for (int index = 0; index < QWEN_PREDICTOR_LAYERS; index++)
        free_layer(&predictor->layers[index]);
    for (int index = 0; index < PREDICTOR_STEPS; index++) {
        h3_gpu_tensor_free(predictor->embeddings[index]);
        h3_gpu_tensor_free(predictor->heads[index]);
    }
    workspace_release(&predictor->work);
    h3_gpu_tensor_free(predictor->final_norm);
    h3_gpu_tensor_free(predictor->logits);
    h3_gpu_tensor_free(predictor->code_slots);
    free(predictor->staging);
    free(predictor->host_logits);
    qwen_weights_close(predictor->weights);
    if (predictor->owns_gpu && predictor->gpu) h3_gpu_free(predictor->gpu);
    free(predictor);
}

/* xorshift, so a run is reproducible from its seed without pulling in a
 * generator whose sequence could change under us. Kept identical to the CPU
 * predictor's, so the same seed walks the same path on either. */
static float next_uniform(uint64_t *state) {
    uint64_t value = *state ? *state : 0x9E3779B97F4A7C15ull;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    return (float)((value >> 11) * (1.0 / 9007199254740992.0));
}

static uint32_t choose(const float *logits, int vocab, float temperature,
                       uint64_t *seed) {
    int best = 0;
    for (int index = 1; index < vocab; index++)
        if (logits[index] > logits[best]) best = index;
    if (temperature <= 0.0f || !seed) return (uint32_t)best;

    const float highest = logits[best];
    float sum = 0.0f;
    for (int index = 0; index < vocab; index++)
        sum += expf((logits[index] - highest) / temperature);
    float target = next_uniform(seed) * sum, running = 0.0f;
    for (int index = 0; index < vocab; index++) {
        running += expf((logits[index] - highest) / temperature);
        if (running >= target) return (uint32_t)index;
    }
    return (uint32_t)best;
}

static int predictor_step(struct qwen_gpu_predictor *predictor, int step,
                          int tokens, int base, char *error,
                          size_t error_size) {
    h3_gpu *gpu = predictor->gpu;
    const qwen_gpu_workspace *work = &predictor->work;
    if (!select_rope_window(gpu, work, tokens, base, error, error_size))
        return 0;
    for (int index = 0; index < QWEN_PREDICTOR_LAYERS; index++) {
        if (!forward_block(gpu, &predictor->layers[index], work, tokens, base,
                           error, error_size))
            return 0;
    }
    /* Only the last position carries the state the head reads. */
    if (!h3_gpu_copy_bf16(gpu, work->branch, 0, work->hidden,
                          (size_t)(tokens - 1) * WIDTH, WIDTH))
        return fail_gpu(gpu, error, error_size, "select last position");
    if (!h3_gpu_rms_norm_bf16(gpu, work->normed, work->branch,
                              predictor->final_norm, 1, WIDTH, RMS_EPSILON))
        return fail_gpu(gpu, error, error_size, "final RMSNorm");
    if (!h3_gpu_linear_bf16(gpu, predictor->logits, work->normed,
                            predictor->heads[step], NULL, 1, WIDTH,
                            QWEN_PREDICTOR_VOCAB))
        return fail_gpu(gpu, error, error_size, "code head");
    return 1;
}

/* The whole frame in one command buffer.
 *
 * Nothing needs to reach the host between steps: h3_gpu_argmax_bf16 writes the
 * chosen code where h3_gpu_embedding_bf16 reads it, so fifteen round trips
 * become one. That is the difference that matters here — the arithmetic is
 * about 1 ms a step and a round trip costs rather more. */
static int run_on_device(qwen_gpu_predictor *predictor, uint32_t *codes,
                         char *error, size_t error_size) {
    h3_gpu *gpu = predictor->gpu;
    const qwen_gpu_workspace *work = &predictor->work;
    if (!h3_gpu_begin(gpu))
        return fail_gpu(gpu, error, error_size, "command stream");

    int tokens = 2, base = 0;
    for (int step = 0; step < PREDICTOR_STEPS; step++) {
        /* codec_embedding[i] embeds what lm_head[i] produced, so the step that
         * consumes it is the one after. */
        if (step && !h3_gpu_embedding_bf16(gpu, work->hidden,
                                           predictor->embeddings[step - 1],
                                           predictor->code_slots, 1,
                                           QWEN_PREDICTOR_VOCAB, WIDTH))
            return fail_gpu(gpu, error, error_size, "embed the produced code");
        if (!predictor_step(predictor, step, tokens, base, error, error_size))
            return 0;
        if (!h3_gpu_argmax_bf16(gpu, predictor->code_slots, predictor->logits,
                                1, QWEN_PREDICTOR_VOCAB, 0) ||
            !h3_gpu_argmax_bf16(gpu, predictor->code_slots, predictor->logits,
                                1, QWEN_PREDICTOR_VOCAB, (uint32_t)(1 + step)))
            return fail_gpu(gpu, error, error_size, "choose the code");
        base += tokens;
        tokens = 1;
    }
    if (!h3_gpu_submit(gpu))
        return fail_gpu(gpu, error, error_size, "submit");

    uint32_t slots[QWEN_CODE_GROUPS];
    if (!h3_gpu_tensor_read_u32(predictor->code_slots, slots, QWEN_CODE_GROUPS))
        return fail(error, error_size, "cannot read the produced codes");
    for (int step = 0; step < PREDICTOR_STEPS; step++) {
        if (slots[1 + step] >= QWEN_PREDICTOR_VOCAB)
            return fail(error, error_size, "the device produced an invalid code");
        codes[step] = slots[1 + step];
    }
    return 1;
}

/* Sampling, or reporting logits to a probe, needs them on the host, so this
 * pays one round trip a step. Each step's embedding copy is still encoded at
 * the head of the next step's buffer rather than submitted alone, since only
 * the readback genuinely needs the device to finish. */
static int run_through_host(qwen_gpu_predictor *predictor, uint32_t *codes,
                            float temperature, uint64_t *seed, char *error,
                            size_t error_size) {
    h3_gpu *gpu = predictor->gpu;
    const qwen_gpu_workspace *work = &predictor->work;
    int tokens = 2, base = 0, pending_code = -1, pending_step = -1;

    for (int step = 0; step < PREDICTOR_STEPS; step++) {
        if (!h3_gpu_begin(gpu))
            return fail_gpu(gpu, error, error_size, "command stream");
        if (pending_code >= 0 &&
            !h3_gpu_copy_bf16(gpu, work->hidden, 0,
                              predictor->embeddings[pending_step],
                              (size_t)pending_code * WIDTH, WIDTH))
            return fail_gpu(gpu, error, error_size, "embed the produced code");
        if (!predictor_step(predictor, step, tokens, base, error, error_size))
            return 0;
        if (!h3_gpu_submit(gpu))
            return fail_gpu(gpu, error, error_size, "submit");

        if (!h3_gpu_tensor_read_bf16(predictor->logits, predictor->staging,
                                     QWEN_PREDICTOR_VOCAB))
            return fail(error, error_size, "cannot read the predictor's logits");
        for (int index = 0; index < QWEN_PREDICTOR_VOCAB; index++)
            predictor->host_logits[index] = qwen_widen(predictor->staging[index]);
        if (predictor->probe)
            predictor->probe(step, predictor->host_logits,
                             QWEN_PREDICTOR_VOCAB, predictor->probe_opaque);

        const uint32_t code = choose(predictor->host_logits,
                                     QWEN_PREDICTOR_VOCAB, temperature, seed);
        codes[step] = code;
        base += tokens;
        tokens = 1;
        pending_code = (int)code;
        pending_step = step;
    }
    return 1;
}

int qwen_gpu_predictor_run(qwen_gpu_predictor *predictor, const float *hidden,
                           const float *group0, uint32_t *codes,
                           float temperature, uint64_t *seed, char *error,
                           size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!predictor || !hidden || !group0 || !codes)
        return fail(error, error_size, "invalid arguments");

    /* The prefill is two positions — the talker's state and the group-0 code's
     * embedding — and every step after it adds one. */
    for (int channel = 0; channel < WIDTH; channel++) {
        predictor->staging[channel] = narrow(hidden[channel]);
        predictor->staging[WIDTH + channel] = narrow(group0[channel]);
    }
    if (!h3_gpu_tensor_write_bf16(predictor->work.hidden, predictor->staging,
                                  (size_t)2 * WIDTH))
        return fail(error, error_size, "cannot upload the predictor's input");

    /* A probe wants the logits of every step, which only the host path has. */
    if (temperature <= 0.0f && !predictor->probe)
        return run_on_device(predictor, codes, error, error_size);
    return run_through_host(predictor, codes, temperature, seed, error,
                            error_size);
}

int qwen_gpu_predictor_stats(const qwen_gpu_predictor *predictor,
                             uint64_t *dispatches, uint64_t *submissions,
                             double *encode_seconds, double *gpu_seconds) {
    h3_gpu_stats stats;
    if (!predictor || !h3_gpu_get_stats(predictor->gpu, &stats)) return 0;
    if (dispatches) *dispatches = stats.direct_dispatches;
    if (submissions) *submissions = stats.submissions;
    if (encode_seconds) *encode_seconds = stats.command_encode_seconds;
    if (gpu_seconds) *gpu_seconds = stats.gpu_seconds;
    return 1;
}
