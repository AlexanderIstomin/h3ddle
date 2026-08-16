#include "qwen_predictor.h"

#include "qwen_block.h"
#include "qwen_talker.h"
#include "qwen_weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH    QWEN_TALKER_WIDTH
#define STEPS    (QWEN_CODE_GROUPS - 1)
#define KV_WIDTH (QWEN_TALKER_KV_HEADS * QWEN_TALKER_HEAD_DIM)

struct qwen_predictor {
    qwen_weights *weights;
    qwen_block_config config;
    qwen_block_weights layers[QWEN_PREDICTOR_LAYERS];
    const uint16_t *embeddings[STEPS];   /* codec_embedding[i]: group i+1 */
    const uint16_t *heads[STEPS];        /* lm_head[i]:         group i+1 */
    const uint16_t *final_norm;

    qwen_scratch scratch;
    qwen_rope rope;
    float *sequence;   /* [QWEN_CODE_GROUPS][WIDTH] */
    float *working;
    float *normed;
    float *logits;
    float *keys, *values;

    qwen_predictor_probe probe;
    void *probe_opaque;
};

qwen_predictor *qwen_predictor_load(const char *path, char *error,
                                    size_t error_size) {
    if (error && error_size) error[0] = '\0';
    qwen_predictor *predictor = calloc(1, sizeof(*predictor));
    if (!predictor) {
        snprintf(error, error_size, "out of memory");
        return NULL;
    }
    predictor->weights = qwen_weights_open(path, error, error_size);
    if (!predictor->weights) {
        free(predictor);
        return NULL;
    }
    predictor->config = (qwen_block_config){
        .width = WIDTH,
        .heads = QWEN_TALKER_HEADS,
        .kv_heads = QWEN_TALKER_KV_HEADS,
        .head_dim = QWEN_TALKER_HEAD_DIM,
        .ffn = QWEN_TALKER_FFN,
        .rope_theta = 1000000.0f,
        .rms_epsilon = 1e-6f,
    };

    const qwen_weights *w = predictor->weights;
#define REQUIRE_AT(target, pattern, index, ndim, ...) do {                     \
        const int64_t expected[] = __VA_ARGS__;                                \
        (target) = qwen_weights_bf16_at(w, (pattern), (index), (ndim),         \
                                        expected, error, error_size);          \
        if (!(target)) { qwen_predictor_free(predictor); return NULL; }        \
    } while (0)

    for (int index = 0; index < QWEN_PREDICTOR_LAYERS; index++) {
        qwen_block_weights *layer = &predictor->layers[index];
        REQUIRE_AT(layer->input_layernorm,
                   "model.layers.%d.input_layernorm.weight", index, 1, {WIDTH});
        REQUIRE_AT(layer->post_attention_layernorm,
                   "model.layers.%d.post_attention_layernorm.weight", index, 1,
                   {WIDTH});
        REQUIRE_AT(layer->q_proj, "model.layers.%d.self_attn.q_proj.weight",
                   index, 2, {QWEN_TALKER_HEADS * QWEN_TALKER_HEAD_DIM, WIDTH});
        REQUIRE_AT(layer->k_proj, "model.layers.%d.self_attn.k_proj.weight",
                   index, 2, {KV_WIDTH, WIDTH});
        REQUIRE_AT(layer->v_proj, "model.layers.%d.self_attn.v_proj.weight",
                   index, 2, {KV_WIDTH, WIDTH});
        REQUIRE_AT(layer->o_proj, "model.layers.%d.self_attn.o_proj.weight",
                   index, 2, {WIDTH, QWEN_TALKER_HEADS * QWEN_TALKER_HEAD_DIM});
        REQUIRE_AT(layer->q_norm, "model.layers.%d.self_attn.q_norm.weight",
                   index, 1, {QWEN_TALKER_HEAD_DIM});
        REQUIRE_AT(layer->k_norm, "model.layers.%d.self_attn.k_norm.weight",
                   index, 1, {QWEN_TALKER_HEAD_DIM});
        REQUIRE_AT(layer->gate_proj, "model.layers.%d.mlp.gate_proj.weight",
                   index, 2, {QWEN_TALKER_FFN, WIDTH});
        REQUIRE_AT(layer->up_proj, "model.layers.%d.mlp.up_proj.weight",
                   index, 2, {QWEN_TALKER_FFN, WIDTH});
        REQUIRE_AT(layer->down_proj, "model.layers.%d.mlp.down_proj.weight",
                   index, 2, {WIDTH, QWEN_TALKER_FFN});
    }
    for (int index = 0; index < STEPS; index++) {
        REQUIRE_AT(predictor->embeddings[index],
                   "model.codec_embedding.%d.weight", index, 2,
                   {QWEN_PREDICTOR_VOCAB, WIDTH});
        REQUIRE_AT(predictor->heads[index], "lm_head.%d.weight", index, 2,
                   {QWEN_PREDICTOR_VOCAB, WIDTH});
    }
#undef REQUIRE_AT
    predictor->final_norm = qwen_weights_bf16(w, "model.norm.weight", 1,
                                              (const int64_t[]){WIDTH},
                                              error, error_size);
    if (!predictor->final_norm) {
        qwen_predictor_free(predictor);
        return NULL;
    }

    if (!qwen_scratch_init(&predictor->scratch, &predictor->config,
                           QWEN_CODE_GROUPS) ||
        !qwen_rope_init(&predictor->rope, &predictor->config,
                        QWEN_CODE_GROUPS)) {
        snprintf(error, error_size, "out of memory");
        qwen_predictor_free(predictor);
        return NULL;
    }
    predictor->sequence = calloc((size_t)QWEN_CODE_GROUPS * WIDTH, sizeof(float));
    predictor->working = calloc((size_t)QWEN_CODE_GROUPS * WIDTH, sizeof(float));
    predictor->normed = calloc((size_t)QWEN_CODE_GROUPS * WIDTH, sizeof(float));
    predictor->logits = calloc((size_t)QWEN_CODE_GROUPS * QWEN_PREDICTOR_VOCAB,
                               sizeof(float));
    predictor->keys = calloc((size_t)QWEN_PREDICTOR_LAYERS * QWEN_CODE_GROUPS *
                             KV_WIDTH, sizeof(float));
    predictor->values = calloc((size_t)QWEN_PREDICTOR_LAYERS * QWEN_CODE_GROUPS *
                               KV_WIDTH, sizeof(float));
    if (!predictor->sequence || !predictor->working || !predictor->normed ||
        !predictor->logits || !predictor->keys || !predictor->values) {
        snprintf(error, error_size, "out of memory");
        qwen_predictor_free(predictor);
        return NULL;
    }
    return predictor;
}

void qwen_predictor_free(qwen_predictor *predictor) {
    if (!predictor) return;
    qwen_scratch_release(&predictor->scratch);
    qwen_rope_release(&predictor->rope);
    free(predictor->sequence);
    free(predictor->working);
    free(predictor->normed);
    free(predictor->logits);
    free(predictor->keys);
    free(predictor->values);
    qwen_weights_close(predictor->weights);
    free(predictor);
}

void qwen_predictor_set_probe(qwen_predictor *predictor,
                              qwen_predictor_probe probe, void *opaque) {
    if (!predictor) return;
    predictor->probe = probe;
    predictor->probe_opaque = opaque;
}

/* xorshift, so a run is reproducible from its seed without pulling in a
 * generator whose sequence could change under us. */
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

int qwen_predictor_run(qwen_predictor *predictor, const float *hidden,
                       const float *group0, uint32_t *codes,
                       float temperature, uint64_t *seed,
                       char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!predictor || !hidden || !group0 || !codes) {
        snprintf(error, error_size, "invalid arguments");
        return 0;
    }

    /* The prefill is two positions — the talker's state and the group-0 code's
     * embedding — and every step after it adds one. The keys and values are
     * kept, so a step costs one position rather than replaying the sequence:
     * cached, a frame is 16 position-passes; recomputed it is 135, and since
     * the predictor is about two thirds of generation time that difference is
     * most of the run. */
    memset(predictor->keys, 0,
           (size_t)QWEN_PREDICTOR_LAYERS * QWEN_CODE_GROUPS * KV_WIDTH *
           sizeof(float));
    memset(predictor->values, 0,
           (size_t)QWEN_PREDICTOR_LAYERS * QWEN_CODE_GROUPS * KV_WIDTH *
           sizeof(float));
    memcpy(predictor->working, hidden, WIDTH * sizeof(float));
    memcpy(predictor->working + WIDTH, group0, WIDTH * sizeof(float));
    int tokens = 2, base = 0;

    for (int step = 0; step < STEPS; step++) {
        for (int layer = 0; layer < QWEN_PREDICTOR_LAYERS; layer++) {
            float *keys = predictor->keys +
                (size_t)layer * QWEN_CODE_GROUPS * KV_WIDTH;
            float *values = predictor->values +
                (size_t)layer * QWEN_CODE_GROUPS * KV_WIDTH;
            qwen_block_forward(&predictor->config, &predictor->layers[layer],
                               &predictor->rope, predictor->working, tokens,
                               base, keys, values, QWEN_CODE_GROUPS,
                               &predictor->scratch);
        }
        const float *last = predictor->working + (size_t)(tokens - 1) * WIDTH;
        qwen_rms_norm(last, predictor->final_norm, predictor->normed, WIDTH, 1,
                      predictor->config.rms_epsilon);
        qwen_matmul(predictor->heads[step], predictor->normed,
                    predictor->logits, WIDTH, QWEN_PREDICTOR_VOCAB, 1);
        if (predictor->probe)
            predictor->probe(step, predictor->logits, QWEN_PREDICTOR_VOCAB,
                             predictor->probe_opaque);

        const uint32_t code = choose(predictor->logits, QWEN_PREDICTOR_VOCAB,
                                     temperature, seed);
        codes[step] = code;

        /* the next position is what this head produced, embedded by the table
         * that shares its index */
        base += tokens;
        tokens = 1;
        const uint16_t *row =
            predictor->embeddings[step] + (size_t)code * WIDTH;
        for (int channel = 0; channel < WIDTH; channel++)
            predictor->working[channel] = qwen_widen(row[channel]);
    }
    return 1;
}

int qwen_predictor_accumulate(const qwen_predictor *predictor,
                              const uint32_t *codes, float *out,
                              char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!predictor || !codes || !out) {
        snprintf(error, error_size, "invalid arguments");
        return 0;
    }
    for (int step = 0; step < STEPS; step++) {
        if (codes[step] >= QWEN_PREDICTOR_VOCAB) {
            snprintf(error, error_size, "code %u is out of range for group %d",
                     codes[step], step + 1);
            return 0;
        }
        const uint16_t *row =
            predictor->embeddings[step] + (size_t)codes[step] * WIDTH;
        for (int channel = 0; channel < WIDTH; channel++)
            out[channel] += qwen_widen(row[channel]);
    }
    return 1;
}
