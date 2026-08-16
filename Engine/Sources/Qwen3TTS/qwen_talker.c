#include "qwen_talker.h"

#include "qwen_block.h"
#include "qwen_weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KV_WIDTH (QWEN_TALKER_KV_HEADS * QWEN_TALKER_HEAD_DIM)  /* 1024 */
#define Q_WIDTH  (QWEN_TALKER_HEADS * QWEN_TALKER_HEAD_DIM)     /* 2048 */

struct qwen_talker {
    qwen_weights *weights;
    qwen_block_config config;
    qwen_block_weights layers[QWEN_TALKER_LAYERS];
    const uint16_t *text_embedding;      /* [text vocab, 2048] */
    const uint16_t *codec_embedding;     /* [3072, 1024]       */
    const uint16_t *projection_1_weight; /* [2048, 2048]       */
    const uint16_t *projection_1_bias;
    const uint16_t *projection_2_weight; /* [1024, 2048]       */
    const uint16_t *projection_2_bias;
    const uint16_t *final_norm;
    const uint16_t *codec_head;          /* [3072, 1024]       */

    int max_tokens;
    int cached;
    float *keys;    /* [layers][max_tokens][KV_WIDTH] */
    float *values;
    float *hidden;
    float *normed;
    qwen_scratch scratch;
    qwen_rope rope;

    qwen_talker_probe probe;
    void *probe_opaque;
};

qwen_talker *qwen_talker_load(const char *path, int max_tokens,
                              char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (max_tokens < 1) max_tokens = 1;

    qwen_talker *talker = calloc(1, sizeof(*talker));
    if (!talker) {
        snprintf(error, error_size, "out of memory");
        return NULL;
    }
    talker->max_tokens = max_tokens;
    talker->config = (qwen_block_config){
        .width = QWEN_TALKER_WIDTH,
        .heads = QWEN_TALKER_HEADS,
        .kv_heads = QWEN_TALKER_KV_HEADS,
        .head_dim = QWEN_TALKER_HEAD_DIM,
        .ffn = QWEN_TALKER_FFN,
        .rope_theta = 1000000.0f,
        .rms_epsilon = 1e-6f,
    };
    talker->weights = qwen_weights_open(path, error, error_size);
    if (!talker->weights) {
        free(talker);
        return NULL;
    }

    const qwen_weights *w = talker->weights;
#define REQUIRE(target, name, ndim, ...) do {                                  \
        const int64_t expected[] = __VA_ARGS__;                                \
        (target) = qwen_weights_bf16(w, (name), (ndim), expected,              \
                                     error, error_size);                       \
        if (!(target)) { qwen_talker_free(talker); return NULL; }              \
    } while (0)
#define REQUIRE_AT(target, pattern, index, ndim, ...) do {                     \
        const int64_t expected[] = __VA_ARGS__;                                \
        (target) = qwen_weights_bf16_at(w, (pattern), (index), (ndim),         \
                                        expected, error, error_size);          \
        if (!(target)) { qwen_talker_free(talker); return NULL; }              \
    } while (0)

    REQUIRE(talker->codec_embedding, "model.codec_embedding.weight", 2,
            {QWEN_TALKER_VOCAB, QWEN_TALKER_WIDTH});
    REQUIRE(talker->text_embedding, "model.text_embedding.weight", 2,
            {-1, QWEN_TALKER_TEXT_WIDTH});
    REQUIRE(talker->projection_1_weight, "text_projection.linear_fc1.weight", 2,
            {QWEN_TALKER_TEXT_WIDTH, QWEN_TALKER_TEXT_WIDTH});
    REQUIRE(talker->projection_1_bias, "text_projection.linear_fc1.bias", 1,
            {QWEN_TALKER_TEXT_WIDTH});
    REQUIRE(talker->projection_2_weight, "text_projection.linear_fc2.weight", 2,
            {QWEN_TALKER_WIDTH, QWEN_TALKER_TEXT_WIDTH});
    REQUIRE(talker->projection_2_bias, "text_projection.linear_fc2.bias", 1,
            {QWEN_TALKER_WIDTH});
    REQUIRE(talker->final_norm, "model.norm.weight", 1, {QWEN_TALKER_WIDTH});
    REQUIRE(talker->codec_head, "codec_head.weight", 2,
            {QWEN_TALKER_VOCAB, QWEN_TALKER_WIDTH});

    for (int index = 0; index < QWEN_TALKER_LAYERS; index++) {
        qwen_block_weights *layer = &talker->layers[index];
        REQUIRE_AT(layer->input_layernorm,
                   "model.layers.%d.input_layernorm.weight", index, 1,
                   {QWEN_TALKER_WIDTH});
        REQUIRE_AT(layer->post_attention_layernorm,
                   "model.layers.%d.post_attention_layernorm.weight", index, 1,
                   {QWEN_TALKER_WIDTH});
        REQUIRE_AT(layer->q_proj, "model.layers.%d.self_attn.q_proj.weight",
                   index, 2, {Q_WIDTH, QWEN_TALKER_WIDTH});
        REQUIRE_AT(layer->k_proj, "model.layers.%d.self_attn.k_proj.weight",
                   index, 2, {KV_WIDTH, QWEN_TALKER_WIDTH});
        REQUIRE_AT(layer->v_proj, "model.layers.%d.self_attn.v_proj.weight",
                   index, 2, {KV_WIDTH, QWEN_TALKER_WIDTH});
        REQUIRE_AT(layer->o_proj, "model.layers.%d.self_attn.o_proj.weight",
                   index, 2, {QWEN_TALKER_WIDTH, Q_WIDTH});
        REQUIRE_AT(layer->q_norm, "model.layers.%d.self_attn.q_norm.weight",
                   index, 1, {QWEN_TALKER_HEAD_DIM});
        REQUIRE_AT(layer->k_norm, "model.layers.%d.self_attn.k_norm.weight",
                   index, 1, {QWEN_TALKER_HEAD_DIM});
        REQUIRE_AT(layer->gate_proj, "model.layers.%d.mlp.gate_proj.weight",
                   index, 2, {QWEN_TALKER_FFN, QWEN_TALKER_WIDTH});
        REQUIRE_AT(layer->up_proj, "model.layers.%d.mlp.up_proj.weight",
                   index, 2, {QWEN_TALKER_FFN, QWEN_TALKER_WIDTH});
        REQUIRE_AT(layer->down_proj, "model.layers.%d.mlp.down_proj.weight",
                   index, 2, {QWEN_TALKER_WIDTH, QWEN_TALKER_FFN});
    }
#undef REQUIRE
#undef REQUIRE_AT

    const size_t cache_entries =
        (size_t)QWEN_TALKER_LAYERS * (size_t)max_tokens * KV_WIDTH;
    talker->keys = calloc(cache_entries, sizeof(float));
    talker->values = calloc(cache_entries, sizeof(float));
    talker->hidden = calloc((size_t)max_tokens * QWEN_TALKER_WIDTH, sizeof(float));
    talker->normed = calloc((size_t)max_tokens * QWEN_TALKER_WIDTH, sizeof(float));
    if (!talker->keys || !talker->values || !talker->hidden || !talker->normed ||
        !qwen_scratch_init(&talker->scratch, &talker->config, max_tokens) ||
        !qwen_rope_init(&talker->rope, &talker->config, max_tokens)) {
        snprintf(error, error_size,
                 "cannot allocate the talker's %d-token cache", max_tokens);
        qwen_talker_free(talker);
        return NULL;
    }
    return talker;
}

void qwen_talker_free(qwen_talker *talker) {
    if (!talker) return;
    qwen_scratch_release(&talker->scratch);
    qwen_rope_release(&talker->rope);
    free(talker->keys);
    free(talker->values);
    free(talker->hidden);
    free(talker->normed);
    qwen_weights_close(talker->weights);
    free(talker);
}

void qwen_talker_reset(qwen_talker *talker) {
    if (talker) talker->cached = 0;
}

int qwen_talker_cached(const qwen_talker *talker) {
    return talker ? talker->cached : 0;
}

void qwen_talker_set_probe(qwen_talker *talker, qwen_talker_probe probe,
                           void *opaque) {
    if (!talker) return;
    talker->probe = probe;
    talker->probe_opaque = opaque;
}

int qwen_talker_embed_codec(const qwen_talker *talker, const uint32_t *ids,
                            int count, float *out, char *error,
                            size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!talker || !ids || !out) {
        snprintf(error, error_size, "invalid arguments");
        return 0;
    }
    for (int index = 0; index < count; index++) {
        if (ids[index] >= QWEN_TALKER_VOCAB) {
            snprintf(error, error_size, "codec id %u is out of range",
                     ids[index]);
            return 0;
        }
        const uint16_t *row =
            talker->codec_embedding + (size_t)ids[index] * QWEN_TALKER_WIDTH;
        float *target = out + (size_t)index * QWEN_TALKER_WIDTH;
        for (int channel = 0; channel < QWEN_TALKER_WIDTH; channel++)
            target[channel] = qwen_widen(row[channel]);
    }
    return 1;
}

int qwen_talker_embed_text(const qwen_talker *talker, const uint32_t *ids,
                           int count, float *out, char *error,
                           size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!talker || !ids || !out || count < 1) {
        snprintf(error, error_size, "invalid arguments");
        return 0;
    }
    float *wide = calloc((size_t)count * QWEN_TALKER_TEXT_WIDTH, sizeof(float));
    float *hidden = calloc((size_t)count * QWEN_TALKER_TEXT_WIDTH, sizeof(float));
    if (!wide || !hidden) {
        free(wide);
        free(hidden);
        snprintf(error, error_size, "out of memory");
        return 0;
    }
    for (int index = 0; index < count; index++) {
        const uint16_t *row = talker->text_embedding +
            (size_t)ids[index] * QWEN_TALKER_TEXT_WIDTH;
        float *target = wide + (size_t)index * QWEN_TALKER_TEXT_WIDTH;
        for (int channel = 0; channel < QWEN_TALKER_TEXT_WIDTH; channel++)
            target[channel] = qwen_widen(row[channel]);
    }

    /* linear_fc2(silu(linear_fc1(x))), both carrying a bias */
    qwen_matmul(talker->projection_1_weight, wide, hidden,
                QWEN_TALKER_TEXT_WIDTH, QWEN_TALKER_TEXT_WIDTH, count);
    for (int index = 0; index < count; index++) {
        float *row = hidden + (size_t)index * QWEN_TALKER_TEXT_WIDTH;
        for (int channel = 0; channel < QWEN_TALKER_TEXT_WIDTH; channel++) {
            const float value =
                row[channel] + qwen_widen(talker->projection_1_bias[channel]);
            row[channel] = value / (1.0f + expf(-value));
        }
    }
    qwen_matmul(talker->projection_2_weight, hidden, out,
                QWEN_TALKER_TEXT_WIDTH, QWEN_TALKER_WIDTH, count);
    for (int index = 0; index < count; index++) {
        float *row = out + (size_t)index * QWEN_TALKER_WIDTH;
        for (int channel = 0; channel < QWEN_TALKER_WIDTH; channel++)
            row[channel] += qwen_widen(talker->projection_2_bias[channel]);
    }
    free(wide);
    free(hidden);
    return 1;
}

int qwen_talker_forward(qwen_talker *talker, const float *embeddings,
                        int tokens, float *hidden_out, float *logits,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!talker || !embeddings || tokens < 1) {
        snprintf(error, error_size, "invalid arguments");
        return 0;
    }
    if (talker->cached + tokens > talker->max_tokens) {
        snprintf(error, error_size,
                 "the utterance needs %d tokens but the cache holds %d",
                 talker->cached + tokens, talker->max_tokens);
        return 0;
    }

    const int base = talker->cached;
    float *x = talker->hidden;
    memcpy(x, embeddings, (size_t)tokens * QWEN_TALKER_WIDTH * sizeof(float));

    for (int index = 0; index < QWEN_TALKER_LAYERS; index++) {
        float *keys = talker->keys +
            ((size_t)index * talker->max_tokens) * KV_WIDTH;
        float *values = talker->values +
            ((size_t)index * talker->max_tokens) * KV_WIDTH;
        qwen_block_forward(&talker->config, &talker->layers[index],
                           &talker->rope, x, tokens, base, keys, values,
                           talker->max_tokens, &talker->scratch);
        if (talker->probe) talker->probe(index, x, tokens, talker->probe_opaque);
    }

    talker->cached = base + tokens;

    if (hidden_out || logits) {
        qwen_rms_norm(x, talker->final_norm, talker->normed,
                      QWEN_TALKER_WIDTH, tokens, talker->config.rms_epsilon);
        if (hidden_out)
            memcpy(hidden_out, talker->normed,
                   (size_t)tokens * QWEN_TALKER_WIDTH * sizeof(float));
        if (logits)
            qwen_matmul(talker->codec_head, talker->normed, logits,
                        QWEN_TALKER_WIDTH, QWEN_TALKER_VOCAB, tokens);
    }
    return 1;
}
