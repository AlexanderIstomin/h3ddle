#include "zimage_encoder.h"
#include "qwen_block.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH     ZIMAGE_CAP_DIM
#define HEADS     32
#define KV_HEADS  8
#define HEAD_DIM  128
#define FFN       9728
#define VOCAB     151936
#define THETA     1000000.0f
#define EPSILON   1e-6f

int zimage_encode(qwen_weights *encoder, const uint32_t *ids, int count,
                  float *out, zimage_encode_tick tick, void *tick_context,
                  char *error, size_t error_size) {
    const qwen_block_config config = {
        .width = WIDTH, .heads = HEADS, .kv_heads = KV_HEADS,
        .head_dim = HEAD_DIM, .ffn = FFN,
        .rope_theta = THETA, .rms_epsilon = EPSILON,
    };

    const int64_t embed_shape[2] = {VOCAB, WIDTH};
    const uint16_t *table = qwen_weights_bf16(encoder, "model.embed_tokens.weight",
                                              2, embed_shape, error, error_size);
    if (!table) return 0;
    for (int token = 0; token < count; token++) {
        const uint16_t *row = table + (size_t)ids[token] * WIDTH;
        for (int channel = 0; channel < WIDTH; channel++)
            out[(size_t)token * WIDTH + channel] = qwen_widen(row[channel]);
    }

    qwen_rope rope;
    qwen_scratch scratch;
    if (!qwen_rope_init(&rope, &config, count) ||
        !qwen_scratch_init(&scratch, &config, count)) {
        snprintf(error, error_size, "out of memory sizing the encoder");
        return 0;
    }
    float *keys = calloc((size_t)count * KV_HEADS * HEAD_DIM, sizeof(float));
    float *values = calloc((size_t)count * KV_HEADS * HEAD_DIM, sizeof(float));
    if (!keys || !values) {
        snprintf(error, error_size, "out of memory sizing the encoder cache");
        free(keys); free(values);
        return 0;
    }

    int ok = 1;
    for (int layer = 0; layer < ZIMAGE_ENCODER_USED && ok; layer++) {
        qwen_block_weights block = {0};
#define TAKE(field, pattern, ndim, ...) do {                                  \
        const int64_t shape[] = __VA_ARGS__;                                  \
        block.field = qwen_weights_bf16_at(encoder, pattern, layer, ndim,     \
                                           shape, error, error_size);         \
        if (!block.field) { ok = 0; break; }                                  \
    } while (0)
        TAKE(input_layernorm, "model.layers.%d.input_layernorm.weight", 1, {WIDTH});
        TAKE(post_attention_layernorm,
             "model.layers.%d.post_attention_layernorm.weight", 1, {WIDTH});
        TAKE(q_proj, "model.layers.%d.self_attn.q_proj.weight", 2,
             {HEADS * HEAD_DIM, WIDTH});
        TAKE(k_proj, "model.layers.%d.self_attn.k_proj.weight", 2,
             {KV_HEADS * HEAD_DIM, WIDTH});
        TAKE(v_proj, "model.layers.%d.self_attn.v_proj.weight", 2,
             {KV_HEADS * HEAD_DIM, WIDTH});
        TAKE(o_proj, "model.layers.%d.self_attn.o_proj.weight", 2,
             {WIDTH, HEADS * HEAD_DIM});
        TAKE(q_norm, "model.layers.%d.self_attn.q_norm.weight", 1, {HEAD_DIM});
        TAKE(k_norm, "model.layers.%d.self_attn.k_norm.weight", 1, {HEAD_DIM});
        TAKE(gate_proj, "model.layers.%d.mlp.gate_proj.weight", 2, {FFN, WIDTH});
        TAKE(up_proj, "model.layers.%d.mlp.up_proj.weight", 2, {FFN, WIDTH});
        TAKE(down_proj, "model.layers.%d.mlp.down_proj.weight", 2, {WIDTH, FFN});
#undef TAKE
        if (!ok) break;
        qwen_block_forward(&config, &block, &rope, out, count, 0,
                           keys, values, count, &scratch);
        /* After the block rather than before it, so the count reports work
         * finished rather than work started. */
        if (tick) tick(layer + 1, ZIMAGE_ENCODER_USED, tick_context);
    }
    /* No final norm, deliberately: the DiT taps the stack one block early,
     * before model.norm would apply. */

    free(keys); free(values);
    qwen_rope_release(&rope);
    qwen_scratch_release(&scratch);
    return ok;
}
