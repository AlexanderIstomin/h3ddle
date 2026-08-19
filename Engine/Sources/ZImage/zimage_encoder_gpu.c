#include "zimage_encoder.h"

#include "h3_gpu.h"
#include "qwen_block.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    WIDTH = ZIMAGE_CAP_DIM,
    HEADS = 32,
    KV_HEADS = 8,
    HEAD_DIM = 128,
    QUERY_WIDTH = HEADS * HEAD_DIM,
    KV_WIDTH = KV_HEADS * HEAD_DIM,
    FFN = 9728,
    VOCAB = 151936,
    ROPE_HALF = HEAD_DIM / 2,
};

static const float RMS_EPSILON = 1e-6f;
static const double ROPE_THETA = 1000000.0;

typedef struct {
    h3_gpu_tensor *input_norm;
    h3_gpu_tensor *post_attention_norm;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *attention_output;
    h3_gpu_tensor *query_norm;
    h3_gpu_tensor *key_norm;
    h3_gpu_tensor *gate;
    h3_gpu_tensor *up;
    h3_gpu_tensor *down;
} zimage_encoder_layer;

typedef struct {
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *normed;
    h3_gpu_tensor *branch;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *attention;
    h3_gpu_tensor *gate;
    h3_gpu_tensor *up;
    h3_gpu_tensor *rope_cos;
    h3_gpu_tensor *rope_sin;
} zimage_encoder_workspace;

static int fail(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
    return 0;
}

static int fail_gpu(h3_gpu *gpu, int layer, const char *operation,
                    char *error, size_t error_size) {
    if (error && error_size) {
        if (layer >= 0)
            snprintf(error, error_size, "Z-Image text encoder layer %d %s "
                     "failed: %s", layer, operation, h3_gpu_error(gpu));
        else
            snprintf(error, error_size, "Z-Image text encoder %s failed: %s",
                     operation, h3_gpu_error(gpu));
    }
    return 0;
}

static h3_gpu_tensor *upload_at(h3_gpu *gpu, const qwen_weights *encoder,
                                const char *pattern, int layer, int dimensions,
                                const int64_t *shape, size_t elements,
                                char *error, size_t error_size) {
    const uint16_t *values = qwen_weights_bf16_at(
        encoder, pattern, layer, dimensions, shape, error, error_size);
    if (!values) return NULL;
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(gpu, values, elements);
    if (!tensor && error && error_size)
        snprintf(error, error_size, "cannot upload Z-Image text encoder "
                 "layer %d %s: %s", layer, pattern, h3_gpu_error(gpu));
    return tensor;
}

static int load_layer(h3_gpu *gpu, const qwen_weights *encoder, int index,
                      zimage_encoder_layer *layer,
                      char *error, size_t error_size) {
#define UPLOAD(field, pattern, elements, dimensions, ...) do {                 \
    const int64_t shape[] = __VA_ARGS__;                                       \
    layer->field = upload_at(gpu, encoder, pattern, index, dimensions, shape, \
                             elements, error, error_size);                     \
    if (!layer->field) return 0;                                               \
} while (0)
    UPLOAD(input_norm, "model.layers.%d.input_layernorm.weight",
           WIDTH, 1, {WIDTH});
    UPLOAD(post_attention_norm,
           "model.layers.%d.post_attention_layernorm.weight",
           WIDTH, 1, {WIDTH});
    UPLOAD(query, "model.layers.%d.self_attn.q_proj.weight",
           (size_t)QUERY_WIDTH * WIDTH, 2, {QUERY_WIDTH, WIDTH});
    UPLOAD(key, "model.layers.%d.self_attn.k_proj.weight",
           (size_t)KV_WIDTH * WIDTH, 2, {KV_WIDTH, WIDTH});
    UPLOAD(value, "model.layers.%d.self_attn.v_proj.weight",
           (size_t)KV_WIDTH * WIDTH, 2, {KV_WIDTH, WIDTH});
    UPLOAD(attention_output, "model.layers.%d.self_attn.o_proj.weight",
           (size_t)WIDTH * QUERY_WIDTH, 2, {WIDTH, QUERY_WIDTH});
    UPLOAD(query_norm, "model.layers.%d.self_attn.q_norm.weight",
           HEAD_DIM, 1, {HEAD_DIM});
    UPLOAD(key_norm, "model.layers.%d.self_attn.k_norm.weight",
           HEAD_DIM, 1, {HEAD_DIM});
    UPLOAD(gate, "model.layers.%d.mlp.gate_proj.weight",
           (size_t)FFN * WIDTH, 2, {FFN, WIDTH});
    UPLOAD(up, "model.layers.%d.mlp.up_proj.weight",
           (size_t)FFN * WIDTH, 2, {FFN, WIDTH});
    UPLOAD(down, "model.layers.%d.mlp.down_proj.weight",
           (size_t)WIDTH * FFN, 2, {WIDTH, FFN});
#undef UPLOAD
    return 1;
}

static void release_layer(zimage_encoder_layer *layer) {
    h3_gpu_tensor_free(layer->input_norm);
    h3_gpu_tensor_free(layer->post_attention_norm);
    h3_gpu_tensor_free(layer->query);
    h3_gpu_tensor_free(layer->key);
    h3_gpu_tensor_free(layer->value);
    h3_gpu_tensor_free(layer->attention_output);
    h3_gpu_tensor_free(layer->query_norm);
    h3_gpu_tensor_free(layer->key_norm);
    h3_gpu_tensor_free(layer->gate);
    h3_gpu_tensor_free(layer->up);
    h3_gpu_tensor_free(layer->down);
    memset(layer, 0, sizeof(*layer));
}

static void release_workspace(zimage_encoder_workspace *work) {
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
    memset(work, 0, sizeof(*work));
}

static int allocate_workspace(h3_gpu *gpu, int tokens,
                              zimage_encoder_workspace *work,
                              char *error, size_t error_size) {
    const size_t rows = (size_t)tokens;
    work->hidden = h3_gpu_tensor_new_bf16(gpu, rows * WIDTH);
    work->normed = h3_gpu_tensor_new_bf16(gpu, rows * WIDTH);
    work->branch = h3_gpu_tensor_new_bf16(gpu, rows * WIDTH);
    work->query = h3_gpu_tensor_new_bf16(gpu, rows * QUERY_WIDTH);
    work->key = h3_gpu_tensor_new_bf16(gpu, rows * KV_WIDTH);
    work->value = h3_gpu_tensor_new_bf16(gpu, rows * KV_WIDTH);
    work->attention = h3_gpu_tensor_new_bf16(gpu, rows * QUERY_WIDTH);
    work->gate = h3_gpu_tensor_new_bf16(gpu, rows * FFN);
    work->up = h3_gpu_tensor_new_bf16(gpu, rows * FFN);
    if (!work->hidden || !work->normed || !work->branch || !work->query ||
        !work->key || !work->value || !work->attention || !work->gate ||
        !work->up)
        return fail_gpu(gpu, -1, "workspace allocation", error, error_size);

    const size_t rope_count = rows * ROPE_HALF;
    float *cosines = malloc(rope_count * sizeof(*cosines));
    float *sines = malloc(rope_count * sizeof(*sines));
    if (!cosines || !sines) {
        free(cosines);
        free(sines);
        return fail(error, error_size,
                    "out of memory allocating Z-Image text encoder RoPE");
    }
    for (int dimension = 0; dimension < ROPE_HALF; dimension++) {
        const double inverse = pow(
            ROPE_THETA, -(double)(2 * dimension) / (double)HEAD_DIM);
        for (int position = 0; position < tokens; position++) {
            const double angle = (double)position * inverse;
            const size_t offset = (size_t)position * ROPE_HALF + dimension;
            cosines[offset] = (float)cos(angle);
            sines[offset] = (float)sin(angle);
        }
    }
    work->rope_cos = h3_gpu_tensor_from_f32(gpu, cosines, rope_count);
    work->rope_sin = h3_gpu_tensor_from_f32(gpu, sines, rope_count);
    free(cosines);
    free(sines);
    if (!work->rope_cos || !work->rope_sin)
        return fail_gpu(gpu, -1, "RoPE upload", error, error_size);
    return 1;
}

static int encode_layer(h3_gpu *gpu, const zimage_encoder_layer *layer,
                        const zimage_encoder_workspace *work, int tokens,
                        int index, char *error, size_t error_size) {
    const uint32_t rows = (uint32_t)tokens;
#define OP(call, name) do {                                                    \
    if (!(call)) return fail_gpu(gpu, index, name, error, error_size);         \
} while (0)
    OP(h3_gpu_rms_norm_bf16(gpu, work->normed, work->hidden,
                            layer->input_norm, rows, WIDTH, RMS_EPSILON),
       "input RMSNorm");
    OP(h3_gpu_linear_bf16(gpu, work->query, work->normed, layer->query, NULL,
                          rows, WIDTH, QUERY_WIDTH), "query projection");
    OP(h3_gpu_linear_bf16(gpu, work->key, work->normed, layer->key, NULL,
                          rows, WIDTH, KV_WIDTH), "key projection");
    OP(h3_gpu_linear_bf16(gpu, work->value, work->normed, layer->value, NULL,
                          rows, WIDTH, KV_WIDTH), "value projection");
    OP(h3_gpu_head_rms_norm_bf16(gpu, work->query, layer->query_norm, rows,
                                 HEADS, HEAD_DIM, RMS_EPSILON),
       "query head RMSNorm");
    OP(h3_gpu_head_rms_norm_bf16(gpu, work->key, layer->key_norm, rows,
                                 KV_HEADS, HEAD_DIM, RMS_EPSILON),
       "key head RMSNorm");
    OP(h3_gpu_rope_text_bf16(gpu, work->query, work->key, work->rope_cos,
                             work->rope_sin, rows, HEADS, KV_HEADS, HEAD_DIM, 0),
       "RoPE");
    OP(h3_gpu_gqa_causal_bf16(gpu, work->attention, work->query, work->key,
                              work->value, rows, HEADS, KV_HEADS, HEAD_DIM,
                              1.0f / sqrtf((float)HEAD_DIM)),
       "causal grouped-query attention");
    OP(h3_gpu_linear_bf16(gpu, work->branch, work->attention,
                          layer->attention_output, NULL, rows, QUERY_WIDTH,
                          WIDTH), "attention output projection");
    OP(h3_gpu_add_bf16(gpu, work->hidden, work->hidden, work->branch,
                       rows * WIDTH), "attention residual");
    OP(h3_gpu_rms_norm_bf16(gpu, work->normed, work->hidden,
                            layer->post_attention_norm, rows, WIDTH,
                            RMS_EPSILON), "post-attention RMSNorm");
    OP(h3_gpu_linear_bf16(gpu, work->gate, work->normed, layer->gate, NULL,
                          rows, WIDTH, FFN), "MLP gate projection");
    OP(h3_gpu_linear_bf16(gpu, work->up, work->normed, layer->up, NULL,
                          rows, WIDTH, FFN), "MLP up projection");
    OP(h3_gpu_silu_mul_bf16(gpu, work->gate, work->gate, work->up,
                            rows * FFN), "SwiGLU");
    OP(h3_gpu_linear_bf16(gpu, work->branch, work->gate, layer->down, NULL,
                          rows, FFN, WIDTH), "MLP down projection");
    OP(h3_gpu_add_bf16(gpu, work->hidden, work->hidden, work->branch,
                       rows * WIDTH), "MLP residual");
#undef OP
    return 1;
}

int zimage_encode_metal(qwen_weights *encoder, const char *shader_path,
                        const uint32_t *ids, int count, float *out,
                        zimage_encode_tick tick, void *tick_context,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!encoder || !shader_path || !shader_path[0] || !ids || count < 1 ||
        !out || (uint32_t)count > UINT32_MAX / FFN)
        return fail(error, error_size,
                    "invalid Z-Image Metal text encoder arguments");
    for (int token = 0; token < count; token++) {
        if (ids[token] >= VOCAB) {
            if (error && error_size)
                snprintf(error, error_size, "Z-Image token ID %u is outside "
                         "the %d-entry vocabulary", ids[token], VOCAB);
            return 0;
        }
    }

    h3_gpu *gpu = h3_gpu_create(shader_path, error, error_size);
    if (!gpu) return 0;
    h3_gpu_profile_set_label(gpu, "Z-Image Qwen prompt encoder");

    zimage_encoder_workspace work = {0};
    const size_t hidden_count = (size_t)count * WIDTH;
    uint16_t *staging = malloc(hidden_count * sizeof(*staging));
    int ok = staging && allocate_workspace(gpu, count, &work, error, error_size);
    if (!staging && error && error_size)
        snprintf(error, error_size,
                 "out of memory staging the Z-Image text encoder");

    const int64_t embedding_shape[2] = {VOCAB, WIDTH};
    const uint16_t *embedding = ok ? qwen_weights_bf16(
        encoder, "model.embed_tokens.weight", 2, embedding_shape,
        error, error_size) : NULL;
    if (ok && !embedding) ok = 0;
    if (ok) {
        for (int token = 0; token < count; token++)
            memcpy(staging + (size_t)token * WIDTH,
                   embedding + (size_t)ids[token] * WIDTH,
                   WIDTH * sizeof(*staging));
        if (!h3_gpu_tensor_write_bf16(work.hidden, staging, hidden_count))
            ok = fail_gpu(gpu, -1, "embedding upload", error, error_size);
    }

    for (int index = 0; index < ZIMAGE_ENCODER_USED && ok; index++) {
        zimage_encoder_layer layer = {0};
        ok = load_layer(gpu, encoder, index, &layer, error, error_size);
        if (ok && !h3_gpu_begin(gpu))
            ok = fail_gpu(gpu, index, "command stream", error, error_size);
        if (ok)
            ok = encode_layer(gpu, &layer, &work, count, index,
                              error, error_size);
        if (ok && !h3_gpu_submit(gpu))
            ok = fail_gpu(gpu, index, "submission", error, error_size);
        release_layer(&layer);
        if (ok && tick) tick(index + 1, ZIMAGE_ENCODER_USED, tick_context);
    }

    if (ok && !h3_gpu_tensor_read_bf16(work.hidden, staging, hidden_count))
        ok = fail_gpu(gpu, -1, "output readback", error, error_size);
    if (ok) {
        for (size_t index = 0; index < hidden_count; index++)
            out[index] = qwen_widen(staging[index]);
    }

    free(staging);
    release_workspace(&work);
    h3_gpu_free(gpu);
    return ok;
}
