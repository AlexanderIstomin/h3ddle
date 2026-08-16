#include "qwen_block.h"

#include <arm_neon.h>
#include <unistd.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int qwen_scratch_init(qwen_scratch *scratch, const qwen_block_config *config,
                      int max_tokens) {
    if (!scratch || !config || max_tokens < 1) return 0;
    memset(scratch, 0, sizeof(*scratch));
    scratch->tokens = max_tokens;
    const size_t tokens = (size_t)max_tokens;
    const size_t query_width = (size_t)config->heads * config->head_dim;
    const size_t kv_width = (size_t)config->kv_heads * config->head_dim;
    scratch->normed = calloc(tokens * (size_t)config->width, sizeof(float));
    scratch->query = calloc(tokens * query_width, sizeof(float));
    scratch->key = calloc(tokens * kv_width, sizeof(float));
    scratch->value = calloc(tokens * kv_width, sizeof(float));
    scratch->attention = calloc(tokens * query_width, sizeof(float));
    scratch->gate = calloc(tokens * (size_t)config->ffn, sizeof(float));
    scratch->up = calloc(tokens * (size_t)config->ffn, sizeof(float));
    if (!scratch->normed || !scratch->query || !scratch->key ||
        !scratch->value || !scratch->attention || !scratch->gate ||
        !scratch->up) {
        qwen_scratch_release(scratch);
        return 0;
    }
    return 1;
}

void qwen_scratch_release(qwen_scratch *scratch) {
    if (!scratch) return;
    free(scratch->normed);
    free(scratch->query);
    free(scratch->key);
    free(scratch->value);
    free(scratch->attention);
    free(scratch->gate);
    free(scratch->up);
    memset(scratch, 0, sizeof(*scratch));
}

/* Parallel over output rows: each worker reads a disjoint slice of the
 * weights, which is what lets the memory system deliver its full bandwidth.
 * Splitting by token instead would have every worker stream the whole
 * matrix, and these are matrices no cache can hold.
 *
 * The inner loop widens eight bf16 weights at a time. bf16 is the top half of
 * an f32, so the widening is a left shift into place — no conversion
 * instruction, and none is needed even though the M1 has no bf16 arithmetic:
 * the cost here is reading the weights, not multiplying them. */
void qwen_matmul(const uint16_t *weights, const float *x, float *y,
                 int inputs, int outputs, int tokens) {
    /* One chunk per core, not a fixed row count. Incremental decoding runs
     * one token at a time, so a projection is only a few hundred microseconds
     * of work; splitting it into forty small chunks spends more on scheduling
     * and the closing barrier than on arithmetic. */
    static int cores;
    if (!cores) {
        cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (cores < 1) cores = 8;
        if (cores > 16) cores = 16;
    }
    const int chunk = (outputs + cores - 1) / cores;
    const size_t chunks = (size_t)((outputs + chunk - 1) / chunk);
    dispatch_apply(chunks, DISPATCH_APPLY_AUTO, ^(size_t slot) {
        int first = (int)slot * chunk;
        int last = first + chunk;
        if (last > outputs) last = outputs;
        for (int token = 0; token < tokens; token++) {
            const float *row = x + (size_t)token * inputs;
            float *out = y + (size_t)token * outputs;
            for (int output = first; output < last; output++) {
                const uint16_t *w = weights + (size_t)output * inputs;
                float32x4_t low = vdupq_n_f32(0.0f), high = vdupq_n_f32(0.0f);
                int index = 0;
                for (; index + 8 <= inputs; index += 8) {
                    const uint16x8_t raw = vld1q_u16(w + index);
                    low = vfmaq_f32(low, vreinterpretq_f32_u32(
                        vshll_n_u16(vget_low_u16(raw), 16)), vld1q_f32(row + index));
                    high = vfmaq_f32(high, vreinterpretq_f32_u32(
                        vshll_n_u16(vget_high_u16(raw), 16)), vld1q_f32(row + index + 4));
                }
                float total = vaddvq_f32(low) + vaddvq_f32(high);
                for (; index < inputs; index++)
                    total += qwen_widen(w[index]) * row[index];
                out[output] = total;
            }
        }
    });
}

void qwen_rms_norm(const float *x, const uint16_t *weight, float *y,
                   int width, int rows, float epsilon) {
    for (int row = 0; row < rows; row++) {
        const float *in = x + (size_t)row * width;
        float *out = y + (size_t)row * width;
        float sum = 0.0f;
        for (int index = 0; index < width; index++) sum += in[index] * in[index];
        const float scale = 1.0f / sqrtf(sum / (float)width + epsilon);
        for (int index = 0; index < width; index++)
            out[index] = qwen_widen(weight[index]) * in[index] * scale;
    }
}

int qwen_rope_init(qwen_rope *rope, const qwen_block_config *config,
                   int positions) {
    if (!rope || !config || positions < 1) return 0;
    memset(rope, 0, sizeof(*rope));
    rope->half = config->head_dim / 2;
    rope->positions = positions;
    const size_t count = (size_t)positions * rope->half;
    rope->cosines = malloc(count * sizeof(float));
    rope->sines = malloc(count * sizeof(float));
    if (!rope->cosines || !rope->sines) {
        qwen_rope_release(rope);
        return 0;
    }
    for (int index = 0; index < rope->half; index++) {
        /* in double, so the table is right to the last bit of a float */
        const double inverse = pow((double)config->rope_theta,
                                   -(double)(2 * index) / (double)config->head_dim);
        for (int position = 0; position < positions; position++) {
            const double angle = (double)position * inverse;
            rope->cosines[(size_t)position * rope->half + index] = (float)cos(angle);
            rope->sines[(size_t)position * rope->half + index] = (float)sin(angle);
        }
    }
    return 1;
}

void qwen_rope_release(qwen_rope *rope) {
    if (!rope) return;
    free(rope->cosines);
    free(rope->sines);
    memset(rope, 0, sizeof(*rope));
}

/* Pairs channel i with i + head_dim/2, matching rotate_half rather than the
 * adjacent-pair convention. */
static void rotate(float *head, const qwen_rope *rope, int position) {
    const int half = rope->half;
    const float *cosines = rope->cosines + (size_t)position * half;
    const float *sines = rope->sines + (size_t)position * half;
    for (int index = 0; index < half; index++) {
        const float low = head[index], high = head[index + half];
        head[index] = low * cosines[index] - high * sines[index];
        head[index + half] = high * cosines[index] + low * sines[index];
    }
}

void qwen_block_forward(const qwen_block_config *config,
                        const qwen_block_weights *weights,
                        const qwen_rope *rope,
                        float *x, int tokens, int base,
                        float *keys, float *values, int max_tokens,
                        qwen_scratch *scratch) {
    (void)max_tokens;
    const int width = config->width;
    const int head_dim = config->head_dim;
    const int query_width = config->heads * head_dim;
    const int kv_width = config->kv_heads * head_dim;
    const int group = config->heads / config->kv_heads;

    qwen_rms_norm(x, weights->input_layernorm, scratch->normed, width, tokens,
                  config->rms_epsilon);
    qwen_matmul(weights->q_proj, scratch->normed, scratch->query,
                width, query_width, tokens);
    qwen_matmul(weights->k_proj, scratch->normed, scratch->key,
                width, kv_width, tokens);
    qwen_matmul(weights->v_proj, scratch->normed, scratch->value,
                width, kv_width, tokens);

    /* normalise and rotate before the cache, so cached keys are ready to be
     * dotted against on every later step */
    for (int token = 0; token < tokens; token++) {
        const int position = base + token;
        float *query = scratch->query + (size_t)token * query_width;
        for (int head = 0; head < config->heads; head++) {
            float *slice = query + head * head_dim;
            qwen_rms_norm(slice, weights->q_norm, slice, head_dim, 1,
                          config->rms_epsilon);
            rotate(slice, rope, position);
        }
        float *key = scratch->key + (size_t)token * kv_width;
        for (int head = 0; head < config->kv_heads; head++) {
            float *slice = key + head * head_dim;
            qwen_rms_norm(slice, weights->k_norm, slice, head_dim, 1,
                          config->rms_epsilon);
            rotate(slice, rope, position);
        }
        memcpy(keys + (size_t)position * kv_width, key,
               (size_t)kv_width * sizeof(float));
        memcpy(values + (size_t)position * kv_width,
               scratch->value + (size_t)token * kv_width,
               (size_t)kv_width * sizeof(float));
    }

    const float scale = 1.0f / sqrtf((float)head_dim);
    dispatch_apply((size_t)tokens, DISPATCH_APPLY_AUTO, ^(size_t slot) {
        const int token = (int)slot;
        const int position = base + token;
        float *scores = malloc((size_t)(position + 1) * sizeof(float));
        if (!scores) return;
        for (int head = 0; head < config->heads; head++) {
            const int kv_head = head / group;   /* grouped query attention */
            const float *query = scratch->query + (size_t)token * query_width +
                head * head_dim;
            float highest = -INFINITY;
            for (int step = 0; step <= position; step++) {
                const float *key = keys + (size_t)step * kv_width +
                    kv_head * head_dim;
                float total = 0.0f;
                for (int channel = 0; channel < head_dim; channel++)
                    total += query[channel] * key[channel];
                total *= scale;
                scores[step] = total;
                if (total > highest) highest = total;
            }
            float sum = 0.0f;
            for (int step = 0; step <= position; step++) {
                scores[step] = expf(scores[step] - highest);
                sum += scores[step];
            }
            float *out = scratch->attention + (size_t)token * query_width +
                head * head_dim;
            memset(out, 0, (size_t)head_dim * sizeof(float));
            for (int step = 0; step <= position; step++) {
                const float weight = scores[step] / sum;
                const float *value = values + (size_t)step * kv_width +
                    kv_head * head_dim;
                for (int channel = 0; channel < head_dim; channel++)
                    out[channel] += weight * value[channel];
            }
        }
        free(scores);
    });

    qwen_matmul(weights->o_proj, scratch->attention, scratch->normed,
                query_width, width, tokens);
    for (int index = 0; index < tokens * width; index++)
        x[index] += scratch->normed[index];

    qwen_rms_norm(x, weights->post_attention_layernorm, scratch->normed,
                  width, tokens, config->rms_epsilon);
    qwen_matmul(weights->gate_proj, scratch->normed, scratch->gate,
                width, config->ffn, tokens);
    qwen_matmul(weights->up_proj, scratch->normed, scratch->up,
                width, config->ffn, tokens);
    for (int index = 0; index < tokens * config->ffn; index++) {
        const float gate = scratch->gate[index];
        scratch->gate[index] = gate / (1.0f + expf(-gate)) * scratch->up[index];
    }
    qwen_matmul(weights->down_proj, scratch->gate, scratch->normed,
                config->ffn, width, tokens);
    for (int index = 0; index < tokens * width; index++)
        x[index] += scratch->normed[index];
}
