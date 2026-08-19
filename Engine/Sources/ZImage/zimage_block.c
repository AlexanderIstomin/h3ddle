#include "zimage_block.h"
#include "qwen_block.h"

#include <dispatch/dispatch.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static const int AXES[3] = {32, 48, 48};

float zimage_silu(float x) { return x / (1.0f + expf(-x)); }

int zimage_scratch_init(zimage_scratch *scratch, int max_tokens) {
    memset(scratch, 0, sizeof(*scratch));
    scratch->tokens = max_tokens;
    const size_t wide = (size_t)max_tokens * DIM * sizeof(float);
    scratch->normed = malloc(wide);
    scratch->qkv = malloc((size_t)max_tokens * 3 * DIM * sizeof(float));
    scratch->query = malloc(wide);
    scratch->key = malloc(wide);
    scratch->value = malloc(wide);
    scratch->attention = malloc(wide);
    scratch->residual = malloc(wide);
    scratch->gate = malloc((size_t)max_tokens * FFN * sizeof(float));
    scratch->up = malloc((size_t)max_tokens * FFN * sizeof(float));
    scratch->modulation = malloc(4 * DIM * sizeof(float));
    return scratch->normed && scratch->qkv && scratch->query && scratch->key &&
           scratch->value && scratch->attention && scratch->residual &&
           scratch->gate && scratch->up && scratch->modulation;
}

void zimage_scratch_release(zimage_scratch *scratch) {
    free(scratch->normed); free(scratch->qkv); free(scratch->query);
    free(scratch->key); free(scratch->value); free(scratch->attention);
    free(scratch->residual); free(scratch->gate); free(scratch->up);
    free(scratch->modulation);
    memset(scratch, 0, sizeof(*scratch));
}

void zimage_linear(const uint16_t *weight, const uint16_t *bias,
                   const float *x, float *y,
                   int inputs, int outputs, int rows) {
    qwen_matmul(weight, x, y, inputs, outputs, rows);
    if (!bias) return;
    for (int row = 0; row < rows; row++)
        for (int output = 0; output < outputs; output++)
            y[(size_t)row * outputs + output] += qwen_widen(bias[output]);
}

void zimage_rope_table(const float *pos_ids, float *cosines, float *sines,
                       int tokens) {
    for (int token = 0; token < tokens; token++) {
        int channel = 0;
        for (int axis = 0; axis < 3; axis++) {
            const int dim = AXES[axis];
            const double position = (double)pos_ids[(size_t)token * 3 + axis];
            for (int index = 0; index < dim / 2; index++) {
                const double frequency =
                    1.0 / pow(ROPE_THETA, (double)(2 * index) / (double)dim);
                const double angle = position * frequency;
                cosines[(size_t)token * HALF + channel] = (float)cos(angle);
                sines[(size_t)token * HALF + channel] = (float)sin(angle);
                channel++;
            }
        }
    }
}

/* Adjacent-pair rotation, in place over [tokens][heads][head_dim]. */
static void apply_rope(float *x, const float *cosines, const float *sines,
                       int tokens) {
    for (int token = 0; token < tokens; token++) {
        const float *cosine = cosines + (size_t)token * HALF;
        const float *sine = sines + (size_t)token * HALF;
        for (int head = 0; head < HEADS; head++) {
            float *row = x + ((size_t)token * HEADS + head) * HEAD_DIM;
            for (int pair = 0; pair < HALF; pair++) {
                const float real = row[2 * pair], imaginary = row[2 * pair + 1];
                row[2 * pair] = real * cosine[pair] - imaginary * sine[pair];
                row[2 * pair + 1] = real * sine[pair] + imaginary * cosine[pair];
            }
        }
    }
}

/* The engine's h3_qkv_rope_f32 pairs channel i with i + head_dim/2
 * (rotate_half), where Z-Image pairs adjacent channels. Attention is invariant
 * to permuting the head dimension of q and k *together*, so sending channel 2i
 * to slot i and 2i+1 to slot i+64 makes the engine's kernel compute exactly
 * Z-Image's rotation. The production GPU boundary reads the corresponding
 * source channels directly, so the qkv and norm weights stay untouched.
 *
 * Built with -DZIMAGE_ROPE_ROTATE_HALF this takes that path instead, permuting
 * the activations after the projection, which is the same ordering one row at
 * a time. If the forward still matches its
 * golden, the trick holds on real weights and real data and not only on paper.
 * v is deliberately left alone, and nothing is un-permuted afterwards: the dot
 * product is unchanged, so the attention output already is. */
static void permute_pairs(float *x, int tokens) {
    float row[HEAD_DIM];
    for (size_t index = 0; index < (size_t)tokens * HEADS; index++) {
        float *values = x + index * HEAD_DIM;
        for (int pair = 0; pair < HALF; pair++) {
            row[pair] = values[2 * pair];
            row[pair + HALF] = values[2 * pair + 1];
        }
        memcpy(values, row, sizeof(row));
    }
}

static void apply_rope_rotate_half(float *x, const float *cosines,
                                   const float *sines, int tokens) {
    for (int token = 0; token < tokens; token++) {
        const float *cosine = cosines + (size_t)token * HALF;
        const float *sine = sines + (size_t)token * HALF;
        for (int head = 0; head < HEADS; head++) {
            float *row = x + ((size_t)token * HEADS + head) * HEAD_DIM;
            for (int index = 0; index < HALF; index++) {
                const float low = row[index], high = row[index + HALF];
                row[index] = low * cosine[index] - high * sine[index];
                row[index + HALF] = low * sine[index] + high * cosine[index];
            }
        }
    }
}

/* Per-head RMS norm over head_dim, one [head_dim] weight shared across every
 * head, after projection and before rotation. */
static void head_norm(float *x, const uint16_t *weight, int tokens) {
    for (size_t row = 0; row < (size_t)tokens * HEADS; row++) {
        float *values = x + row * HEAD_DIM;
        float sum = 0.0f;
        for (int index = 0; index < HEAD_DIM; index++)
            sum += values[index] * values[index];
        const float scale = 1.0f / sqrtf(sum / (float)HEAD_DIM + NORM_EPS);
        for (int index = 0; index < HEAD_DIM; index++)
            values[index] *= qwen_widen(weight[index]) * scale;
    }
}

static void attention(const zimage_block_weights *weights, zimage_scratch *scratch,
                      const float *cosines, const float *sines, int tokens) {
    qwen_matmul(weights->qkv, scratch->normed, scratch->qkv, DIM, 3 * DIM, tokens);
    for (int token = 0; token < tokens; token++) {
        const float *row = scratch->qkv + (size_t)token * 3 * DIM;
        memcpy(scratch->query + (size_t)token * DIM, row, DIM * sizeof(float));
        memcpy(scratch->key + (size_t)token * DIM, row + DIM, DIM * sizeof(float));
        memcpy(scratch->value + (size_t)token * DIM, row + 2 * DIM, DIM * sizeof(float));
    }
    head_norm(scratch->query, weights->q_norm, tokens);
    head_norm(scratch->key, weights->k_norm, tokens);
#ifdef ZIMAGE_ROPE_ROTATE_HALF
    permute_pairs(scratch->query, tokens);
    permute_pairs(scratch->key, tokens);
    apply_rope_rotate_half(scratch->query, cosines, sines, tokens);
    apply_rope_rotate_half(scratch->key, cosines, sines, tokens);
#else
    apply_rope(scratch->query, cosines, sines, tokens);
    apply_rope(scratch->key, cosines, sines, tokens);
#endif

    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    float *attention_out = scratch->attention;
    const float *query = scratch->query, *key = scratch->key, *value = scratch->value;
    /* Bidirectional: every token sees every token, so no mask and no cache. */
    dispatch_apply(HEADS, DISPATCH_APPLY_AUTO, ^(size_t head) {
        float *scores = malloc((size_t)tokens * sizeof(float));
        for (int i = 0; i < tokens; i++) {
            const float *q = query + ((size_t)i * HEADS + head) * HEAD_DIM;
            float largest = -INFINITY;
            for (int j = 0; j < tokens; j++) {
                const float *k = key + ((size_t)j * HEADS + head) * HEAD_DIM;
                float dot = 0.0f;
                for (int index = 0; index < HEAD_DIM; index++)
                    dot += q[index] * k[index];
                scores[j] = dot * scale;
                if (scores[j] > largest) largest = scores[j];
            }
            float total = 0.0f;
            for (int j = 0; j < tokens; j++) {
                scores[j] = expf(scores[j] - largest);
                total += scores[j];
            }
            const float inverse = 1.0f / total;
            float *out = attention_out + ((size_t)i * HEADS + head) * HEAD_DIM;
            memset(out, 0, HEAD_DIM * sizeof(float));
            for (int j = 0; j < tokens; j++) {
                const float weight = scores[j] * inverse;
                const float *v = value + ((size_t)j * HEADS + head) * HEAD_DIM;
                for (int index = 0; index < HEAD_DIM; index++)
                    out[index] += weight * v[index];
            }
        }
        free(scores);
    });
    qwen_matmul(weights->out, scratch->attention, scratch->normed, DIM, DIM, tokens);
}

static void feed_forward(const zimage_block_weights *weights,
                         zimage_scratch *scratch,
                         const float *x, float *y, int tokens) {
    qwen_matmul(weights->w1, x, scratch->gate, DIM, FFN, tokens);
    qwen_matmul(weights->w3, x, scratch->up, DIM, FFN, tokens);
    for (size_t index = 0; index < (size_t)tokens * FFN; index++)
        scratch->gate[index] = zimage_silu(scratch->gate[index]) * scratch->up[index];
    qwen_matmul(weights->w2, scratch->gate, y, FFN, DIM, tokens);
}

/* Sandwich norm: a norm before the sublayer and another after it, the residual
 * added outside both. The unmodulated form keeps both norms and drops the
 * scale and gate entirely rather than setting them to one. */
void zimage_block_forward(const zimage_block_weights *weights,
                          zimage_scratch *scratch, float *x,
                          const float *adaln_input,
                          const float *cosines, const float *sines, int tokens) {
    const float *scale_msa = NULL, *gate_msa = NULL;
    const float *scale_mlp = NULL, *gate_mlp = NULL;
    if (weights->adaln_weight) {
        zimage_linear(weights->adaln_weight, weights->adaln_bias, adaln_input,
                      scratch->modulation, ADALN, 4 * DIM, 1);
        float *modulation = scratch->modulation;
        for (int index = 0; index < DIM; index++) {
            modulation[index] += 1.0f;
            modulation[DIM + index] = tanhf(modulation[DIM + index]);
            modulation[2 * DIM + index] += 1.0f;
            modulation[3 * DIM + index] = tanhf(modulation[3 * DIM + index]);
        }
        scale_msa = modulation;
        gate_msa = modulation + DIM;
        scale_mlp = modulation + 2 * DIM;
        gate_mlp = modulation + 3 * DIM;
    }
    float *residual = scratch->residual;

    qwen_rms_norm(x, weights->attention_norm1, scratch->normed, DIM, tokens, NORM_EPS);
    if (scale_msa)
        for (int token = 0; token < tokens; token++)
            for (int index = 0; index < DIM; index++)
                scratch->normed[(size_t)token * DIM + index] *= scale_msa[index];
    attention(weights, scratch, cosines, sines, tokens);
    qwen_rms_norm(scratch->normed, weights->attention_norm2, residual, DIM, tokens, NORM_EPS);
    for (int token = 0; token < tokens; token++)
        for (int index = 0; index < DIM; index++)
            x[(size_t)token * DIM + index] +=
                (gate_msa ? gate_msa[index] : 1.0f) * residual[(size_t)token * DIM + index];

    qwen_rms_norm(x, weights->ffn_norm1, scratch->normed, DIM, tokens, NORM_EPS);
    if (scale_mlp)
        for (int token = 0; token < tokens; token++)
            for (int index = 0; index < DIM; index++)
                scratch->normed[(size_t)token * DIM + index] *= scale_mlp[index];
    feed_forward(weights, scratch, scratch->normed, residual, tokens);
    qwen_rms_norm(residual, weights->ffn_norm2, residual, DIM, tokens, NORM_EPS);
    for (int token = 0; token < tokens; token++)
        for (int index = 0; index < DIM; index++)
            x[(size_t)token * DIM + index] +=
                (gate_mlp ? gate_mlp[index] : 1.0f) * residual[(size_t)token * DIM + index];
}
