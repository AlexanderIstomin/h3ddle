#ifndef QWEN_BLOCK_H
#define QWEN_BLOCK_H

#include <stddef.h>
#include <stdint.h>

/* The Qwen3 decoder block, shared by the talker and the code predictor.
 *
 * The two models are the same block in different quantities — 28 layers
 * against 5 — with identical width, head counts and rope theta. They differ
 * only in what feeds them and what reads the result, so the arithmetic lives
 * here once.
 *
 * Two details are easy to place wrongly and are worth stating where the code
 * is rather than where it is called:
 *
 *  - queries and keys are RMS-normalised per head, over head_dim alone, after
 *    projection and before rotation;
 *  - the rotation pairs channel i with i + head_dim/2, matching rotate_half,
 *    not the adjacent-pair convention.
 */

typedef struct {
    int width;          /* residual width, 1024 for both models */
    int heads;          /* query heads */
    int kv_heads;       /* key/value heads; heads/kv_heads share each key */
    int head_dim;
    int ffn;
    float rope_theta;
    float rms_epsilon;
} qwen_block_config;

typedef struct {
    const uint16_t *input_layernorm;
    const uint16_t *post_attention_layernorm;
    const uint16_t *q_proj, *k_proj, *v_proj, *o_proj;
    const uint16_t *q_norm, *k_norm;
    const uint16_t *gate_proj, *up_proj, *down_proj;
} qwen_block_weights;

/* Precomputed rotation angles.
 *
 * The inverse frequencies depend only on the head width and theta, and the
 * angles only on those and the position — none of it on the weights or the
 * input. Computing them inline costs a powf, a cosf and a sinf per channel per
 * head per layer, which came to roughly 475,000 transcendental calls per audio
 * frame, on the calling thread. The whole table is a megabyte. */
typedef struct {
    float *cosines;      /* [positions][head_dim / 2] */
    float *sines;
    int positions;
    int half;
} qwen_rope;

int qwen_rope_init(qwen_rope *rope, const qwen_block_config *config,
                   int positions);
void qwen_rope_release(qwen_rope *rope);

/* Per-forward working buffers, allocated once for the longest run the caller
 * expects rather than per layer. */
typedef struct {
    float *normed, *query, *key, *value, *attention, *gate, *up;
    int tokens;
} qwen_scratch;

int qwen_scratch_init(qwen_scratch *scratch, const qwen_block_config *config,
                      int max_tokens);
void qwen_scratch_release(qwen_scratch *scratch);

/* bf16 is the top 16 bits of an f32, so widening is a shift rather than a
 * conversion. */
static inline float qwen_widen(uint16_t value) {
    union { uint32_t bits; float number; } cast;
    cast.bits = (uint32_t)value << 16;
    return cast.number;
}

/* y[tokens][outputs] = x[tokens][inputs] . weights[outputs][inputs]^T */
void qwen_matmul(const uint16_t *weights, const float *x, float *y,
                 int inputs, int outputs, int tokens);

/* Qwen3 RMSNorm: scales by weight, not by (1 + weight) as Gemma does. */
void qwen_rms_norm(const float *x, const uint16_t *weight, float *y,
                   int width, int rows, float epsilon);

/* Runs one block in place over `tokens` positions starting at `base`,
 * appending their keys and values to the caches. */
void qwen_block_forward(const qwen_block_config *config,
                        const qwen_block_weights *weights,
                        const qwen_rope *rope,
                        float *x, int tokens, int base,
                        float *keys, float *values, int max_tokens,
                        qwen_scratch *scratch);

#endif
