/* One S3-DiT block, shared by the layer check and the full forward.
 *
 * Shared rather than copied because the two harnesses must be running the
 * same arithmetic for the layer gate to say anything about the forward gate;
 * two copies would drift and the drift would look like a plumbing fault.
 *
 * Two details differ from the Qwen block next door and neither is visible in
 * the shapes:
 *
 *  - the rotation pairs adjacent channels (2i, 2i+1), where Qwen pairs i with
 *    i + head_dim/2. Both are "rope"; they are not the same permutation;
 *  - attention is bidirectional — no causal mask, no kv cache.
 */
#ifndef ZIMAGE_BLOCK_H
#define ZIMAGE_BLOCK_H

#include <stddef.h>
#include <stdint.h>

#define DIM          3840
#define HEADS        30
#define HEAD_DIM     128
#define FFN          10240
#define CAP_DIM      2560
#define ADALN        256
#define PATCH_DIM    64
#define HALF         (HEAD_DIM / 2)

#define NORM_EPS     1e-5f
#define FINAL_EPS    1e-6f
#define ROPE_THETA   256.0

typedef struct {
    const uint16_t *qkv, *out, *q_norm, *k_norm;
    const uint16_t *attention_norm1, *attention_norm2;
    const uint16_t *ffn_norm1, *ffn_norm2;
    const uint16_t *w1, *w2, *w3;
    const uint16_t *adaln_weight, *adaln_bias;   /* NULL for context_refiner */
} zimage_block_weights;

typedef struct {
    float *normed, *qkv, *query, *key, *value, *attention;
    float *gate, *up, *modulation, *residual;
    int tokens;
} zimage_scratch;

int zimage_scratch_init(zimage_scratch *scratch, int max_tokens);
void zimage_scratch_release(zimage_scratch *scratch);

float zimage_silu(float x);

/* y[rows][outputs] = x[rows][inputs] . weight^T + bias, bias optional. */
void zimage_linear(const uint16_t *weight, const uint16_t *bias,
                   const float *x, float *y,
                   int inputs, int outputs, int rows);

/* The three axes own 32 + 48 + 48 channels of the head, so 16 + 24 + 24
 * complex pairs, each indexed by its own coordinate. A caption token, whose
 * two spatial coordinates are zero, still occupies all 64 pairs — 48 of them
 * at angle zero — rather than only the 16 it varies in. */
void zimage_rope_table(const float *pos_ids, float *cosines, float *sines,
                       int tokens);

/* Runs one block in place over `tokens` positions. `adaln_input` is read only
 * when the block carries adaLN weights. */
void zimage_block_forward(const zimage_block_weights *weights,
                          zimage_scratch *scratch, float *x,
                          const float *adaln_input,
                          const float *cosines, const float *sines, int tokens);

#endif
