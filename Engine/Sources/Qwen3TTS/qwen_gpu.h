#ifndef QWEN_GPU_H
#define QWEN_GPU_H

#include <stddef.h>
#include <stdint.h>

/* The talker's 28 blocks on the GPU, through h3.c's Metal layer.
 *
 * H3's own text encoder is a Qwen3, so every kernel this needs already exists
 * and carries its head counts as arguments: `h3_gpu_head_rms_norm_bf16` for the
 * per-head q/k norm, `h3_gpu_rope_text_bf16` for the rotation, and
 * `h3_gpu_gqa_causal_cache_bf16` for grouped-query attention continued over a
 * key cache. Only the last of those had to be added, and only because the
 * causal kernel derived its key count from the query row and so could not
 * decode against a cache.
 *
 * This carries the transformer alone — 426M parameters, about 852 MB resident.
 * The 311M-parameter text embedding stays with the CPU talker, which touches a
 * few dozen of its 151936 rows per utterance and would spend more time
 * uploading the table than reading it.
 *
 * The interface matches qwen_talker_forward deliberately, f32 at the boundary
 * and all, so the two can be swapped and compared. They will not agree to the
 * last bit: this path carries bf16 activations throughout, where the CPU one
 * carries f32, so hidden states differ at around 1e-3 relative and an occasional
 * sampled code will diverge. That is the arithmetic, not a defect.
 */

typedef struct qwen_gpu_talker qwen_gpu_talker;

/* `shader_path` is h3.c's h3_shaders.metal, compiled once per process. */
qwen_gpu_talker *qwen_gpu_talker_load(const char *path, const char *shader_path,
                                      int max_tokens, char *error,
                                      size_t error_size);
void qwen_gpu_talker_free(qwen_gpu_talker *talker);

/* Runs `tokens` positions and appends them to the cache, so a prefill is one
 * call and each generated frame is another with tokens == 1.
 *
 * `hidden`, if not NULL, receives tokens * QWEN_TALKER_WIDTH post-norm states;
 * `logits`, if not NULL, receives tokens * QWEN_TALKER_VOCAB. */
int qwen_gpu_talker_forward(qwen_gpu_talker *talker, const float *embeddings,
                            int tokens, float *hidden, float *logits,
                            char *error, size_t error_size);

void qwen_gpu_talker_reset(qwen_gpu_talker *talker);
int qwen_gpu_talker_cached(const qwen_gpu_talker *talker);

/* Engine counters accumulated since load, for telling dispatch overhead apart
 * from work actually done on the device. Decoding a single token runs hundreds
 * of very small dispatches, so the two are easy to confuse from wall time. */
int qwen_gpu_talker_stats(const qwen_gpu_talker *talker, uint64_t *dispatches,
                          uint64_t *mps_dispatches, uint64_t *submissions,
                          double *encode_seconds, double *gpu_seconds);

/* The code predictor's five layers, which matter more than the talker's
 * twenty-eight: fifteen sequential passes per frame, each re-reading the whole
 * 157 MB of weights, is about two thirds of generation time. Nothing can be
 * batched, because each code selects the embedding the next step consumes.
 *
 * Two ways of running a frame, chosen automatically. When the caller wants the
 * argmax and no probe is attached, the whole frame goes into one command buffer
 * and never returns to the host between steps: the code is chosen on the device
 * and fed straight to the embedding lookup. Otherwise — sampling, or a probe
 * that wants each step's logits — the logits come back every step, fifteen
 * round trips, and the choice is made by the same xorshift the CPU predictor
 * uses, so the same seed walks the same path on either. The two paths are
 * checked against each other and agree code for code. */
typedef struct qwen_gpu_predictor qwen_gpu_predictor;

qwen_gpu_predictor *qwen_gpu_predictor_load(const char *path,
                                            const char *shader_path,
                                            char *error, size_t error_size);
void qwen_gpu_predictor_free(qwen_gpu_predictor *predictor);

/* Emits the fifteen remaining codes for one frame; `codes` receives
 * QWEN_CODE_GROUPS - 1 entries. Mirrors qwen_predictor_run. */
int qwen_gpu_predictor_run(qwen_gpu_predictor *predictor, const float *hidden,
                           const float *group0, uint32_t *codes,
                           float temperature, uint64_t *seed, char *error,
                           size_t error_size);

int qwen_gpu_predictor_stats(const qwen_gpu_predictor *predictor,
                             uint64_t *dispatches, uint64_t *submissions,
                             double *encode_seconds, double *gpu_seconds);

/* Reports the logits of each step, matching qwen_predictor_set_probe. Attaching
 * one forces the host path, since the device path never brings them back. */
typedef void (*qwen_gpu_predictor_probe)(int step, const float *logits,
                                         int vocab, void *opaque);
void qwen_gpu_predictor_set_probe(qwen_gpu_predictor *predictor,
                                  qwen_gpu_predictor_probe probe, void *opaque);

#endif
