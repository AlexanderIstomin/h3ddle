#ifndef QWEN_PREDICTOR_H
#define QWEN_PREDICTOR_H

#include <stddef.h>
#include <stdint.h>

/* The code predictor: the 5-layer model that emits code groups 1..15 for each
 * audio frame, after the talker has produced group 0.
 *
 * It is small and it dominates. Fifteen sequential passes per frame, each
 * re-reading the whole 157 MB of weights, is 73% of the layer passes and about
 * two thirds of generation time — more than the 28-layer talker. Nothing can
 * be batched because each code selects the embedding the next step consumes.
 *
 * Per frame the sequence is [talker hidden state, embedding of the group-0
 * code], then one appended position per code produced. lm_head[i] emits group
 * i+1 and codec_embedding[i] embeds what it produced — the two arrays are
 * indexed alike, which is the off-by-one worth being careful about.
 *
 * No key/value cache: the sequence never exceeds sixteen positions, so
 * recomputing costs less than the bookkeeping would. */

#define QWEN_CODE_GROUPS      16
#define QWEN_PREDICTOR_LAYERS 5
#define QWEN_PREDICTOR_VOCAB  2048

typedef struct qwen_predictor qwen_predictor;

qwen_predictor *qwen_predictor_load(const char *path, char *error,
                                    size_t error_size);
void qwen_predictor_free(qwen_predictor *predictor);

/* Emits the fifteen remaining codes for one frame.
 *
 * `hidden` is the talker's post-norm state for the frame and `group0` the
 * embedding of the code it sampled, both QWEN_TALKER_WIDTH wide. `codes`
 * receives QWEN_CODE_GROUPS - 1 entries.
 *
 * `temperature` at or below zero takes the argmax, which is what the goldens
 * were captured with; above zero it samples. `seed` advances so successive
 * frames differ. */
int qwen_predictor_run(qwen_predictor *predictor, const float *hidden,
                       const float *group0, uint32_t *codes,
                       float temperature, uint64_t *seed,
                       char *error, size_t error_size);

/* Adds the fifteen group embeddings for `codes` into `out`.
 *
 * The talker's next input is the sum of all sixteen group embeddings, so the
 * caller starts from the group-0 embedding and accumulates these on top. */
int qwen_predictor_accumulate(const qwen_predictor *predictor,
                              const uint32_t *codes, float *out,
                              char *error, size_t error_size);

/* Reports the logits of each step, for checking against the reference. */
typedef void (*qwen_predictor_probe)(int step, const float *logits,
                                     int vocab, void *opaque);
void qwen_predictor_set_probe(qwen_predictor *predictor,
                              qwen_predictor_probe probe, void *opaque);

#endif
