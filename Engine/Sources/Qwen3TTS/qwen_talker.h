#ifndef QWEN_TALKER_H
#define QWEN_TALKER_H

#include <stddef.h>
#include <stdint.h>

/* The Qwen3-TTS talker: 28 Qwen3 blocks that emit one audio code group per
 * frame, with the remaining fifteen coming from the code predictor.
 *
 * Two details differ from the transformer h3_text_encoder.c already runs, and
 * both are easy to place wrongly:
 *
 *  - queries and keys are RMS-normalised per head, over head_dim only, after
 *    projection and before the rotation;
 *  - the config asks for interleaved MRoPE with sections [24,20,20], but with
 *    no vision in the sequence all three position channels carry the same row,
 *    and the interleave only ever selects between channels. It collapses to
 *    plain 1-D RoPE at theta 1e6, which is what this implements. Asserted
 *    against the reference's own position_ids, not assumed.
 *
 * The talker's input is the sum of two streams — projected text embeddings and
 * codec embeddings — which the caller assembles; this takes the sum. */

#define QWEN_TALKER_WIDTH       1024
#define QWEN_TALKER_LAYERS      28
#define QWEN_TALKER_HEADS       16
#define QWEN_TALKER_KV_HEADS    8
#define QWEN_TALKER_HEAD_DIM    128
#define QWEN_TALKER_FFN         3072
#define QWEN_TALKER_VOCAB       3072      /* 2048 codes plus control tokens */
#define QWEN_TALKER_TEXT_WIDTH  2048

typedef struct qwen_talker qwen_talker;

/* `max_tokens` bounds the key/value cache, which costs 224 KB per token
 * across the 28 layers — about 90 MB for the 400 tokens a 30-second utterance
 * needs. Prompt tokens and generated frames share the budget. */
qwen_talker *qwen_talker_load(const char *path, int max_tokens,
                              char *error, size_t error_size);
void qwen_talker_free(qwen_talker *talker);

/* Text ids -> QWEN_TALKER_WIDTH, through the wide embedding table and the
 * two-layer resize projection. `out` holds count * QWEN_TALKER_WIDTH floats. */
int qwen_talker_embed_text(const qwen_talker *talker, const uint32_t *ids,
                           int count, float *out, char *error,
                           size_t error_size);

/* Codec ids -> QWEN_TALKER_WIDTH, a plain table lookup. */
int qwen_talker_embed_codec(const qwen_talker *talker, const uint32_t *ids,
                            int count, float *out, char *error,
                            size_t error_size);

/* Runs `tokens` positions and appends them to the cache, so a prefill is one
 * call and each generated frame is another with tokens == 1.
 *
 * `hidden`, if not NULL, receives tokens * QWEN_TALKER_WIDTH post-norm states;
 * the code predictor consumes the last one. `logits`, if not NULL, receives
 * tokens * QWEN_TALKER_VOCAB. */
int qwen_talker_forward(qwen_talker *talker, const float *embeddings,
                        int tokens, float *hidden, float *logits,
                        char *error, size_t error_size);

/* Drops the cache so the next forward starts a new utterance. */
void qwen_talker_reset(qwen_talker *talker);
int qwen_talker_cached(const qwen_talker *talker);

/* Reports each layer's output during a forward, for checking against the
 * reference layer by layer. */
typedef void (*qwen_talker_probe)(int layer, const float *hidden, int tokens,
                                  void *opaque);
void qwen_talker_set_probe(qwen_talker *talker, qwen_talker_probe probe,
                           void *opaque);

#endif
