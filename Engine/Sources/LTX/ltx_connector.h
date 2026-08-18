/* LTX-2.5's embeddings connector: the Gemma tower's features to DiT context.
 *
 * Eight transformer blocks per stream, video and audio, over a span the
 * *tokenizer* sets rather than the connector. Its 128 learnable registers
 * **replace padded positions** and do not extend the sequence -- reading 128
 * as the output length silently halves the context, which is a bug this port
 * shipped for a day.
 *
 * The weights live in the DiT checkpoint, not the text encoder's, which is why
 * this stage runs after the tower has been freed rather than beside it.
 */
#ifndef LTX_CONNECTOR_H
#define LTX_CONNECTOR_H

#include "h3_gpu.h"
#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>

#define LTX_CONNECTOR_VIDEO_DIM 4096
#define LTX_CONNECTOR_AUDIO_DIM 2048
/* Learnable registers, which fill the padded slots -- not a span. */
#define LTX_CONNECTOR_REGISTERS 128

/* `video_features` and `audio_features` are `[tokens][dim]` as the tower's
 * aggregation leaves them. `span` is the tokenizer's padded length -- 256 for
 * `LTXVGemmaTokenizer`'s default -- and slots from `tokens` to `span` take
 * register `slot % 128`.
 *
 * `video_context` receives `span * 4096` floats and `audio_context`
 * `span * 2048`, which is what the DiT cross-attends to.
 *
 * Returns 0 with `error` set on failure. */
int ltx_connector_run(h3_gpu *gpu, const h3_weight_store *dit,
                      const float *video_features, const float *audio_features,
                      uint32_t tokens, uint32_t span, float *video_context,
                      float *audio_context, char *error, size_t error_size);

#endif
