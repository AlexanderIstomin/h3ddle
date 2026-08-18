/* LTX-2.5's prompt encoder: Gemma 4 12B, int8 ConvRot, and the aggregation
 * that turns its hidden states into DiT features.
 *
 * The tower is the memory constraint of the whole pipeline. Its weights and
 * the DiT's do not fit together -- about 37 GB -- so this stage runs, hands
 * back its features, and is freed before the DiT store is opened. That
 * ordering is a requirement of the pipeline, not scaffolding.
 */
#ifndef LTX_TEXT_H
#define LTX_TEXT_H

#include "h3_gpu.h"
#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>

#define LTX_TEXT_VIDEO_DIM 4096
#define LTX_TEXT_AUDIO_DIM 2048
#define LTX_TEXT_LAYERS 48
/* One per layer plus the final-normed state: what the aggregation consumes. */
#define LTX_TEXT_STATES (LTX_TEXT_LAYERS + 1)

/* Called once a layer, so a caller can show that something is happening across
 * the minutes this takes. Optional; returning zero abandons the run. */
typedef int (*ltx_text_tick)(int layer, int layers, void *context);

/* `ids` are the tokenizer's, already left-padded to `tokens` -- 256 for
 * `LTXVGemmaTokenizer`'s default. The padding is not stripped: the aggregation
 * runs over every slot and the connector is what replaces the padded ones.
 *
 * `video_features` receives `tokens * 4096` floats and `audio_features`
 * `tokens * 2048`, which is what `ltx_connector_run` takes.
 *
 * Returns 0 with `error` set on failure, and 0 with an empty `error` when the
 * tick cancelled. */
int ltx_text_encode(h3_gpu *gpu, const h3_weight_store *encoder,
                    const int32_t *ids, uint32_t tokens, float *video_features,
                    float *audio_features, ltx_text_tick tick,
                    void *tick_context, char *error, size_t error_size);

#endif
