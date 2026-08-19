/* Z-Image's prompt encoder: Qwen3-4B, run as a text encoder.
 *
 * The DiT reads `hidden_states[-2]` — the output of the *second-to-last*
 * block, un-normed — so 35 of the 36 layers run and `model.norm` never does.
 * Verified against transformers' indexing rather than read off the source,
 * because taking the last layer instead is a silent 202 MB of wasted work
 * that produces a plausible, subtly wrong conditioning.
 */
#ifndef ZIMAGE_ENCODER_H
#define ZIMAGE_ENCODER_H

#include "qwen_weights.h"

#include <stddef.h>
#include <stdint.h>

#define ZIMAGE_CAP_DIM   2560
#define ZIMAGE_ENCODER_LAYERS 36
/* One short of the whole stack: the last block's output is what model.norm
 * would consume, and neither is used. */
#define ZIMAGE_ENCODER_USED   (ZIMAGE_ENCODER_LAYERS - 1)

/* Called once a layer, so a caller can show that something is happening
 * across the half-minute this takes. Optional. */
typedef void (*zimage_encode_tick)(int layer, int layers, void *context);

/* `out` must hold count * ZIMAGE_CAP_DIM floats. Returns 0 on failure. */
int zimage_encode(qwen_weights *encoder, const uint32_t *ids, int count,
                  float *out, zimage_encode_tick tick, void *tick_context,
                  char *error, size_t error_size);

/* The production path. Activations remain BF16 on Metal and one layer of
 * weights is uploaded at a time, so the 4B encoder does not have to coexist
 * with a second device-resident copy of the whole checkpoint. There is no
 * implicit CPU fallback: a Metal failure is returned to the caller. */
int zimage_encode_metal(qwen_weights *encoder, const char *shader_path,
                        const uint32_t *ids, int count, float *out,
                        zimage_encode_tick tick, void *tick_context,
                        char *error, size_t error_size);

#endif
