/* The Z-Image S3-DiT: 2 noise refiners, 2 context refiners, 30 trunk layers.
 *
 * Split into a setup and a per-step call because the caption half of the
 * sequence does not depend on the timestep or the latent: cap_embedder is a
 * norm and a linear, and context_refiner carries no adaLN at all, so both
 * produce the same rows at every one of the eight steps. They are computed
 * once here. The trunk *does* write through the caption rows, so the refined
 * copy is kept aside and restored at the head of each step.
 */
#ifndef ZIMAGE_DIT_H
#define ZIMAGE_DIT_H

#include "zimage_block.h"
#include "zimage_gpu.h"
#include "qwen_weights.h"

#include <stddef.h>

#define ZIMAGE_LATENT_CHANNELS 16
#define ZIMAGE_PATCH           2
#define ZIMAGE_SEQ_MULTIPLE    32
#define ZIMAGE_LAYERS          30
#define ZIMAGE_REFINERS        2

typedef void (*zimage_dit_tap)(const char *stage, const float *values,
                               size_t count, void *context);

typedef struct {
    qwen_weights *weights;
    int latent_height, latent_width;
    int tokens_high, tokens_wide, image_tokens;
    int caption_tokens, caption_padded, sequence;
    float *cosines, *sines;
    float *unified;           /* [sequence][DIM], image half then caption half */
    float *caption_refined;   /* [caption_padded][DIM], restored each step */
    float *patches, *head;
    zimage_scratch scratch;
    zimage_gpu *device;      /* NULL keeps everything on the CPU */
} zimage_dit;

/* `caption` is [caption_tokens][CAP_DIM] straight from the encoder. */
int zimage_dit_init(zimage_dit *dit, qwen_weights *weights,
                    int latent_height, int latent_width,
                    const float *caption, int caption_tokens,
                    zimage_gpu *device, char *error, size_t error_size);
void zimage_dit_release(zimage_dit *dit);

/* `latent` and `velocity` are both [16][height][width]; `timestep` is what the
 * sampler feeds the model, namely 1 - sigma. */
int zimage_dit_step(zimage_dit *dit, const float *latent, float timestep,
                    float *velocity, zimage_dit_tap tap, void *context);

#endif
