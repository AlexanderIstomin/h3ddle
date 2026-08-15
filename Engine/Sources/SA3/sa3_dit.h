#ifndef SA3_DIT_H
#define SA3_DIT_H

#include <stddef.h>

/* The Stable Audio 3 Small transformer: twenty blocks of self-attention over
 * the latent sequence, cross-attention onto the text and duration
 * conditioning, and adaptive modulation driven by the timestep.
 *
 * It predicts a velocity for rectified flow, so a sampler calls it once per
 * step. Sixty-four learned memory tokens ride at the front of the sequence and
 * are dropped again on the way out — they give the blocks somewhere to keep
 * working state that does not belong to any latent position. */

typedef struct sa3_dit sa3_dit;

#define SA3_DIT_CHANNELS 256   /* latent channels in and out */
#define SA3_DIT_CONTEXT 768    /* width of the conditioning the encoder emits */

sa3_dit *sa3_dit_load(const char *path, char *error, size_t error_size);
void sa3_dit_free(sa3_dit *dit);

/* One denoising evaluation.
 *
 * latents  [256, frames]        channel-major
 * context  [context_tokens, 768] the text embeddings with the duration token
 * global   [768]                 the duration embedding on its own
 * velocity [256, frames]         channel-major, written by the call
 *
 * `sigma` is the current noise level, which the model sees as its timestep. */
int sa3_dit_forward(sa3_dit *dit, const float *latents, int frames,
                    const float *context, int context_tokens,
                    const float *global, float sigma, float *velocity,
                    char *error, size_t error_size);

#endif
