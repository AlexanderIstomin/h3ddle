/* The Z-Image VAE decoder: 16 latent channels up to RGB, 8x on each side.
 *
 * f32 throughout and channel-major [channels][height][width], which is what
 * the CPU loop order wants. The engine's own convolutions are time-major and
 * will want the transpose the speech vocoder needed; that is a question for
 * the GPU port, not for whether the arithmetic is right.
 */
#ifndef ZIMAGE_VAE_H
#define ZIMAGE_VAE_H

#include "qwen_weights.h"

#include <stddef.h>

/* Latents leave the DiT scaled and shifted; the decoder wants them raw. */
#define ZIMAGE_VAE_SCALING  0.3611f
#define ZIMAGE_VAE_SHIFT    0.1159f

/* Called after each named stage when the caller wants to see inside — the
 * check harness compares against a reference, the generator passes NULL. */
typedef void (*zimage_vae_tap)(const char *stage, const float *values,
                               size_t count, void *context);

/* `latent` is [16][side][side] and already unscaled; `image` must hold
 * 3 * (side*8) * (side*8) floats, roughly in [-1, 1]. Returns 0 on failure. */
int zimage_vae_decode(qwen_weights *decoder, const float *latent, int side,
                      float *image, zimage_vae_tap tap, void *context);

#endif
