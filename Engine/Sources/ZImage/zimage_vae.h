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

/* `latent` is [16][height][width] and already unscaled; `image` must hold
 * 3 * (height*8) * (width*8) floats, roughly in [-1, 1].
 *
 * The two sides are independent: this stack is a convolution tower and was
 * always rectangular inside, so a square was a restriction of the entry point
 * rather than of the arithmetic. Returns 0 on failure. */
int zimage_vae_decode(qwen_weights *decoder, const float *latent,
                      int height, int width,
                      float *image, zimage_vae_tap tap, void *context);

#define ZIMAGE_VAE_LATENT_CHANNELS 16
/* The reverse. Note the side it is given is the *picture's*, where decode is
 * given the latent's — each takes the side of the thing it is handed, and
 * naming them apart is cheaper than a comment nobody reads at the call site.
 *
 * `image` is [3][image_side][image_side] roughly in [-1, 1]; `latent`
 * receives [16][image_side/8][image_side/8], raw — the caller applies the
 * scale and shift, exactly as it removes them before decoding.
 * `image_side` must be a multiple of 8. Returns 0 on failure. */
int zimage_vae_encode(qwen_weights *encoder, const float *image,
                      int image_height, int image_width,
                      float *latent, zimage_vae_tap tap, void *context);

#endif
