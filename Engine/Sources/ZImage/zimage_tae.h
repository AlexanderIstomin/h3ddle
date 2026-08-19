/* Tiny live-preview decoder for Z-Image's FLUX-compatible latent. */
#ifndef ZIMAGE_TAE_H
#define ZIMAGE_TAE_H

#include <stddef.h>

typedef struct zimage_tae zimage_tae;

/* Loads madebyollin's TAEF1 decoder and sizes its Metal work buffers for one
 * fixed latent canvas. The decoder is intentionally separate from the full
 * image VAE: it stays resident during sampling, then is released before the
 * full-resolution decoder is opened. */
zimage_tae *zimage_tae_create(const char *shader_source_path,
                              const char *weight_path,
                              int latent_height, int latent_width,
                              char *error, size_t error_size);
void zimage_tae_release(zimage_tae *tae);

/* `latent` is channel-major [16, height, width] in the transformer's own
 * diffusion space. `rgb` receives interleaved [height*8, width*8, 3] floats
 * clamped to [0, 1]. */
int zimage_tae_decode(zimage_tae *tae, const float *latent, float *rgb,
                      char *error, size_t error_size);

#endif
