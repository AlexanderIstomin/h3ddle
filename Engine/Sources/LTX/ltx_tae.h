/* Tiny still-preview decoder for LTX-2.5's 128-channel video latent. */
#ifndef LTX_TAE_H
#define LTX_TAE_H

#include "h3_gpu.h"
#include "ltx_tae_geometry.h"

#include <stddef.h>

typedef struct ltx_tae ltx_tae;

/* Loads madebyollin's taeltx2_3 decoder into an existing LTX Metal context.
 * The decoder is sized for one latent frame and deliberately disables its
 * temporal growers: a denoising preview needs one representative still, not
 * eight causally expanded frames. Spatial expansion remains exact, producing
 * latent_height * 32 by latent_width * 32 RGB. */
ltx_tae *ltx_tae_create(h3_gpu *gpu, const char *weight_path,
                        int latent_height, int latent_width,
                        char *error, size_t error_size);
void ltx_tae_release(ltx_tae *tae);

/* `latent` is interleaved [height, width, 128], matching the DiT's row-major
 * output. `rgb` receives interleaved full-canvas [height*32, width*32, 3]
 * floats clamped to [0, 1]. */
int ltx_tae_decode(ltx_tae *tae, const float *latent, float *rgb,
                   char *error, size_t error_size);

#endif
