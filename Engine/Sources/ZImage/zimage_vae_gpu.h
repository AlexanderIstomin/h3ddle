/* The Z-Image VAE decoder on the GPU.
 *
 * Once the DiT moved to the device this became the largest single cost in the
 * pipeline — 80 s against the whole eight-step DiT's 62 s at 512², and it is
 * the piece that decides whether 1024² is feasible at all.
 *
 * Almost all of it is existing engine ops, because H3's tiny autoencoder is
 * the same shape of conv decoder and already drives them. The two things to
 * know:
 *
 *  - activations are channel-last [height][width][channels] here, where the
 *    CPU decoder is channel-major. That transpose is the seam, and it is the
 *    same one the speech vocoder needed;
 *  - conv weights stay in the checkpoint's `[out, in, kh, kw]` layout. The
 *    kernel indexes them directly, so nothing is repacked.
 */
#ifndef ZIMAGE_VAE_GPU_H
#define ZIMAGE_VAE_GPU_H

#include "h3_gpu.h"
#include "h3_weights.h"

#include <stddef.h>

typedef struct zimage_vae_gpu zimage_vae_gpu;

/* `device` may be an existing context to share, or NULL to create one. */
zimage_vae_gpu *zimage_vae_gpu_create(const char *shaders, const char *decoder,
                                      h3_gpu *device, int max_height,
                                      int max_width,
                                      char *error, size_t error_size);
void zimage_vae_gpu_release(zimage_vae_gpu *vae);

/* `latent` is [16][height][width] and already unscaled; `image` receives
 * [3][height*8][width*8], both channel-major as the CPU path has them. The
 * two sides are independent; see the CPU header. */
int zimage_vae_gpu_decode(zimage_vae_gpu *vae, const float *latent,
                          int height, int width,
                          float *image, char *error, size_t error_size);

/* The same autoencoder the other way, for working from a picture. Built
 * separately because it holds different weights, and the maxima are the
 * *picture's* here where the decoder's are the latent's.
 *
 * `image` is [3][height][width] and `latent` receives [16][h/8][w/8], both
 * channel-major, matching the CPU pair exactly. Released with the same call. */
zimage_vae_gpu *zimage_vae_gpu_create_encoder(const char *shaders,
                                              const char *encoder,
                                              h3_gpu *device, int max_height,
                                              int max_width,
                                              char *error, size_t error_size);
int zimage_vae_gpu_encode(zimage_vae_gpu *vae, const float *image,
                          int image_height, int image_width, float *latent,
                          char *error, size_t error_size);

#endif
