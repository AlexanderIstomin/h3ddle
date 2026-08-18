/* LTX-2.5's video VAE decoder: a latent to frames.
 *
 * 128 latent channels out to RGB, expanding 8x in time and 32x in space -- the
 * last 4x of which is a patch unpack on the host rather than a convolution.
 */
#ifndef LTX_VIDEO_H
#define LTX_VIDEO_H

#include "h3_gpu.h"
#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>

#define LTX_VIDEO_LATENT_CHANNELS 128
/* Spatial factor from latent cell to pixel: three doublings and a 4x patch. */
#define LTX_VIDEO_SPATIAL 32
#define LTX_VIDEO_CHANNELS 3

/* A causal stack cannot produce the seven leading frames, so n latent frames
 * decode to 8(n-1)+1 rather than 8n. */
uint32_t ltx_video_pixel_frames(uint32_t latent_frames);

/* `latent` is [frames][height][width][128] channels-last, in the normalized
 * space the DiT works in -- the layout `h3_ltx_generate` writes.
 *
 * `pixels` receives `3 * ltx_video_pixel_frames(frames) * height * 32 *
 * width * 32` floats, **channel-major** and in [-1, 1]: the range the VAE
 * works in, clipped by the caller rather than rescaled, so a decode that
 * drifted out of range shows up instead of being normalized away.
 *
 * Returns 0 with `error` set on failure. */
int ltx_video_decode(h3_gpu *gpu, const h3_weight_store *store,
                     const float *latent, uint32_t frames, uint32_t height,
                     uint32_t width, float *pixels,
                     char *error, size_t error_size);

#endif
