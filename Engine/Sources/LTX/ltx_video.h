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

/* Decoder work completed across all tiles. Returning zero cancels; the decoder
 * then returns zero with an empty error, matching the pipeline callback. */
typedef int (*ltx_video_tick)(int completed, int total, void *context);

/* Frames to a latent: the inverse of the decode below, and what turns a
 * reference picture into something the DiT can be conditioned on.
 *
 * `pixels` is `3 * frames * height * width` floats, **channel-major** and in
 * [-1, 1] — the layout `ltx_video_decode` produces, so a decode can be fed
 * straight back in. `frames` must be 1 or 8k+1, and both sides a multiple of
 * `LTX_VIDEO_SPATIAL`.
 *
 * `latent` receives `latent_frames * (height/32) * (width/32) * 128` floats,
 * channels-last, already normalized into the space the DiT works in — where
 * `latent_frames` is 1 for a single frame and `(frames - 1) / 8 + 1` otherwise.
 *
 * Returns 0 with `error` set on failure. */
int ltx_video_encode(h3_gpu *gpu, const h3_weight_store *store,
                     const float *pixels, uint32_t frames, uint32_t height,
                     uint32_t width, float *latent,
                     char *error, size_t error_size);

/* `latent` is [frames][height][width][128] channels-last, in the normalized
 * space the DiT works in -- the layout `h3_ltx_generate` writes.
 *
 * `pixels` receives `3 * ltx_video_pixel_frames(frames) * height * 32 *
 * width * 32` floats, **channel-major** and in [-1, 1]: the range the VAE
 * works in, clipped by the caller rather than rescaled, so a decode that
 * drifted out of range shows up instead of being normalized away.
 *
 * Larger clips are decoded as overlapping 80-frame / 768-pixel tiles. The
 * caller still supplies one output buffer, but transient VAE activations are
 * bounded by the tile rather than the complete clip.
 *
 * Returns 0 with `error` set on failure. */
int ltx_video_decode(h3_gpu *gpu, const h3_weight_store *store,
                     const float *latent, uint32_t frames, uint32_t height,
                     uint32_t width, float *pixels,
                     char *error, size_t error_size);

/* The same decode with progress after every actual decoder operation. The
 * counter spans every tile and never restarts, which gives the app both an ETA
 * basis and cancellation points during a long VAE. */
int ltx_video_decode_progress(h3_gpu *gpu, const h3_weight_store *store,
                              const float *latent, uint32_t frames,
                              uint32_t height, uint32_t width, float *pixels,
                              ltx_video_tick tick, void *tick_context,
                              char *error, size_t error_size);

#endif
