#ifndef SA3_DECODER_H
#define SA3_DECODER_H

#include <stddef.h>

/* SAME-S: the decoder half of Stable Audio 3's autoencoder.
 *
 * It turns 256-channel latents into 512-channel audio patches, which unpack
 * to 44.1 kHz stereo at 4096 samples per latent. Unlike the video VAE this
 * sits beside, it is a transformer rather than a convolutional stack: six
 * blocks of differential attention over a sequence built by pairing every
 * latent with sixteen learned placeholder tokens, so upsampling happens by
 * attention rather than by transposed convolution.
 *
 * Blocks 0-2 read windows of 34 tokens; blocks 3-5 read the same windows
 * shifted by half a window, so seams in the first half land mid-window in the
 * second. That shift is why the latent count must be even. */

typedef struct sa3_decoder sa3_decoder;

/* Latent frames per second is 44100/4096; a latent covers 4096 samples. */
#define SA3_SAMPLES_PER_LATENT 4096
#define SA3_PATCHES_PER_LATENT 16
#define SA3_PATCH_CHANNELS 512
#define SA3_LATENT_CHANNELS 256

sa3_decoder *sa3_decoder_load(const char *path, char *error, size_t error_size);
void sa3_decoder_free(sa3_decoder *decoder);

/* latents is [256, frames] channel-major; patches receives
 * [512, frames * 16], also channel-major. `frames` must be even. */
int sa3_decoder_run(sa3_decoder *decoder, const float *latents, int frames,
                    float *patches, char *error, size_t error_size);

/* Decodes long inputs in overlapping windows so memory stays flat and every
 * model call sees real latents rather than zero padding. `chunk + 2 * overlap`
 * must be even, and falls back to sa3_decoder_run when the input fits in one
 * window. */
int sa3_decoder_run_chunked(sa3_decoder *decoder, const float *latents,
                            int frames, int chunk, int overlap, float *patches,
                            char *error, size_t error_size);

/* Unpacks [512, patches] into interleaved-free stereo [2, patches * 256]. */
void sa3_decoder_unpatch(const float *patches, int patch_count, float *audio);

#endif
