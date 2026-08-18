/* LTX-2.5's audio half: a DiT audio latent to a 16 kHz stereo waveform.
 *
 * Unpatchify, denormalize, the 2D VAE decoder over log-mel, then BigVGAN.
 * Every kernel it needs already exists in h3_gpu — including the alias-free
 * snakebeta, which matches LTX's convention exactly rather than merely
 * closely: 12 taps, ratio 2, replicate padding on both the source and the
 * doubled signal, and the same 1e-9 on beta.
 */
#ifndef LTX_AUDIO_H
#define LTX_AUDIO_H

#include "h3_gpu.h"
#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>

/* The DiT's audio token width: 8 z-channels x 16 latent mel bins. */
#define LTX_AUDIO_PATCHED 128
#define LTX_AUDIO_SAMPLE_RATE 16000
#define LTX_AUDIO_HOP 160

/* How many latent rows a clip of this length wants.
 *
 * `AudioLatentShape.from_video_pixel_shape`: the video's duration in seconds
 * times `sample_rate / hop_length / downsample` = 25 rows per second, rounded.
 * This is derived, not chosen -- getting it wrong desynchronizes the streams
 * rather than merely truncating one. */
uint32_t ltx_audio_rows_for(int pixel_frames, int fps);

/* Samples per channel from `rows` latent rows: each row is 4 mel frames less
 * the 3 the causal stack cannot produce, each mel frame `LTX_AUDIO_HOP`. */
uint32_t ltx_audio_frames_for(uint32_t rows);

/* `tokens` is [rows][LTX_AUDIO_PATCHED] as the DiT leaves it, in the
 * normalized latent space. `samples` receives
 * `2 * ltx_audio_frames_for(rows)` interleaved floats in [-1, 1].
 *
 * `store` must hold the audio VAE checkpoint, which carries the decoder, the
 * vocoder and the per-channel statistics together.
 *
 * Returns 0 with `error` set on failure. */
int ltx_audio_decode(h3_gpu *gpu, const h3_weight_store *store,
                     const float *tokens, uint32_t rows, float *samples,
                     char *error, size_t error_size);

#endif
