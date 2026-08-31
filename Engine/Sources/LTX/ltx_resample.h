#ifndef LTX_RESAMPLE_H
#define LTX_RESAMPLE_H

#include <stdint.h>

/* Resamples interleaved stereo input into channel-major output.
 *
 * LTX's vocoder emits [frame][channel], while the shared H3 AV writer accepts
 * [channel][frame]. Keeping that conversion in this named boundary prevents
 * either engine from silently changing the other's PCM contract again.
 *
 * The caller owns the returned allocation. Returns NULL for invalid input or
 * allocation/size failure. */
float *ltx_resample_stereo_to_channel_major(
    const float *interleaved, uint32_t input_frames, int input_rate,
    int output_rate, uint32_t *output_frames);

#endif
