#include "ltx_resample.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

enum { LTX_RESAMPLE_TAPS = 32 };

float *ltx_resample_stereo_to_channel_major(
    const float *interleaved, uint32_t input_frames, int input_rate,
    int output_rate, uint32_t *output_frames) {
    if (!interleaved || !input_frames || input_rate <= 0 || output_rate <= 0 ||
        !output_frames) return NULL;

    const double ratio = (double)output_rate / (double)input_rate;
    const double scaled_frames = (double)input_frames * ratio;
    if (!isfinite(scaled_frames) || scaled_frames < 1.0 ||
        scaled_frames > (double)UINT32_MAX) return NULL;
    const uint32_t frames = (uint32_t)scaled_frames;
    if ((size_t)frames > SIZE_MAX / (2 * sizeof(float))) return NULL;

    float *channel_major = malloc((size_t)frames * 2 * sizeof(*channel_major));
    if (!channel_major) return NULL;
    for (uint32_t frame = 0; frame < frames; frame++) {
        const double at = (double)frame / ratio;
        const long centre = (long)floor(at);
        double left = 0.0, right = 0.0, weight = 0.0;
        for (long tap = -LTX_RESAMPLE_TAPS + 1; tap <= LTX_RESAMPLE_TAPS; tap++) {
            const long index = centre + tap;
            if (index < 0 || index >= (long)input_frames) continue;
            const double distance = at - (double)index;
            double kernel;
            if (distance == 0.0) {
                kernel = 1.0;
            } else {
                const double pi = 3.14159265358979323846;
                const double x = pi * distance;
                /* Hann over the tap span keeps the stopband from ringing
                 * across a transient. */
                const double window =
                    0.5 + 0.5 * cos(pi * distance / (double)LTX_RESAMPLE_TAPS);
                kernel = sin(x) / x * window;
            }
            left += kernel * (double)interleaved[(size_t)index * 2];
            right += kernel * (double)interleaved[(size_t)index * 2 + 1];
            weight += kernel;
        }
        /* Normalizing by the realized weight rather than trusting the kernel
         * sum keeps the first and last few frames from fading. */
        if (weight > 1e-9) {
            left /= weight;
            right /= weight;
        }
        channel_major[frame] = (float)left;
        channel_major[(size_t)frames + frame] = (float)right;
    }
    *output_frames = frames;
    return channel_major;
}
