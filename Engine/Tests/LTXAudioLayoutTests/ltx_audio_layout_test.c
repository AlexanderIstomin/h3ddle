#include "ltx_resample.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void require(int condition, const char *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
}

static void require_near(float actual, float expected, const char *message) {
    require(fabsf(actual - expected) < 1e-5f, message);
}

int main(void) {
    /* LTX emits interleaved stereo. Distinct channels make a layout mistake
     * visible even when centered generated speech would otherwise mask it. */
    const float interleaved[] = {
        0.10f, -0.10f,
        0.25f, -0.30f,
        0.50f, -0.60f,
        0.75f, -0.90f,
    };
    const uint32_t input_frames = 4;
    uint32_t output_frames = 0;
    float *pcm = ltx_resample_stereo_to_channel_major(
        interleaved, input_frames, 16000, 48000, &output_frames);
    require(pcm != NULL, "the LTX soundtrack resamples");
    require(output_frames == input_frames * 3,
            "16 kHz becomes the same-duration 48 kHz soundtrack");

    /* Every third output frame lands exactly on an input frame. The first
     * plane must contain only left samples and the second only right samples,
     * matching the unchanged H3 AV writer contract. */
    for (uint32_t frame = 0; frame < input_frames; frame++) {
        const uint32_t output = frame * 3;
        require_near(pcm[output], interleaved[(size_t)frame * 2],
                     "left samples remain in the left plane");
        require_near(pcm[(size_t)output_frames + output],
                     interleaved[(size_t)frame * 2 + 1],
                     "right samples remain in the right plane");
    }
    free(pcm);

    output_frames = 123;
    require(ltx_resample_stereo_to_channel_major(
                NULL, input_frames, 16000, 48000, &output_frames) == NULL,
            "invalid input is refused");

    printf("ok: LTX audio resampling preserves duration and writes channel-major PCM\n");
    return 0;
}
