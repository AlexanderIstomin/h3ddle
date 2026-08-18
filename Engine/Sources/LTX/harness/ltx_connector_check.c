/* Does the lifted connector still produce the same context as the driver?
 *
 * Same question and same bar as the other two checks: the arithmetic did not
 * change in the move, so anything but identical bytes is the lift.
 *
 * build:
 *   clang -O2 -fobjc-arc -o ltx_connector_check \
 *       harness/ltx_connector_check.c ltx_connector.c \
 *       ../../Vendor/h3.c/h3_gpu.m ../../Vendor/h3.c/h3_weights.c \
 *       ../../Vendor/h3.c/h3_safetensors.c -I../../Vendor/h3.c -I. \
 *       -framework Foundation -framework Metal \
 *       -framework MetalPerformanceShaders \
 *       -framework MetalPerformanceShadersGraph -framework Accelerate -lm
 *
 * usage: ltx_connector_check DIT.safetensors STATES.bin OUT.bin */
#include "ltx_connector.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define STATES_MAGIC 0x4C545854

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s DIT.safetensors STATES.bin OUT.bin\n",
                argv[0]);
        return 2;
    }
    char error[512];
    h3_weight_store *dit = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!dit) { fprintf(stderr, "cannot open the DiT: %s\n", error); return 1; }
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "cannot create Metal: %s\n", error); return 1; }

    /* The tower's dump: header, ids, the hidden states, then the two
     * aggregated feature streams, which are the only part wanted here. */
    uint32_t tokens = 0;
    float *video_features = NULL, *audio_features = NULL;
    {
        FILE *file = fopen(argv[2], "rb");
        if (!file) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
        uint32_t header[6];
        if (fread(header, sizeof(header), 1, file) != 1 ||
            header[0] != STATES_MAGIC) {
            fprintf(stderr, "%s is not a state dump\n", argv[2]);
            return 1;
        }
        tokens = header[1];
        const uint32_t states = header[2], hidden = header[3];
        if (header[4] != LTX_CONNECTOR_VIDEO_DIM ||
            header[5] != LTX_CONNECTOR_AUDIO_DIM) {
            fprintf(stderr, "the dump aggregates to %u/%u, expected %d/%d\n",
                    header[4], header[5], LTX_CONNECTOR_VIDEO_DIM,
                    LTX_CONNECTOR_AUDIO_DIM);
            return 1;
        }
        fseek(file, (long)((size_t)tokens * sizeof(int32_t) +
                           (size_t)states * tokens * hidden * sizeof(float)),
              SEEK_CUR);
        video_features = malloc((size_t)tokens * LTX_CONNECTOR_VIDEO_DIM *
                                sizeof(float));
        audio_features = malloc((size_t)tokens * LTX_CONNECTOR_AUDIO_DIM *
                                sizeof(float));
        if (!video_features || !audio_features ||
            fread(video_features, sizeof(float),
                  (size_t)tokens * LTX_CONNECTOR_VIDEO_DIM, file) !=
                (size_t)tokens * LTX_CONNECTOR_VIDEO_DIM ||
            fread(audio_features, sizeof(float),
                  (size_t)tokens * LTX_CONNECTOR_AUDIO_DIM, file) !=
                (size_t)tokens * LTX_CONNECTOR_AUDIO_DIM) {
            fprintf(stderr, "cannot read the aggregated features\n");
            return 1;
        }
        fclose(file);
    }

    /* The span is the tokenizer's, and every dump so far is already padded to
     * it -- the registers then take no slots at all. Keeping them separate is
     * the point: reading the register count as the span is the bug this whole
     * stage is annotated about. */
    const uint32_t span = tokens;
    printf("%u tokens, span %u, %d registers available\n", tokens, span,
           LTX_CONNECTOR_REGISTERS);

    float *video = malloc((size_t)span * LTX_CONNECTOR_VIDEO_DIM * sizeof(float));
    float *audio = malloc((size_t)span * LTX_CONNECTOR_AUDIO_DIM * sizeof(float));
    if (!video || !audio) { fprintf(stderr, "cannot allocate the context\n"); return 1; }
    if (!ltx_connector_run(gpu, dit, video_features, audio_features, tokens,
                           span, video, audio, error, sizeof(error))) {
        fprintf(stderr, "FAIL: %s\n", error);
        return 1;
    }
    free(video_features); free(audio_features);

    double vp = 0.0, ap = 0.0;
    for (size_t i = 0; i < (size_t)span * LTX_CONNECTOR_VIDEO_DIM; i++)
        if (video[i] > vp || -video[i] > vp) vp = video[i] < 0 ? -video[i] : video[i];
    for (size_t i = 0; i < (size_t)span * LTX_CONNECTOR_AUDIO_DIM; i++)
        if (audio[i] > ap || -audio[i] > ap) ap = audio[i] < 0 ? -audio[i] : audio[i];
    printf("  video peak %.4f, audio peak %.4f\n", vp, ap);

    /* The driver's context format, so the two files compare byte for byte. */
    FILE *out = fopen(argv[3], "wb");
    if (!out) { fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    const uint32_t header[4] = {(uint32_t)STATES_MAGIC, span,
                                LTX_CONNECTOR_VIDEO_DIM,
                                LTX_CONNECTOR_AUDIO_DIM};
    fwrite(header, sizeof(header), 1, out);
    fwrite(video, sizeof(float), (size_t)span * LTX_CONNECTOR_VIDEO_DIM, out);
    fwrite(audio, sizeof(float), (size_t)span * LTX_CONNECTOR_AUDIO_DIM, out);
    fclose(out);
    printf("wrote %s\n", argv[3]);
    free(video); free(audio);
    h3_gpu_free(gpu);
    h3_weight_store_free(dit);
    return 0;
}
