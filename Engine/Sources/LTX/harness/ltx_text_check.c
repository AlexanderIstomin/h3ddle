/* Does the lifted tower still produce the same features as the driver?
 *
 * Unlike the other three checks this one compares in process rather than by
 * file: the module returns the two aggregated feature streams, where the
 * driver's dump also carries the 49 hidden states it no longer exposes. The
 * features are what the connector consumes, so they are what has to match --
 * and the bar is the same, exact equality, because the arithmetic did not
 * change in the move.
 *
 * The ids come from the driver's own dump, so the comparison starts from
 * identical tokens rather than from a tokenizer run twice.
 *
 * build:
 *   clang -O2 -fobjc-arc -o ltx_text_check \
 *       harness/ltx_text_check.c ltx_text.c \
 *       ../../Vendor/h3.c/h3_gpu.m ../../Vendor/h3.c/h3_weights.c \
 *       ../../Vendor/h3.c/h3_safetensors.c -I../../Vendor/h3.c -I. \
 *       -framework Foundation -framework Metal \
 *       -framework MetalPerformanceShaders \
 *       -framework MetalPerformanceShadersGraph -framework Accelerate -lm
 *
 * usage: ltx_text_check ENCODER.safetensors STATES.bin */
#include "ltx_text.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec * 1e-9;
}

#define STATES_MAGIC 0x4C545854

static int tick(int layer, int layers, void *context) {
    (void)context;
    if (layer % 12 == 0) { printf("    %2d/%d layers\n", layer, layers); fflush(stdout); }
    return 1;
}

static double compare(const char *what, const float *have, const float *want,
                      size_t count) {
    double worst = 0.0;
    size_t differing = 0;
    for (size_t index = 0; index < count; index++) {
        const double delta = fabs((double)have[index] - (double)want[index]);
        if (delta > worst) worst = delta;
        if (have[index] != want[index]) differing++;
    }
    printf("  %-6s %zu values, %zu differ, worst %.3e\n", what, count,
           differing, worst);
    return worst;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s ENCODER.safetensors STATES.bin\n", argv[0]);
        return 2;
    }
    char error[512];
    h3_weight_store *encoder = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!encoder) { fprintf(stderr, "cannot open the encoder: %s\n", error); return 1; }
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "cannot create Metal: %s\n", error); return 1; }

    uint32_t tokens = 0;
    int32_t *ids = NULL;
    float *want_video = NULL, *want_audio = NULL;
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
        if (states != LTX_TEXT_STATES || header[4] != LTX_TEXT_VIDEO_DIM ||
            header[5] != LTX_TEXT_AUDIO_DIM) {
            fprintf(stderr, "the dump is %u states aggregating to %u/%u\n",
                    states, header[4], header[5]);
            return 1;
        }
        ids = malloc((size_t)tokens * sizeof(*ids));
        if (!ids || fread(ids, sizeof(*ids), tokens, file) != tokens) {
            fprintf(stderr, "cannot read the token ids\n");
            return 1;
        }
        fseek(file, (long)((size_t)states * tokens * hidden * sizeof(float)),
              SEEK_CUR);
        want_video = malloc((size_t)tokens * LTX_TEXT_VIDEO_DIM * sizeof(float));
        want_audio = malloc((size_t)tokens * LTX_TEXT_AUDIO_DIM * sizeof(float));
        if (!want_video || !want_audio ||
            fread(want_video, sizeof(float),
                  (size_t)tokens * LTX_TEXT_VIDEO_DIM, file) !=
                (size_t)tokens * LTX_TEXT_VIDEO_DIM ||
            fread(want_audio, sizeof(float),
                  (size_t)tokens * LTX_TEXT_AUDIO_DIM, file) !=
                (size_t)tokens * LTX_TEXT_AUDIO_DIM) {
            fprintf(stderr, "cannot read the driver's features\n");
            return 1;
        }
        fclose(file);
    }
    printf("%u tokens, %d layers, %d states\n", tokens, LTX_TEXT_LAYERS,
           LTX_TEXT_STATES);

    float *video = malloc((size_t)tokens * LTX_TEXT_VIDEO_DIM * sizeof(float));
    float *audio = malloc((size_t)tokens * LTX_TEXT_AUDIO_DIM * sizeof(float));
    if (!video || !audio) { fprintf(stderr, "cannot allocate features\n"); return 1; }
    const double began = now();
    if (!ltx_text_encode(gpu, encoder, ids, tokens, video, audio, tick, NULL,
                         error, sizeof(error))) {
        fprintf(stderr, "FAIL: %s\n", error[0] ? error : "cancelled");
        return 1;
    }
    printf("  tower ran in %.1f s\n", now() - began);

    const double v = compare("video", video, want_video,
                             (size_t)tokens * LTX_TEXT_VIDEO_DIM);
    const double a = compare("audio", audio, want_audio,
                             (size_t)tokens * LTX_TEXT_AUDIO_DIM);
    const int identical = v == 0.0 && a == 0.0;
    printf("%s\n", identical ? "IDENTICAL -- the lift changed nothing"
                             : "DIFFERS -- the lift changed something");

    free(ids); free(video); free(audio); free(want_video); free(want_audio);
    h3_gpu_free(gpu);
    h3_weight_store_free(encoder);
    return identical ? 0 : 1;
}
