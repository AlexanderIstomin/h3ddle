/* Does the lifted sampler still produce the same latents as the driver?
 *
 * Same bar as the other four checks -- the arithmetic did not change in the
 * move, so anything but identical bytes is the lift -- but this one earns it
 * differently. The other stages moved whole; here the geometry became a
 * runtime argument, so the comparison is only meaningful when the geometry
 * handed in matches the -D flags the driver was built with. Pass them on the
 * command line and hold the result against that build's own output:
 *
 *     ./ltx_dit_check $DIT fire.context.bin mine.bin 9 16 16 8 16
 *     cmp mine.bin fire.latent.bin
 *
 * 9 16 16 is `h3_ltx_generate_long`, which is 65 pixel frames of 512x512.
 *
 * build:
 *   clang -O2 -fobjc-arc -o ltx_dit_check \
 *       harness/ltx_dit_check.c ltx_dit.c ltx_audio.c \
 *       ../../Vendor/h3.c/h3_gpu.m ../../Vendor/h3.c/h3_weights.c \
 *       ../../Vendor/h3.c/h3_safetensors.c -I../../Vendor/h3.c -I. \
 *       -framework Foundation -framework Metal \
 *       -framework MetalPerformanceShaders \
 *       -framework MetalPerformanceShadersGraph -framework Accelerate -lm
 *
 * usage: ltx_dit_check DIT.safetensors CONTEXT.bin OUT.bin
 *                      FRAMES HEIGHT WIDTH [STEPS] [SEED] [FPS] */
#include "ltx_dit.h"
#include "ltx_audio.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CONTEXT_MAGIC 0x4C545854
#define LATENT_MAGIC  0x4C545847

static double now(void) {
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec * 1e-9;
}

/* The tick is asked between blocks so that a cancel lands inside one rather
 * than at the end of a step, which means it repeats a step's number 48 times.
 * Print only when it advances. */
static int report(int step, int steps, void *context) {
    static int seen = -1;
    double *last = context;
    if (step == seen) return 1;
    const double moment = now();
    if (seen >= 0)
        printf("  step %d/%d, %.1f s\n", step, steps, moment - *last);
    *last = moment;
    seen = step;
    fflush(stdout);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 7 || argc > 10) {
        fprintf(stderr, "usage: %s DIT.safetensors CONTEXT.bin OUT.bin "
                        "FRAMES HEIGHT WIDTH [STEPS] [SEED] [FPS]\n", argv[0]);
        return 2;
    }
    ltx_dit_request request = {0};
    request.frames = (uint32_t)strtoul(argv[4], NULL, 10);
    request.height = (uint32_t)strtoul(argv[5], NULL, 10);
    request.width = (uint32_t)strtoul(argv[6], NULL, 10);
    request.steps = argc > 7 ? atoi(argv[7]) : 8;
    request.seed = argc > 8 ? strtoull(argv[8], NULL, 10) : 20260817u;
    request.fps = argc > 9 ? atoi(argv[9]) : 24;

    /* Derived, not chosen: the audio length follows the video's duration.
     * Getting this from the same helper the decode uses is the point -- a
     * private copy here is how the two drift apart. */
    const int pixel_frames = 8 * ((int)request.frames - 1) + 1;
    request.audio_rows = ltx_audio_rows_for(pixel_frames, request.fps);

    char error[512];
    h3_weight_store *dit = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!dit) { fprintf(stderr, "cannot open the DiT: %s\n", error); return 1; }
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "cannot create Metal: %s\n", error); return 1; }

    uint32_t span = 0;
    float *video_context = NULL, *audio_context = NULL;
    {
        FILE *file = fopen(argv[2], "rb");
        if (!file) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
        uint32_t header[4];
        if (fread(header, sizeof(header), 1, file) != 1 ||
            header[0] != CONTEXT_MAGIC) {
            fprintf(stderr, "%s is not a context dump\n", argv[2]);
            return 1;
        }
        span = header[1];
        if (header[2] != LTX_DIT_VIDEO_DIM || header[3] != LTX_DIT_AUDIO_DIM) {
            fprintf(stderr, "the context is %u/%u wide, expected %d/%d\n",
                    header[2], header[3], LTX_DIT_VIDEO_DIM,
                    LTX_DIT_AUDIO_DIM);
            return 1;
        }
        video_context = malloc((size_t)span * LTX_DIT_VIDEO_DIM * sizeof(float));
        audio_context = malloc((size_t)span * LTX_DIT_AUDIO_DIM * sizeof(float));
        if (!video_context || !audio_context ||
            fread(video_context, sizeof(float),
                  (size_t)span * LTX_DIT_VIDEO_DIM, file) !=
                (size_t)span * LTX_DIT_VIDEO_DIM ||
            fread(audio_context, sizeof(float),
                  (size_t)span * LTX_DIT_AUDIO_DIM, file) !=
                (size_t)span * LTX_DIT_AUDIO_DIM) {
            fprintf(stderr, "cannot read the context\n");
            return 1;
        }
        fclose(file);
    }

    const size_t video_rows =
        (size_t)request.frames * request.height * request.width;
    float *video_latent = malloc(video_rows * LTX_DIT_LATENT * sizeof(float));
    float *audio_latent =
        malloc((size_t)request.audio_rows * LTX_DIT_LATENT * sizeof(float));
    if (!video_latent || !audio_latent) {
        fprintf(stderr, "cannot allocate the latents\n");
        return 1;
    }

    printf("LTX-2.5, %d steps, %zu video tokens (%ux%ux%u), %u audio, "
           "%u context\n", request.steps, video_rows, request.frames,
           request.height, request.width, request.audio_rows, span);
    printf("%u latent frames = %d pixel frames at %d fps = %.3f s; "
           "%u audio rows = %.3f s\n", request.frames, pixel_frames,
           request.fps, (double)pixel_frames / request.fps, request.audio_rows,
           (double)ltx_audio_frames_for(request.audio_rows) /
               LTX_AUDIO_SAMPLE_RATE);
    fflush(stdout);

    double last = now();
    const double began = last;
    if (!ltx_dit_sample(gpu, dit, &request, video_context, audio_context, span,
                        video_latent, audio_latent, report, &last, error,
                        sizeof(error))) {
        fprintf(stderr, error[0] ? "FAIL: %s\n" : "cancelled\n", error);
        return 1;
    }
    const double took = now() - began;
    free(video_context); free(audio_context);

    double peak = 0.0;
    for (size_t index = 0; index < video_rows * LTX_DIT_LATENT; index++) {
        const double magnitude = video_latent[index] < 0 ?
            -video_latent[index] : video_latent[index];
        if (magnitude > peak) peak = magnitude;
    }
    printf("%d steps of %d blocks in %.1f s (%.1f s a step), latent peak %.3f\n",
           request.steps, LTX_DIT_BLOCKS, took, took / request.steps, peak);

    /* The driver's latent format, so the two files compare byte for byte. */
    FILE *out = fopen(argv[3], "wb");
    if (!out) { fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    const uint32_t header[8] = {(uint32_t)LATENT_MAGIC, request.frames,
                                request.height, request.width, LTX_DIT_LATENT,
                                request.audio_rows, (uint32_t)request.steps, 0};
    fwrite(header, sizeof(header), 1, out);
    fwrite(video_latent, sizeof(float), video_rows * LTX_DIT_LATENT, out);
    fwrite(audio_latent, sizeof(float),
           (size_t)request.audio_rows * LTX_DIT_LATENT, out);
    if (fclose(out) != 0) { fprintf(stderr, "cannot close %s\n", argv[3]); return 1; }
    printf("wrote %s\n", argv[3]);

    free(video_latent); free(audio_latent);
    h3_gpu_free(gpu);
    h3_weight_store_free(dit);
    return 0;
}
