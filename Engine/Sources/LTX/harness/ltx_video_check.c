/* Does the lifted decoder still make the same frames as the driver?
 *
 * Same question as `ltx_audio_check.c`, same answer required: the arithmetic
 * did not change in the move, so anything but identical bytes is the lift.
 *
 * build:
 *   clang -O2 -fobjc-arc -o ltx_video_check \
 *       harness/ltx_video_check.c ltx_video.c \
 *       ../../Vendor/h3.c/h3_gpu.m ../../Vendor/h3.c/h3_weights.c \
 *       ../../Vendor/h3.c/h3_safetensors.c -I../../Vendor/h3.c -I. \
 *       -framework Foundation -framework Metal \
 *       -framework MetalPerformanceShaders \
 *       -framework MetalPerformanceShadersGraph -framework Accelerate -lm
 *
 * usage: ltx_video_check VAE.safetensors LATENT.bin OUT.bin */
#include "ltx_video.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s VAE.safetensors LATENT.bin OUT.bin\n",
                argv[0]);
        return 2;
    }
    char error[512];
    h3_weight_store *store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) { fprintf(stderr, "cannot open the VAE: %s\n", error); return 1; }
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "cannot create Metal: %s\n", error); return 1; }

    uint32_t frames = 0, height = 0, width = 0;
    float *latent = NULL;
    {
        FILE *file = fopen(argv[2], "rb");
        if (!file) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
        uint32_t header[8];
        if (fread(header, sizeof(header), 1, file) != 1 ||
            header[0] != UINT32_C(0x4C545847)) {
            fprintf(stderr, "%s is not a latent\n", argv[2]);
            return 1;
        }
        frames = header[1]; height = header[2]; width = header[3];
        if (header[4] != LTX_VIDEO_LATENT_CHANNELS) {
            fprintf(stderr, "the latent is %u wide, expected %d\n", header[4],
                    LTX_VIDEO_LATENT_CHANNELS);
            return 1;
        }
        const size_t count = (size_t)frames * height * width *
                             LTX_VIDEO_LATENT_CHANNELS;
        latent = malloc(count * sizeof(*latent));
        if (!latent || fread(latent, sizeof(*latent), count, file) != count) {
            fprintf(stderr, "cannot read the video latent\n");
            return 1;
        }
        fclose(file);
    }

    const uint32_t depth = ltx_video_pixel_frames(frames);
    const uint32_t side_h = height * LTX_VIDEO_SPATIAL;
    const uint32_t side_w = width * LTX_VIDEO_SPATIAL;
    const size_t count = (size_t)LTX_VIDEO_CHANNELS * depth * side_h * side_w;
    printf("latent [%u, %u, %u, %d] -> %u frames of %ux%u\n", frames, height,
           width, LTX_VIDEO_LATENT_CHANNELS, depth, side_w, side_h);

    float *pixels = malloc(count * sizeof(*pixels));
    if (!pixels) { fprintf(stderr, "cannot allocate the frames\n"); return 1; }
    if (!ltx_video_decode(gpu, store, latent, frames, height, width, pixels,
                          error, sizeof(error))) {
        fprintf(stderr, "FAIL: %s\n", error);
        return 1;
    }
    free(latent);

    double lo = pixels[0], hi = pixels[0], sum = 0.0;
    for (size_t index = 0; index < count; index++) {
        if (pixels[index] < lo) lo = pixels[index];
        if (pixels[index] > hi) hi = pixels[index];
        sum += pixels[index];
    }
    printf("  pixels: %.3f .. %.3f, mean %.3f (the VAE works in [-1, 1])\n",
           lo, hi, sum / (double)count);

    /* The driver's format, so the two files compare byte for byte. */
    FILE *out = fopen(argv[3], "wb");
    if (!out) { fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    const uint32_t header[5] = {UINT32_C(0x4C545846), LTX_VIDEO_CHANNELS,
                                depth, side_h, side_w};
    fwrite(header, sizeof(header), 1, out);
    fwrite(pixels, sizeof(*pixels), count, out);
    fclose(out);
    printf("wrote %s\n", argv[3]);

    free(pixels);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    return 0;
}
