/* Model-backed smoke test for the LTX tiny live-preview decoder. The
 * repository does not carry weights, so this runs explicitly with the pinned
 * managed artifact (or the same file downloaded to a temporary directory). */
#include "ltx_tae.h"

#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum { LATENT_HEIGHT = 15, LATENT_WIDTH = 27, LATENT_CHANNELS = 128 };

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s TAELTX_SAFETENSORS H3_SHADERS_METAL\n",
                argv[0]);
        return 2;
    }
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(argv[2], error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "FAIL GPU: %s\n", error);
        return 1;
    }
    ltx_tae *tae = ltx_tae_create(
        gpu, argv[1], LATENT_HEIGHT, LATENT_WIDTH, error, sizeof(error));
    if (!tae) {
        fprintf(stderr, "FAIL load: %s\n", error);
        h3_gpu_free(gpu);
        return 1;
    }
    const size_t latent_values =
        (size_t)LATENT_HEIGHT * LATENT_WIDTH * LATENT_CHANNELS;
    const size_t rgb_values =
        (size_t)LATENT_HEIGHT * 32 * LATENT_WIDTH * 32 * 3;
    float *latent = malloc(latent_values * sizeof(*latent));
    float *rgb = malloc(rgb_values * sizeof(*rgb));
    if (!latent || !rgb) {
        fprintf(stderr, "FAIL: out of memory\n");
        free(latent); free(rgb);
        ltx_tae_release(tae);
        h3_gpu_free(gpu);
        return 1;
    }
    unsigned state = 20260831u;
    for (size_t index = 0; index < latent_values; index++) {
        state = state * 1103515245u + 12345u;
        latent[index] = ((float)((state >> 16) & 0x7fffu) / 16383.5f) - 1.0f;
    }
    if (!ltx_tae_decode(tae, latent, rgb, error, sizeof(error))) {
        fprintf(stderr, "FAIL decode: %s\n", error);
        free(latent); free(rgb);
        ltx_tae_release(tae);
        h3_gpu_free(gpu);
        return 1;
    }
    float low = 1.0f, high = 0.0f;
    double sum = 0.0;
    for (size_t index = 0; index < rgb_values; index++) {
        const float value = rgb[index];
        if (!isfinite(value) || value < 0.0f || value > 1.0f) {
            fprintf(stderr, "FAIL: pixel %zu out of range (%f)\n", index,
                    (double)value);
            return 1;
        }
        if (value < low) low = value;
        if (value > high) high = value;
        sum += value;
    }
    if (high - low < 1e-6f) {
        fprintf(stderr, "FAIL: constant output\n");
        return 1;
    }
    printf("decoded %dx%d, mean %.4f, range [%.4f, %.4f]\n",
           LATENT_WIDTH * 32, LATENT_HEIGHT * 32,
           sum / (double)rgb_values, (double)low, (double)high);
    free(latent); free(rgb);
    ltx_tae_release(tae);
    h3_gpu_free(gpu);
    puts("ok: taeltx2_3 decodes an LTX-2.5 still preview end to end");
    return 0;
}
