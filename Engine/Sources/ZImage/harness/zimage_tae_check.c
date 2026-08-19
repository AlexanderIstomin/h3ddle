/* Model-backed smoke test for the live TAEF1 decoder. The repository does not
 * carry weights, so this is run explicitly with the managed artifact path. */
#include "zimage_tae.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum { LATENT_HEIGHT = 16, LATENT_WIDTH = 16, LATENT_CHANNELS = 16 };

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s TAEF1_SAFETENSORS H3_SHADERS_METAL\n",
                argv[0]);
        return 2;
    }
    char error[512];
    zimage_tae *tae = zimage_tae_create(
        argv[2], argv[1], LATENT_HEIGHT, LATENT_WIDTH, error, sizeof(error));
    if (!tae) {
        fprintf(stderr, "FAIL load: %s\n", error);
        return 1;
    }
    const size_t latent_values =
        (size_t)LATENT_CHANNELS * LATENT_HEIGHT * LATENT_WIDTH;
    float *latent = malloc(latent_values * sizeof(*latent));
    const size_t rgb_values =
        (size_t)LATENT_HEIGHT * 8 * LATENT_WIDTH * 8 * 3;
    float *rgb = malloc(rgb_values * sizeof(*rgb));
    if (!latent || !rgb) {
        fprintf(stderr, "FAIL: out of memory\n");
        return 1;
    }
    unsigned state = 20260819u;
    for (size_t index = 0; index < latent_values; index++) {
        state = state * 1103515245u + 12345u;
        latent[index] = ((float)((state >> 16) & 0x7fffu) / 16383.5f) - 1.0f;
    }
    if (!zimage_tae_decode(tae, latent, rgb, error, sizeof(error))) {
        fprintf(stderr, "FAIL decode: %s\n", error);
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
    printf("decoded 128x128, mean %.4f, range [%.4f, %.4f]\n",
           sum / (double)rgb_values, (double)low, (double)high);
    free(rgb);
    free(latent);
    zimage_tae_release(tae);
    puts("ok: TAEF1 decodes a Z-Image latent end to end");
    return 0;
}
