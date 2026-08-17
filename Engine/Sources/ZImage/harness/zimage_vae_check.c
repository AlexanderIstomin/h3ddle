/* Stage 3: the VAE decoder, sixteen latent channels up to RGB.
 *
 * f32 throughout, and checked at f32: this is a convolution stack, not a
 * transformer carrying bf16 weights, so "within tolerance" is not the claim —
 * it should agree to rounding the way the speech codec's vocoder did.
 *
 * The decoder itself lives in zimage_vae.c and is the same code the generator
 * runs; this file only supplies a tap that compares each stage against the
 * reference. Keeping the comparison outside the implementation is what stops
 * the checked path and the shipped path from drifting apart.
 *
 *   ./zimage_vae_check <package-dir> <golden.safetensors>
 */
#include "zimage_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static qwen_weights *reference;
static char problem[512];

static void compare(const char *stage, const float *ours, size_t count,
                    void *context) {
    (void)context;
    /* A stage the reference did not capture is simply not compared. */
    int64_t shape[4] = {-1, -1, -1, -1};
    const float *theirs = qwen_weights_f32(reference, stage, 4, shape,
                                           problem, sizeof(problem));
    if (!theirs) return;
    double square = 0.0, total = 0.0, worst = 0.0;
    for (size_t index = 0; index < count; index++) {
        const double difference = (double)ours[index] - (double)theirs[index];
        square += difference * difference;
        total += (double)theirs[index] * (double)theirs[index];
        if (fabs(difference) > worst) worst = fabs(difference);
    }
    printf("  %-12s RMS rel %9.2e   worst %9.2e\n", stage,
           sqrt(square / (double)count) / sqrt(total / (double)count), worst);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <package-dir> <golden.safetensors>\n", argv[0]);
        return 2;
    }
    char path[1200];
    snprintf(path, sizeof(path), "%s/vae_decoder.safetensors", argv[1]);
    qwen_weights *decoder = qwen_weights_open(path, problem, sizeof(problem));
    if (!decoder) { fprintf(stderr, "decoder: %s\n", problem); return 1; }
    reference = qwen_weights_open(argv[2], problem, sizeof(problem));
    if (!reference) { fprintf(stderr, "golden: %s\n", problem); return 1; }

    int64_t latent_shape[4] = {1, 16, -1, -1};
    const float *latent = qwen_weights_f32(reference, "latent", 4, latent_shape,
                                           problem, sizeof(problem));
    if (!latent) { fprintf(stderr, "golden: %s\n", problem); return 1; }

    /* The latent is square and its side sets everything downstream. */
    int side = 0;
    {
        FILE *probe = fopen(argv[2], "rb");
        uint64_t header_bytes = 0;
        if (!probe || fread(&header_bytes, 8, 1, probe) != 1) return 1;
        char *header = malloc(header_bytes + 1);
        if (fread(header, 1, header_bytes, probe) != header_bytes) return 1;
        header[header_bytes] = '\0';
        fclose(probe);
        const char *entry = strstr(header, "\"latent\"");
        const char *shape = entry ? strstr(entry, "\"shape\":[1,16,") : NULL;
        if (shape) side = atoi(shape + strlen("\"shape\":[1,16,"));
        free(header);
    }
    if (side < 1) { fprintf(stderr, "cannot read the latent size\n"); return 1; }
    printf("latent %d x %d x 16 -> image %d x %d x 3\n\n", side, side,
           side * 8, side * 8);

    float *image = malloc((size_t)3 * (side * 8) * (side * 8) * sizeof(float));
    if (!image) { fprintf(stderr, "out of memory\n"); return 1; }
    if (!zimage_vae_decode(decoder, latent, side, image, compare, NULL)) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    free(image);
    qwen_weights_close(decoder);
    qwen_weights_close(reference);
    return 0;
}
