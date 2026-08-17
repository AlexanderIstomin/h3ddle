/* Does the GPU decoder agree with the reference the CPU one already matches?
 *
 * Same golden as stage 3, so the two decoders are being held to one standard
 * rather than to each other — an agreement between two of my own
 * implementations would prove only that I made the same mistake twice.
 *
 *   ./zimage_vae_gpu_check <shaders.metal> <package/vae_decoder.safetensors> <golden>
 */
#include "zimage_vae_gpu.h"
#include "qwen_weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char problem[512];

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <shaders.metal> <vae_decoder.safetensors> "
                        "<golden.safetensors>\n", argv[0]);
        return 2;
    }
    qwen_weights *golden = qwen_weights_open(argv[3], problem, sizeof(problem));
    if (!golden) { fprintf(stderr, "golden: %s\n", problem); return 1; }

    int side = 0;
    {
        FILE *probe = fopen(argv[3], "rb");
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

    const int64_t latent_shape[4] = {1, 16, side, side};
    const float *latent = qwen_weights_f32(golden, "latent", 4, latent_shape,
                                           problem, sizeof(problem));
    if (!latent) { fprintf(stderr, "golden: %s\n", problem); return 1; }

    double began = now();
    zimage_vae_gpu *vae = zimage_vae_gpu_create(argv[1], argv[2], NULL, side,
                                                problem, sizeof(problem));
    if (!vae) { fprintf(stderr, "create: %s\n", problem); return 1; }
    printf("latent %dx%d -> image %dx%d, loaded in %.1f s\n\n",
           side, side, side * 8, side * 8, now() - began);

    const int pixels = side * 8;
    float *image = malloc((size_t)3 * pixels * pixels * sizeof(float));
    double elapsed = 0.0;
    for (int pass = 0; pass < 3; pass++) {   /* the first pays for pipelines */
        began = now();
        if (!zimage_vae_gpu_decode(vae, latent, side, image,
                                   problem, sizeof(problem))) {
            fprintf(stderr, "decode: %s\n", problem);
            return 1;
        }
        elapsed = now() - began;
        printf("  pass %d: %.3f s\n", pass, elapsed);
    }

    const int64_t image_shape[4] = {1, 3, pixels, pixels};
    const float *theirs = qwen_weights_f32(golden, "image", 4, image_shape,
                                           problem, sizeof(problem));
    if (!theirs) { fprintf(stderr, "golden: %s\n", problem); return 1; }
    double square = 0.0, total = 0.0, worst = 0.0;
    for (size_t index = 0; index < (size_t)3 * pixels * pixels; index++) {
        const double difference = (double)image[index] - (double)theirs[index];
        square += difference * difference;
        total += (double)theirs[index] * (double)theirs[index];
        if (fabs(difference) > worst) worst = fabs(difference);
    }
    const size_t count = (size_t)3 * pixels * pixels;
    printf("\n  image        RMS rel %9.2e   worst %9.2e\n",
           sqrt(square / (double)count) / sqrt(total / (double)count), worst);
    printf("  the CPU decoder takes 21 s at 256px and 80 s at 512px\n");

    free(image);
    zimage_vae_gpu_release(vae);
    qwen_weights_close(golden);
    return 0;
}
