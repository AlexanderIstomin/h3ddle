/* Does the device encoder agree with the CPU one, and how much faster is it?
 *
 * The CPU encoder is the verified path — it inverts a decoder that was checked
 * stage by stage against the reference, and a round trip through the pair
 * reconstructs at 32 dB. So this compares against it directly rather than
 * against a golden that does not exist for this stage.
 *
 *   ./zimage_vae_encode_gpu_check <encoder.safetensors> <shaders.metal> [side]
 */
#include "zimage_vae.h"
#include "zimage_vae_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <encoder> <shaders.metal> [side]\n", argv[0]);
        return 2;
    }
    const int side = argc > 3 ? atoi(argv[3]) : 256;
    char problem[512] = {0};

    const size_t pixels = (size_t)side * side;
    const int latent_side = side / 8;
    const size_t latent_count = (size_t)16 * latent_side * latent_side;
    float *image = malloc(3 * pixels * sizeof(float));
    float *host = malloc(latent_count * sizeof(float));
    float *device = malloc(latent_count * sizeof(float));
    if (!image || !host || !device) { fprintf(stderr, "oom\n"); return 1; }
    /* Something with edges and gradients in every channel. */
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < side; y++)
            for (int x = 0; x < side; x++)
                image[((size_t)c * side + y) * side + x] =
                    sinf((float)(x + c * 7) * 0.05f) * cosf((float)y * 0.03f)
                    + ((x / 16 + y / 16) % 2 ? 0.35f : -0.35f);

    qwen_weights *weights = qwen_weights_open(argv[1], problem, sizeof(problem));
    if (!weights) { fprintf(stderr, "cpu weights: %s\n", problem); return 1; }
    double began = now();
    if (!zimage_vae_encode(weights, image, side, side, host, NULL, NULL)) {
        fprintf(stderr, "cpu encode failed\n"); return 1;
    }
    const double cpu_seconds = now() - began;

    zimage_vae_gpu *gpu = zimage_vae_gpu_create_encoder(
        argv[2], argv[1], NULL, side, side, problem, sizeof(problem));
    if (!gpu) { fprintf(stderr, "gpu: %s\n", problem); return 1; }
    began = now();
    if (!zimage_vae_gpu_encode(gpu, image, side, side, device, problem,
                               sizeof(problem))) {
        fprintf(stderr, "gpu encode: %s\n", problem); return 1;
    }
    const double gpu_seconds = now() - began;

    double worst = 0, sum = 0, scale = 0;
    for (size_t i = 0; i < latent_count; i++) {
        const double delta = fabs((double)host[i] - device[i]);
        if (delta > worst) worst = delta;
        sum += delta * delta;
        scale += (double)host[i] * host[i];
    }
    printf("%d square -> %d x %d x 16\n", side, latent_side, latent_side);
    printf("  cpu %7.2f s   device %6.2f s   %5.1fx faster\n",
           cpu_seconds, gpu_seconds, cpu_seconds / gpu_seconds);
    printf("  worst |difference| %.3e   relative rms %.3e\n",
           worst, sqrt(sum / latent_count) / sqrt(scale / latent_count));
    printf("  %s\n", worst < 2e-2 ? "they agree" : "THEY DO NOT AGREE");
    zimage_vae_gpu_release(gpu);
    return worst < 2e-2 ? 0 : 1;
}
