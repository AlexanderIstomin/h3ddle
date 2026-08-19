/* Compare the production Metal prompt encoder with the f32 CPU reference.
 * Model weights and shaders are deliberately external to the repository.
 *
 *   ./zimage_encoder_gpu_compare <text_encoder.safetensors> <h3_shaders.metal>
 */
#include "zimage_encoder.h"
#include "qwen_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TOKENS 32

static double now(void) {
    struct timespec stamp;
    clock_gettime(CLOCK_MONOTONIC, &stamp);
    return (double)stamp.tv_sec + (double)stamp.tv_nsec / 1e9;
}

static void tick(int layer, int layers, void *context) {
    (void)context;
    fprintf(stderr, "\r%d/%d", layer, layers);
    if (layer == layers) fputc('\n', stderr);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <text_encoder.safetensors> "
                        "<h3_shaders.metal>\n", argv[0]);
        return 2;
    }
    char error[512] = {0};
    qwen_weights *weights = qwen_weights_open(argv[1], error, sizeof(error));
    if (!weights) {
        fprintf(stderr, "weights: %s\n", error);
        return 1;
    }

    /* Ordinary in-vocabulary rows verify every projection, norm, rotary
     * position, attention row and residual in all 35 layers. */
    uint32_t ids[TOKENS];
    for (uint32_t index = 0; index < TOKENS; index++)
        ids[index] = 1000u + index * 97u;
    const size_t elements = (size_t)TOKENS * ZIMAGE_CAP_DIM;
    float *cpu = malloc(elements * sizeof(*cpu));
    float *gpu = malloc(elements * sizeof(*gpu));
    if (!cpu || !gpu) {
        fprintf(stderr, "out of memory\n");
        free(cpu);
        free(gpu);
        qwen_weights_close(weights);
        return 1;
    }

    double started = now();
    if (!zimage_encode(weights, ids, TOKENS, cpu, tick, NULL,
                       error, sizeof(error))) {
        fprintf(stderr, "CPU encoder: %s\n", error);
        return 1;
    }
    double cpu_seconds = now() - started;
    started = now();
    if (!zimage_encode_metal(weights, argv[2], ids, TOKENS, gpu, tick, NULL,
                             error, sizeof(error))) {
        fprintf(stderr, "Metal encoder: %s\n", error);
        return 1;
    }
    double gpu_seconds = now() - started;

    double difference = 0.0;
    double reference = 0.0;
    double worst = 0.0;
    for (size_t index = 0; index < elements; index++) {
        const double delta = (double)gpu[index] - cpu[index];
        if (!isfinite(gpu[index])) {
            fprintf(stderr, "Metal output is not finite at element %zu\n",
                    index);
            return 1;
        }
        difference += delta * delta;
        reference += (double)cpu[index] * cpu[index];
        if (fabs(delta) > worst) worst = fabs(delta);
    }
    const double relative = reference > 0.0
        ? sqrt(difference / reference) : 0.0;
    printf("CPU %.3fs, Metal %.3fs, %.2fx faster\n",
           cpu_seconds, gpu_seconds, cpu_seconds / gpu_seconds);
    printf("RMS relative %.6f, worst absolute %.6f\n", relative, worst);

    free(cpu);
    free(gpu);
    qwen_weights_close(weights);
    /* BF16 activations round at every operation while the CPU reference keeps
     * f32 activations. Several BF16 ulps can accumulate through 35 residual
     * blocks without indicating a route or layout error. */
    return relative <= 0.05 ? 0 : 1;
}
