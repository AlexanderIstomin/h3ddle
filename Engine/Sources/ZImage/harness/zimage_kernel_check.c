/* Does the one new kernel do what the block needs?
 *
 * The Metal library is compiled at runtime from source, so a syntax error or a
 * missing pipeline registration surfaces only when something actually runs.
 * Checking that here, before six hundred lines of DiT are built on top, keeps
 * a typo in the shader from arriving disguised as a wrong picture.
 *
 * Also checks the *contract* rather than just the arithmetic: adaln reads a
 * shift slot Z-Image does not have, so slot 0 must be zero, and adaln adds the
 * 1 to the scale itself, so the scales must pass through untouched.
 *
 *   ./zimage_kernel_check <h3_shaders.metal>
 */
#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 3840

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <h3_shaders.metal>\n", argv[0]);
        return 2;
    }
    char error[512] = {0};
    if (!h3_gpu_prepare(argv[1], error, sizeof(error))) {
        fprintf(stderr, "prepare: %s\n", error);
        return 1;
    }
    h3_gpu *gpu = h3_gpu_create(argv[1], error, sizeof(error));
    if (!gpu) { fprintf(stderr, "gpu: %s\n", error); return 1; }

    float *input = malloc(WIDTH * 4 * sizeof(float));
    /* A spread that reaches into tanh's flat tails, where a missing tanh would
     * otherwise look almost right. */
    for (int index = 0; index < WIDTH * 4; index++)
        input[index] = (float)((index % 97) - 48) * 0.15f;

    h3_gpu_tensor *linear = h3_gpu_tensor_from_f32(gpu, input, WIDTH * 4);
    h3_gpu_tensor *modulation = h3_gpu_tensor_new_f32(gpu, WIDTH * 5);
    if (!linear || !modulation) { fprintf(stderr, "alloc failed\n"); return 1; }

    if (!h3_gpu_begin(gpu) ||
        !h3_gpu_zimage_modulation_f32(gpu, modulation, linear, WIDTH) ||
        !h3_gpu_submit(gpu)) {
        fprintf(stderr, "dispatch failed\n");
        return 1;
    }

    float *got = malloc(WIDTH * 5 * sizeof(float));
    if (!h3_gpu_tensor_read_f32(modulation, got, WIDTH * 5)) {
        fprintf(stderr, "read failed\n");
        return 1;
    }

    double worst_shift = 0.0, worst_scale = 0.0, worst_gate = 0.0;
    for (int index = 0; index < WIDTH; index++) {
        const double shift = fabs(got[index]);
        const double scale_msa = fabs(got[WIDTH + index] - input[index]);
        const double gate_msa = fabs(got[WIDTH * 2 + index] - tanh(input[WIDTH + index]));
        const double scale_mlp = fabs(got[WIDTH * 3 + index] - input[WIDTH * 2 + index]);
        const double gate_mlp = fabs(got[WIDTH * 4 + index] - tanh(input[WIDTH * 3 + index]));
        if (shift > worst_shift) worst_shift = shift;
        if (scale_msa > worst_scale) worst_scale = scale_msa;
        if (scale_mlp > worst_scale) worst_scale = scale_mlp;
        if (gate_msa > worst_gate) worst_gate = gate_msa;
        if (gate_mlp > worst_gate) worst_gate = gate_mlp;
    }
    printf("shift slot held at zero   worst %.2e\n", worst_shift);
    printf("scales passed through     worst %.2e\n", worst_scale);
    printf("gates through tanh        worst %.2e\n", worst_gate);
    const int ok = worst_shift == 0.0 && worst_scale == 0.0 && worst_gate < 1e-6;
    printf("%s\n", ok ? "kernel agrees" : "KERNEL DISAGREES");

    free(input); free(got);
    h3_gpu_tensor_free(modulation);
    h3_gpu_tensor_free(linear);
    h3_gpu_free(gpu);
    return ok ? 0 : 1;
}
