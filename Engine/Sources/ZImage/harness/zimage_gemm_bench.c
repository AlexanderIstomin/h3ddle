/* What fraction of the machine does the int8 GEMM actually get?
 *
 * The block timing mixes four projections with norms, rope, attention and
 * gates, so a number derived from it cannot say whether the GEMM is near the
 * hardware or far from it. This times the projections alone, on the exact
 * shapes the DiT uses, and prints both the achieved rate and the traffic the
 * tile geometry implies — because the two candidate ceilings, 5.3 TFLOP/s of
 * arithmetic and 200 GB/s of bandwidth, are answered by different numbers.
 *
 *   ./zimage_gemm_bench <shaders.metal>
 */
#include "h3_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DIM  3840
#define FFN  10240

/* 16-core M1 Pro: 2048 ALUs, two flops per FMA, ~1.296 GHz. */
#define PEAK_TFLOPS   5.31
#define BANDWIDTH_GBS 200.0

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

typedef struct { const char *name; int rows, inputs, outputs; } shape;

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <shaders.metal>\n", argv[0]); return 2; }
    char error[512] = {0};
    if (!h3_gpu_prepare(argv[1], error, sizeof(error))) {
        fprintf(stderr, "prepare: %s\n", error); return 1;
    }
    h3_gpu *gpu = h3_gpu_create(argv[1], error, sizeof(error));
    if (!gpu) { fprintf(stderr, "gpu: %s\n", error); return 1; }

    const shape shapes[] = {
        {"qkv   ", 320, DIM, DIM * 3},
        {"out   ", 320, DIM, DIM},
        {"w13   ", 320, DIM, FFN * 2},
        {"w2    ", 320, FFN, DIM},
        {"qkv@1k", 1056, DIM, DIM * 3},
        {"w13@1k", 1056, DIM, FFN * 2},
        {"qkv@4k", 4128, DIM, DIM * 3},
        {"w13@4k", 4128, DIM, FFN * 2},
    };
    const int count = (int)(sizeof(shapes) / sizeof(shapes[0]));
    const int repeats = 8;

    printf("%-7s %6s %6s %6s | %8s %8s %8s | %7s %7s | %6s\n",
           "shape", "M", "K", "N", "8x8", "64x8", "32x32", "TFLOP/s", "GB/s", "peak");
    for (int index = 0; index < count; index++) {
        const shape s = shapes[index];
        const size_t in = (size_t)s.rows * s.inputs;
        const size_t out = (size_t)s.rows * s.outputs;
        const size_t weights = (size_t)s.outputs * s.inputs;

        uint16_t *host = calloc(in > out ? in : out, sizeof(uint16_t));
        int8_t *w = calloc(weights, 1);
        float *scales = calloc(s.outputs, sizeof(float));
        for (size_t i = 0; i < weights; i++) w[i] = (int8_t)((i % 13) - 6);
        for (int i = 0; i < s.outputs; i++) scales[i] = 0.01f;
        for (size_t i = 0; i < in; i++) host[i] = 0x3f80;   /* bf16 1.0 */

        h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(gpu, host, in);
        h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, out);
        h3_gpu_tensor *weight = h3_gpu_tensor_from_i8(gpu, w, weights);
        h3_gpu_tensor *scale = h3_gpu_tensor_from_f32(gpu, scales, s.outputs);
        free(host); free(w); free(scales);
        if (!input || !output || !weight || !scale) {
            fprintf(stderr, "alloc failed for %s\n", s.name); return 1;
        }

        double timing[3];
        for (int variant = 0; variant < 3; variant++) {
            for (int pass = 0; pass < 2; pass++) {          /* warm, then time */
                const double began = now();
                if (!h3_gpu_begin(gpu)) return 1;
                for (int repeat = 0; repeat < repeats; repeat++) {
                    int ok = 0;
                    if (variant == 0)
                        ok = h3_gpu_linear_i8_weight_bf16(
                                 gpu, output, input, weight, scale, NULL,
                                 s.rows, s.inputs, s.outputs);
                    else if (variant == 1)
                        ok = h3_gpu_linear_i8_weight_bf16_wide(
                                 gpu, output, input, weight, scale, NULL,
                                 s.rows, s.inputs, s.outputs);
                    else
                        ok = h3_gpu_linear_i8_weight_bf16_square(
                                 gpu, output, input, weight, scale, NULL,
                                 s.rows, s.inputs, s.outputs);
                    if (!ok) { fprintf(stderr, "dispatch failed\n"); return 1; }
                }
                if (!h3_gpu_submit(gpu)) return 1;
                timing[variant] = (now() - began) / repeats;
            }
        }

        const double flops = 2.0 * s.rows * s.inputs * s.outputs;
        const double rate = flops / timing[2] / 1e12;
        /* The square tile re-reads the weight M/32 times and the input N/32. */
        const double traffic = (double)(s.rows + 31) / 32 * weights +
                               (double)(s.outputs + 31) / 32 * in * 2;
        printf("%-7s %6d %6d %6d | %7.1fms %7.1fms %7.1fms | %7.2f %7.0f | %5.1f%%\n",
               s.name, s.rows, s.inputs, s.outputs,
               timing[0] * 1e3, timing[1] * 1e3, timing[2] * 1e3, rate,
               traffic / timing[2] / 1e9, 100.0 * rate / PEAK_TFLOPS);

        h3_gpu_tensor_free(input); h3_gpu_tensor_free(output);
        h3_gpu_tensor_free(weight); h3_gpu_tensor_free(scale);
    }
    printf("\npeak %.2f TFLOP/s, bandwidth %.0f GB/s\n", PEAK_TFLOPS, BANDWIDTH_GBS);
    h3_gpu_free(gpu);
    return 0;
}
