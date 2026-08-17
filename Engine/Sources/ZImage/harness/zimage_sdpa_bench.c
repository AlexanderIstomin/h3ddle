/* What does attention cost as the canvas grows?
 *
 * At 320 tokens stubbing SDPA out changed the wall clock not at all, so it was
 * written off. That conclusion does not survive a bigger picture: attention is
 * quadratic in the sequence, so going from 320 to 4128 tokens is 166x the work
 * for 12.9x the tokens, and a term that was 1.3% of a block can become the
 * largest single item. Measuring it across the three canvases the app offers
 * says which.
 *
 * The GEMM totals are printed beside it so the comparison is like for like —
 * "attention is slow" means nothing without what it is slow *against*.
 *
 *   ./zimage_sdpa_bench <shaders.metal>
 */
#include "h3_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define HEADS     30
#define HEAD_DIM  128
#define DIM       (HEADS * HEAD_DIM)
#define FFN       10240
#define PEAK      5.31

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <shaders.metal>\n", argv[0]); return 2; }
    char error[512] = {0};
    if (!h3_gpu_prepare(argv[1], error, sizeof(error))) {
        fprintf(stderr, "prepare: %s\n", error); return 1;
    }
    h3_gpu *gpu = h3_gpu_create(argv[1], error, sizeof(error));
    if (!gpu) { fprintf(stderr, "gpu: %s\n", error); return 1; }

    const int sequences[] = {320, 1056, 4128};
    printf("%8s %8s %11s %13s %10s %10s\n", "canvas", "tokens",
           "token-major", "head-major-out", "1 transpose", "TFLOP/s");
    for (int index = 0; index < 3; index++) {
        const int tokens = sequences[index];
        const size_t elements = (size_t)tokens * DIM;
        uint16_t *host = calloc(elements, sizeof(uint16_t));
        for (size_t i = 0; i < elements; i++) host[i] = (uint16_t)(0x3f00 + (i % 64));
        h3_gpu_tensor *q = h3_gpu_tensor_from_bf16(gpu, host, elements);
        h3_gpu_tensor *k = h3_gpu_tensor_from_bf16(gpu, host, elements);
        h3_gpu_tensor *v = h3_gpu_tensor_from_bf16(gpu, host, elements);
        h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, elements);
        free(host);
        if (!q || !k || !v || !out) { fprintf(stderr, "alloc failed\n"); return 1; }

        /* Both entry points, because they differ by exactly one transpose —
         * our tensors are token-major and MPSGraph wants head-major — and the
         * gap between them is the only honest way to price that reshaping
         * rather than inferring it from a model of two data points. */
        double seconds = 0.0, head_major = 0.0;
        for (int variant = 0; variant < 2; variant++) {
            for (int pass = 0; pass < 2; pass++) {
                const double began = now();
                if (!h3_gpu_begin(gpu)) return 1;
                for (int repeat = 0; repeat < 4; repeat++) {
                    const int ok = variant
                        ? h3_gpu_sdpa_bf16_head_major_output(
                              gpu, out, q, k, v, tokens, HEADS, HEAD_DIM, 0.088388f)
                        : h3_gpu_sdpa_bf16(gpu, out, q, k, v, tokens, HEADS,
                                           HEAD_DIM, 0.088388f);
                    if (!ok) { fprintf(stderr, "sdpa failed\n"); return 1; }
                }
                if (!h3_gpu_submit(gpu)) return 1;
                if (variant) head_major = (now() - began) / 4;
                else seconds = (now() - began) / 4;
            }
        }
        /* QK^T and PV, both 2 * S * S * head_dim per head. */
        const double flops = 4.0 * (double)tokens * tokens * HEAD_DIM * HEADS;
        /* The four projections of one block, for scale. */
        const double gemm = 2.0 * tokens *
            ((double)DIM * DIM * 3 + (double)DIM * DIM +
             (double)DIM * FFN * 2 + (double)FFN * DIM);
        const double gemm_seconds = gemm / 3.0e12;   /* measured ~3 TFLOP/s */
        (void)gemm; (void)gemm_seconds;
        printf("%8d %8d %8.1f ms %10.1f ms %8.1f ms %10.2f\n",
               tokens == 320 ? 256 : (tokens == 1056 ? 512 : 1024), tokens,
               seconds * 1e3, head_major * 1e3,
               (seconds - head_major) * 1e3, flops / head_major / 1e12);

        h3_gpu_tensor_free(q); h3_gpu_tensor_free(k);
        h3_gpu_tensor_free(v); h3_gpu_tensor_free(out);
    }
    printf("\npeak %.2f TFLOP/s; the GEMM column is one block's four "
           "projections at the 3 TFLOP/s the tiles now reach\n", PEAK);
    h3_gpu_free(gpu);
    return 0;
}
