/* Does the tiled attention agree with MPSGraph's, and is it faster?
 *
 * Correctness first and separately: a flash kernel that is wrong is not a
 * faster kernel. The reference here is the SDPA the port already ships, so a
 * disagreement is this kernel's fault and nothing else's.
 */
#include "h3_gpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define HEADS 30
#define WIDE  128

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static float widen(uint16_t v) {
    union { uint32_t b; float f; } c; c.b = (uint32_t)v << 16; return c.f;
}

int main(int argc, char **argv) {
    char error[512] = {0};
    if (!h3_gpu_prepare(argv[1], error, sizeof(error))) {
        fprintf(stderr, "prepare: %s\n", error); return 1;
    }
    h3_gpu *gpu = h3_gpu_create(argv[1], error, sizeof(error));
    if (!gpu) { fprintf(stderr, "gpu: %s\n", error); return 1; }

    const int sizes[] = {320, 1056, 4128, 9248};
    printf("%8s %10s %10s %8s %10s %s\n", "tokens", "MPSGraph", "flash",
           "speedup", "TFLOP/s", "agreement");
    for (int index = 0; index < 4; index++) {
        const int tokens = sizes[index];
        const size_t count = (size_t)tokens * HEADS * WIDE;
        uint16_t *host = malloc(count * sizeof(uint16_t));
        /* A spread wide enough that the online maximum actually moves between
         * tiles; constant input would pass a kernel that ignored rescaling. */
        for (size_t i = 0; i < count; i++) {
            const float v = sinf((float)(i % 9973) * 0.001f) * 2.0f;
            union { float f; uint32_t b; } c; c.f = v;
            host[i] = (uint16_t)((c.b + 0x7fff + ((c.b >> 16) & 1)) >> 16);
        }
        h3_gpu_tensor *q = h3_gpu_tensor_from_bf16(gpu, host, count);
        h3_gpu_tensor *k = h3_gpu_tensor_from_bf16(gpu, host, count);
        h3_gpu_tensor *v = h3_gpu_tensor_from_bf16(gpu, host, count);
        h3_gpu_tensor *reference = h3_gpu_tensor_new_bf16(gpu, count);
        h3_gpu_tensor *mine = h3_gpu_tensor_new_bf16(gpu, count);
        const float scale = 1.0f / sqrtf((float)WIDE);

        double slow = 0.0, fast = 0.0;
        for (int pass = 0; pass < 2; pass++) {
            double began = now();
            if (!h3_gpu_begin(gpu)) return 1;
            for (int r = 0; r < 3; r++)
                if (!h3_gpu_sdpa_bf16(gpu, reference, q, k, v, tokens, HEADS, WIDE, scale))
                    { printf("  reference failed\n"); return 1; }
            if (!h3_gpu_submit(gpu)) return 1;
            slow = (now() - began) / 3;

            began = now();
            if (!h3_gpu_begin(gpu)) return 1;
            for (int r = 0; r < 3; r++)
                if (!h3_gpu_flash_attention_bf16(gpu, mine, q, k, v, tokens, HEADS, WIDE, scale))
                    { printf("  flash failed\n"); return 1; }
            if (!h3_gpu_submit(gpu)) return 1;
            fast = (now() - began) / 3;
        }

        uint16_t *a = malloc(count * 2), *b = malloc(count * 2);
        h3_gpu_tensor_read_bf16(reference, a, count);
        h3_gpu_tensor_read_bf16(mine, b, count);
        double square = 0.0, total = 0.0;
        for (size_t i = 0; i < count; i++) {
            const double d = (double)widen(b[i]) - widen(a[i]);
            square += d * d; total += (double)widen(a[i]) * widen(a[i]);
        }
        printf("%8d %8.1f ms %8.1f ms %7.2fx %10.2f  %.2e\n", tokens,
               slow * 1e3, fast * 1e3, slow / fast,
               4.0 * tokens * tokens * WIDE * HEADS / fast / 1e12,
               sqrt(square / count) / sqrt(total / count));
        free(host); free(a); free(b);
        h3_gpu_tensor_free(q); h3_gpu_tensor_free(k); h3_gpu_tensor_free(v);
        h3_gpu_tensor_free(reference); h3_gpu_tensor_free(mine);
    }
    h3_gpu_free(gpu);
    return 0;
}
