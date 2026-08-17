/* Where does a forward's time actually go, at the size that matters?
 *
 * Isolated op benchmarks disagreed with the real thing by 1.8x at 4128 tokens:
 * measured GEMM plus measured attention predicted ~25 s and the forward takes
 * 44.6 s. Isolation flatters — repeats on hot buffers in one command buffer —
 * so this runs the real sequence and removes one op at a time.
 *
 *   ZIMAGE_SKIP=sdpa ./zimage_forward_bench <shaders> <transformer> <tokens>
 */
#include "zimage_gpu.h"
#include "zimage_block.h"

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
    if (argc < 4) {
        fprintf(stderr, "usage: %s <shaders> <transformer> <tokens>\n", argv[0]);
        return 2;
    }
    const int sequence = atoi(argv[3]);
    const int image_tokens = sequence - 32;
    char problem[512];

    float *pos_ids = calloc((size_t)sequence * 3, sizeof(float));
    for (int token = 0; token < sequence; token++)
        pos_ids[(size_t)token * 3] = (float)(token % 512);
    float *cosines = malloc((size_t)sequence * HALF * sizeof(float));
    float *sines = malloc((size_t)sequence * HALF * sizeof(float));
    zimage_rope_table(pos_ids, cosines, sines, sequence);

    float *unified = malloc((size_t)sequence * DIM * sizeof(float));
    for (size_t index = 0; index < (size_t)sequence * DIM; index++)
        unified[index] = (float)((index % 71) - 35) * 0.01f;
    float adaln[ADALN];
    for (int index = 0; index < ADALN; index++) adaln[index] = 0.01f * index;

    zimage_gpu *gpu = zimage_gpu_create(argv[1], argv[2], sequence,
                                        problem, sizeof(problem));
    if (!gpu) { fprintf(stderr, "create: %s\n", problem); return 1; }

    /* Every pass printed, not the best: if the first is much slower than the
     * rest the difference is warm-up, and reporting the best would hide the
     * very thing that makes a real eight-step run cost what it does. */
    const char *skip = getenv("ZIMAGE_SKIP");
    printf("%5d tokens  skip=%-18s", sequence, skip ? skip : "(nothing)");
    for (int pass = 0; pass < 4; pass++) {
        const double began = now();
        if (!zimage_gpu_forward(gpu, unified, image_tokens, sequence, adaln,
                                cosines, sines, problem, sizeof(problem))) {
            fprintf(stderr, "forward: %s\n", problem);
            return 1;
        }
        printf(" %6.2f", now() - began);
        fflush(stdout);
    }
    printf("  s\n");
    zimage_gpu_release(gpu);
    return 0;
}
