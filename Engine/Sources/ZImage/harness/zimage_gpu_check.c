/* Does the GPU block agree with the CPU one?
 *
 * Compared against the *int8* reference, not the bf16 one. The GPU runs the
 * shipped quantised weights, so measuring it against bf16 would fold ~3% of
 * quantisation error into the same number as the port's own error and leave
 * neither legible. Against the int8 reference the only difference left is
 * arithmetic, and it should be at f32 noise.
 *
 *   ./zimage_gpu_check <shaders.metal> <transformer.safetensors> <int8-golden>
 *   ./zimage_gpu_check <shaders.metal> <transformer.safetensors> --load-only [tokens]
 *   ./zimage_gpu_check <shaders.metal> <transformer.safetensors> --synthetic
 */
#include "zimage_gpu.h"
#include "zimage_block.h"
#include "qwen_weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LATENT_SIDE  32
#define PATCH        2
#define TOKENS_SIDE  (LATENT_SIDE / PATCH)
#define IMAGE_TOKENS (TOKENS_SIDE * TOKENS_SIDE)
#define CAP_PADDED   64
#define SEQUENCE     (IMAGE_TOKENS + CAP_PADDED)

static char problem[512];

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void compare(const char *label, const float *ours, const float *theirs,
                    size_t count) {
    double square = 0.0, total = 0.0, worst = 0.0;
    for (size_t index = 0; index < count; index++) {
        const double difference = (double)ours[index] - (double)theirs[index];
        square += difference * difference;
        total += (double)theirs[index] * (double)theirs[index];
        if (fabs(difference) > worst) worst = fabs(difference);
    }
    printf("  %-24s RMS rel %8.2e   worst %8.2e\n", label,
           sqrt(square / (double)count) / sqrt(total / (double)count), worst);
}

static uint64_t hash_floats(const float *values, size_t count) {
    const unsigned char *bytes = (const unsigned char *)values;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < count * sizeof(*values); index++) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int synthetic_run(const char *shaders, const char *weights) {
    float *cosines = malloc((size_t)SEQUENCE * HALF * sizeof(float));
    float *sines = malloc((size_t)SEQUENCE * HALF * sizeof(float));
    float *caption = malloc((size_t)CAP_PADDED * DIM * sizeof(float));
    float *unified = malloc((size_t)SEQUENCE * DIM * sizeof(float));
    if (!cosines || !sines || !caption || !unified) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    for (size_t index = 0; index < (size_t)SEQUENCE * HALF; index++) {
        cosines[index] = cosf((float)index * 0.0007f);
        sines[index] = sinf((float)index * 0.0007f);
    }
    for (size_t index = 0; index < (size_t)CAP_PADDED * DIM; index++)
        caption[index] = 0.15f * sinf((float)index * 0.0013f);

    zimage_gpu *gpu = zimage_gpu_create(shaders, weights, SEQUENCE,
                                        problem, sizeof(problem));
    if (!gpu) { fprintf(stderr, "create: %s\n", problem); return 1; }
    if (!zimage_gpu_refine_context(
            gpu, caption, CAP_PADDED,
            cosines + (size_t)IMAGE_TOKENS * HALF,
            sines + (size_t)IMAGE_TOKENS * HALF,
            problem, sizeof(problem))) {
        fprintf(stderr, "context: %s\n", problem);
        return 1;
    }
    for (size_t index = 0; index < (size_t)IMAGE_TOKENS * DIM; index++)
        unified[index] = 0.2f * cosf((float)index * 0.0009f);
    memcpy(unified + (size_t)IMAGE_TOKENS * DIM, caption,
           (size_t)CAP_PADDED * DIM * sizeof(float));
    float adaln[256];
    for (int index = 0; index < 256; index++)
        adaln[index] = 0.1f * sinf((float)index * 0.07f);
    if (!zimage_gpu_forward(gpu, unified, IMAGE_TOKENS, SEQUENCE, adaln,
                            cosines, sines, problem, sizeof(problem))) {
        fprintf(stderr, "forward: %s\n", problem);
        return 1;
    }
    printf("synthetic context %016llx\n",
           (unsigned long long)hash_floats(caption, (size_t)CAP_PADDED * DIM));
    printf("synthetic forward %016llx\n",
           (unsigned long long)hash_floats(unified, (size_t)SEQUENCE * DIM));
    zimage_gpu_release(gpu);
    free(cosines); free(sines); free(caption); free(unified);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <shaders.metal> <transformer.safetensors> "
                        "<int8-golden.safetensors>\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[3], "--load-only") == 0) {
        const int tokens = argc > 4 ? atoi(argv[4]) : SEQUENCE;
        double began = now();
        zimage_gpu *gpu = zimage_gpu_create(argv[1], argv[2], tokens,
                                            problem, sizeof(problem));
        if (!gpu) { fprintf(stderr, "create: %s\n", problem); return 1; }
        h3_gpu_stats stats;
        if (!h3_gpu_get_stats(zimage_gpu_device(gpu), &stats)) {
            fprintf(stderr, "cannot read GPU statistics\n");
            return 1;
        }
        printf("loaded 34 blocks in %.3f s\n", now() - began);
        printf("  %.3f GiB live, %.3f GiB peak, %.3f GiB cumulatively allocated\n",
               (double)stats.live_bytes / (1024.0 * 1024.0 * 1024.0),
               (double)stats.peak_live_bytes / (1024.0 * 1024.0 * 1024.0),
               (double)stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0));
        printf("  %llu tensor allocations\n",
               (unsigned long long)stats.tensor_allocations);
        zimage_gpu_release(gpu);
        return 0;
    }
    if (strcmp(argv[3], "--synthetic") == 0)
        return synthetic_run(argv[1], argv[2]);
    qwen_weights *golden = qwen_weights_open(argv[3], problem, sizeof(problem));
    if (!golden) { fprintf(stderr, "golden: %s\n", problem); return 1; }
    const int64_t any[2] = {-1, -1};

    /* Positions, exactly as the CPU path builds them: the caption occupies
     * 1..P on the temporal axis with both spatial axes zero, the image starts
     * at P+1 and spreads over the two spatial ones, image rows first. */
    float *pos_ids = malloc((size_t)SEQUENCE * 3 * sizeof(float));
    for (int row = 0; row < TOKENS_SIDE; row++)
        for (int column = 0; column < TOKENS_SIDE; column++) {
            float *ids = pos_ids + ((size_t)row * TOKENS_SIDE + column) * 3;
            ids[0] = (float)(CAP_PADDED + 1);
            ids[1] = (float)row;
            ids[2] = (float)column;
        }
    for (int token = 0; token < CAP_PADDED; token++) {
        float *ids = pos_ids + ((size_t)IMAGE_TOKENS + token) * 3;
        ids[0] = (float)(1 + token);
        ids[1] = ids[2] = 0.0f;
    }
    float *cosines = malloc((size_t)SEQUENCE * HALF * sizeof(float));
    float *sines = malloc((size_t)SEQUENCE * HALF * sizeof(float));
    zimage_rope_table(pos_ids, cosines, sines, SEQUENCE);

    double began = now();
    zimage_gpu *gpu = zimage_gpu_create(argv[1], argv[2], SEQUENCE,
                                        problem, sizeof(problem));
    if (!gpu) { fprintf(stderr, "create: %s\n", problem); return 1; }
    printf("loaded 34 blocks in %.1f s\n\n", now() - began);

    /* ---- the context refiners, caption rows alone ------------------- */
    const int64_t cap_shape[2] = {CAP_PADDED, DIM};
    const float *cap_in = qwen_weights_f32(golden, "cap_embedded", 2, cap_shape,
                                           problem, sizeof(problem));
    if (!cap_in) { fprintf(stderr, "golden: %s\n", problem); return 1; }
    float *caption = malloc((size_t)CAP_PADDED * DIM * sizeof(float));
    memcpy(caption, cap_in, (size_t)CAP_PADDED * DIM * sizeof(float));
    if (!zimage_gpu_refine_context(gpu, caption, CAP_PADDED,
                                   cosines + (size_t)IMAGE_TOKENS * HALF,
                                   sines + (size_t)IMAGE_TOKENS * HALF,
                                   problem, sizeof(problem))) {
        fprintf(stderr, "context: %s\n", problem);
        return 1;
    }
    printf("against the int8 reference:\n");
    compare("after context_refiner", caption,
            qwen_weights_f32(golden, "after_context_refiner", 2, cap_shape,
                             problem, sizeof(problem)),
            (size_t)CAP_PADDED * DIM);

    /* ---- noise refiners and the trunk -------------------------------- */
    const int64_t image_shape[2] = {IMAGE_TOKENS, DIM};
    const int64_t unified_shape[2] = {SEQUENCE, DIM};
    const float *image_in = qwen_weights_f32(golden, "image_embedded", 2,
                                             image_shape, problem, sizeof(problem));
    const float *adaln = qwen_weights_f32(golden, "adaln_input", 2, any,
                                          problem, sizeof(problem));
    if (!image_in || !adaln) { fprintf(stderr, "golden: %s\n", problem); return 1; }

    float *unified = malloc((size_t)SEQUENCE * DIM * sizeof(float));
    memcpy(unified, image_in, (size_t)IMAGE_TOKENS * DIM * sizeof(float));
    memcpy(unified + (size_t)IMAGE_TOKENS * DIM, caption,
           (size_t)CAP_PADDED * DIM * sizeof(float));

    /* Three passes, because the first pays for pipeline construction and
     * MPSGraph compilation and so measures the warm-up rather than the work.
     * Quoting that number as the speed-up would flatter nothing and mislead
     * about everything. */
    double elapsed = 0.0;
    float *fresh = malloc((size_t)SEQUENCE * DIM * sizeof(float));
    memcpy(fresh, unified, (size_t)SEQUENCE * DIM * sizeof(float));
    for (int pass = 0; pass < 3; pass++) {
        memcpy(unified, fresh, (size_t)SEQUENCE * DIM * sizeof(float));
        began = now();
        if (!zimage_gpu_forward(gpu, unified, IMAGE_TOKENS, SEQUENCE, adaln,
                                cosines, sines, problem, sizeof(problem))) {
            fprintf(stderr, "forward: %s\n", problem);
            return 1;
        }
        elapsed = now() - began;
        h3_gpu_stats stats;
        h3_gpu_get_stats(zimage_gpu_device(gpu), &stats);
        printf("  pass %d: %.3f s wall | encode %.3f  wait %.3f  device %.3f "
               "| dispatches %llu direct, %llu mps-linear, %llu mps-sdpa, %llu blits\n",
               pass, elapsed, stats.command_encode_seconds,
               stats.command_wait_seconds, stats.gpu_seconds,
               (unsigned long long)stats.direct_dispatches,
               (unsigned long long)stats.mps_linear_dispatches,
               (unsigned long long)stats.mps_sdpa_dispatches,
               (unsigned long long)stats.blit_copies);
    }
    compare("after trunk", unified,
            qwen_weights_f32(golden, "trunk_29", 2, unified_shape,
                             problem, sizeof(problem)),
            (size_t)SEQUENCE * DIM);

    printf("\n%d tokens: %.3f s per forward (%.3f s inside command buffers)\n",
           SEQUENCE, elapsed, zimage_gpu_seconds(gpu));
    printf("the CPU path takes 33 s at this size, so %.0fx\n", 33.0 / elapsed);

    zimage_gpu_release(gpu);
    qwen_weights_close(golden);
    return 0;
}
