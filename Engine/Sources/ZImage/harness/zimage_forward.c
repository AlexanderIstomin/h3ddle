/* Stage 5: does a whole S3-DiT forward come out right?
 *
 * Stage 4 proved one block. What this proves is everything around the blocks,
 * which is the part that produces a well-formed wrong picture rather than an
 * obvious failure: the refiners run over their own half of the sequence and
 * not the joint one, the position ids are built rather than read, and only
 * the image tokens are unpatchified.
 *
 * The forward itself lives in zimage_dit.c and is the same code the generator
 * runs; this file supplies a tap that compares each stage. Run it against
 * bf16 weights and against the shipped int8: the first says whether the
 * plumbing is right, the second says what quantisation costs once thirty
 * layers have compounded it. One number cannot answer both.
 *
 *   ./zimage_forward <weights.safetensors> <golden.safetensors>
 */
#include "zimage_dit.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LATENT_SIDE  32
#define CAP_TOKENS   40

static qwen_weights *reference;
static char problem[512];

static void compare(const char *stage, const float *ours, size_t count,
                    void *context) {
    (void)context;
    /* Stages the reference did not capture — most of the thirty trunk
     * layers — are simply not compared. */
    int64_t shape[4] = {-1, -1, -1, -1};
    const float *theirs = qwen_weights_f32(reference, stage, 2, shape,
                                           problem, sizeof(problem));
    if (!theirs) {
        theirs = qwen_weights_f32(reference, stage, 4, shape,
                                  problem, sizeof(problem));
        if (!theirs) return;
    }
    double square = 0.0, total = 0.0, worst = 0.0;
    for (size_t index = 0; index < count; index++) {
        const double difference = (double)ours[index] - (double)theirs[index];
        square += difference * difference;
        total += (double)theirs[index] * (double)theirs[index];
        if (fabs(difference) > worst) worst = fabs(difference);
    }
    printf("  %-14s RMS rel %8.2e   worst %8.2e\n", stage,
           sqrt(square / (double)count) / sqrt(total / (double)count), worst);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <weights.safetensors> <golden.safetensors>\n",
                argv[0]);
        return 2;
    }
    qwen_weights *store = qwen_weights_open(argv[1], problem, sizeof(problem));
    if (!store) { fprintf(stderr, "weights: %s\n", problem); return 1; }
    reference = qwen_weights_open(argv[2], problem, sizeof(problem));
    if (!reference) { fprintf(stderr, "golden: %s\n", problem); return 1; }

    const int64_t caption_shape[2] = {CAP_TOKENS, CAP_DIM};
    const float *caption = qwen_weights_f32(reference, "caption", 2, caption_shape,
                                            problem, sizeof(problem));
    const int64_t latent_shape[4] = {ZIMAGE_LATENT_CHANNELS, 1,
                                     LATENT_SIDE, LATENT_SIDE};
    const float *latent = qwen_weights_f32(reference, "latent", 4, latent_shape,
                                           problem, sizeof(problem));
    const int64_t one[1] = {1};
    const float *timestep = qwen_weights_f32(reference, "timestep", 1, one,
                                             problem, sizeof(problem));
    if (!caption || !latent || !timestep) {
        fprintf(stderr, "golden: %s\n", problem);
        return 1;
    }

    zimage_dit dit;
    if (!zimage_dit_init(&dit, store, LATENT_SIDE, caption, CAP_TOKENS,
                         NULL, problem, sizeof(problem))) {
        fprintf(stderr, "init: %s\n", problem);
        return 1;
    }
    printf("%d tokens = %d image + %d caption, %d layers + %d refiners each\n\n",
           dit.sequence, dit.image_tokens, dit.caption_padded,
           ZIMAGE_LAYERS, ZIMAGE_REFINERS);

    /* The refined caption is computed once at init, so check it here rather
     * than expecting it to appear in a per-step tap. */
    const int64_t cap_shape[2] = {64, DIM};
    const float *refined = qwen_weights_f32(reference, "after_context_refiner",
                                            2, cap_shape, problem, sizeof(problem));
    if (refined)
        compare("after_context_refiner", dit.caption_refined,
                (size_t)dit.caption_padded * DIM, NULL);

    float *velocity = malloc((size_t)ZIMAGE_LATENT_CHANNELS *
                             LATENT_SIDE * LATENT_SIDE * sizeof(float));
    if (!zimage_dit_step(&dit, latent, timestep[0], velocity, compare, NULL)) {
        fprintf(stderr, "step failed\n");
        return 1;
    }

    free(velocity);
    zimage_dit_release(&dit);
    qwen_weights_close(store);
    qwen_weights_close(reference);
    return 0;
}
