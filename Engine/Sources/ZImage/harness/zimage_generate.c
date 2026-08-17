/* Prompt in, picture out — the whole chain in C, nothing borrowed from Python.
 *
 * Every stage so far has been a tensor comparison against a reference that
 * supplied its own inputs. That proves each component and says nothing about
 * the joins between them, which is where the remaining faults live: the chat
 * template the caption is wrapped in, the encoder tap being one block short of
 * the end, the caption arriving un-normed, the latent being unscaled before
 * the decoder rather than after. A wrong join produces a picture — just not
 * the right one — so the only test that covers them is the whole run.
 *
 *   ./zimage_generate <package-dir> <dit.safetensors> <prompt> <side> <steps> <out.bin>
 *
 * `side` is the latent side; the picture comes out eight times that.
 */
#include "zimage_dit.h"
#include "zimage_gpu.h"
#include "zimage_encoder.h"
#include "zimage_vae.h"
#include "zimage_vae_gpu.h"
#include "h3_tokenizer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SHIFT        3.0f
#define TRAIN_STEPS  1000.0f

static char problem[512];

static double now(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

/* A seeded normal, so a run can be repeated exactly. Box-Muller over
 * xorshift64* — the sampler only needs the right distribution, not any
 * particular library's stream. */
static uint64_t rng_state = 0x2545F4914F6CDD1DULL;

static double uniform(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    const uint64_t value = rng_state * 0x2545F4914F6CDD1DULL;
    return ((double)(value >> 11) + 0.5) / 9007199254740992.0;
}

static float normal(void) {
    const double a = uniform(), b = uniform();
    return (float)(sqrt(-2.0 * log(a)) * cos(2.0 * M_PI * b));
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s <package-dir> <dit.safetensors> <prompt> "
                        "<side> <steps> <out.bin>\n", argv[0]);
        return 2;
    }
    const char *package = argv[1];
    const char *prompt = argv[3];
    const int side = atoi(argv[4]);
    const int steps = atoi(argv[5]);
    char path[1200];

    /* ---- tokenize ---------------------------------------------------- */
    /* The caption is wrapped in the chat template the pipeline applies —
     * apply_chat_template with add_generation_prompt — and no think block is
     * emitted despite enable_thinking, because nothing is being generated. */
    char templated[4096];
    snprintf(templated, sizeof(templated),
             "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt);

    snprintf(path, sizeof(path), "%s/tokenizer.json", package);
    h3_tokenizer *tokenizer = h3_tokenizer_load(path, problem, sizeof(problem));
    if (!tokenizer) { fprintf(stderr, "tokenizer: %s\n", problem); return 1; }
    uint32_t *ids = NULL;
    size_t count = 0;
    if (!h3_tokenizer_encode(tokenizer, templated, 0, &ids, &count,
                             problem, sizeof(problem))) {
        fprintf(stderr, "encode: %s\n", problem);
        return 1;
    }
    h3_tokenizer_free(tokenizer);
    printf("prompt: %zu tokens\n", count);

    /* ---- encode ------------------------------------------------------ */
    double clock_start = now();
    snprintf(path, sizeof(path), "%s/text_encoder.safetensors", package);
    qwen_weights *encoder = qwen_weights_open(path, problem, sizeof(problem));
    if (!encoder) { fprintf(stderr, "encoder: %s\n", problem); return 1; }
    float *caption = malloc(count * ZIMAGE_CAP_DIM * sizeof(float));
    if (!caption || !zimage_encode(encoder, ids, (int)count, caption,
                                   problem, sizeof(problem))) {
        fprintf(stderr, "encoder: %s\n", problem);
        return 1;
    }
    /* Closed before the DiT opens: between them they map over 20 GB, and
     * nothing needs the encoder again. */
    qwen_weights_close(encoder);
    h3_tokenizer_ids_free(ids);
    printf("encoded  %d of %d layers in %.1f s\n",
           ZIMAGE_ENCODER_USED, ZIMAGE_ENCODER_LAYERS, now() - clock_start);

    /* ---- sample ------------------------------------------------------ */
    qwen_weights *transformer = qwen_weights_open(argv[2], problem, sizeof(problem));
    if (!transformer) { fprintf(stderr, "dit: %s\n", problem); return 1; }
    /* A seventh argument names the Metal shaders and moves the 34 blocks to
     * the device; without it everything stays on the CPU path. */
    /* The blocks read the shipped int8 package; the embedders and the head
     * stay on the CPU and read bf16, because the package quantises the head's
     * two linears too and the CPU path has no int8 reader. Two files, each in
     * the dtype its side of the split can use. */
    zimage_gpu *device = NULL;
    const int on_device = argc > 8;
    if (on_device) {
        const int tokens_side = side / ZIMAGE_PATCH;
        const int sequence = tokens_side * tokens_side + 512;
        device = zimage_gpu_create(argv[7], argv[8], sequence,
                                   problem, sizeof(problem));
        if (!device) { fprintf(stderr, "gpu: %s\n", problem); return 1; }
        printf("blocks on the GPU\n");
    }
    zimage_dit dit;
    if (!zimage_dit_init(&dit, transformer, side, caption, (int)count, device,
                         problem, sizeof(problem))) {
        fprintf(stderr, "dit: %s\n", problem);
        return 1;
    }
    printf("sequence %d = %d image + %d caption\n",
           dit.sequence, dit.image_tokens, dit.caption_padded);

    const size_t latent_count = (size_t)ZIMAGE_LATENT_CHANNELS * side * side;
    float *latent = malloc(latent_count * sizeof(float));
    float *velocity = malloc(latent_count * sizeof(float));
    if (!latent || !velocity) { fprintf(stderr, "out of memory\n"); return 1; }
    for (size_t index = 0; index < latent_count; index++) latent[index] = normal();

    /* sigmas: the pipeline's linspace(1, 1/N, N), then the *static* shift.
     * The mu the pipeline computes is dead code under use_dynamic_shifting
     * false. A terminal zero is appended so the last step lands on the image
     * rather than short of it. */
    float *sigmas = malloc((size_t)(steps + 1) * sizeof(float));
    for (int step = 0; step < steps; step++) {
        const float linear = 1.0f + (1.0f / (float)steps - 1.0f) *
                                    (float)step / (float)(steps - 1);
        sigmas[step] = SHIFT * linear / (1.0f + (SHIFT - 1.0f) * linear);
    }
    sigmas[steps] = 0.0f;

    for (int step = 0; step < steps; step++) {
        const double began = now();
        /* The model is fed 1 - sigma, not sigma. */
        const float timestep = (TRAIN_STEPS - sigmas[step] * TRAIN_STEPS) / TRAIN_STEPS;
        if (!zimage_dit_step(&dit, latent, timestep, velocity, NULL, NULL)) {
            fprintf(stderr, "step %d failed\n", step);
            return 1;
        }
        /* Euler on the flow, with the model output negated. */
        const float dt = sigmas[step + 1] - sigmas[step];
        for (size_t index = 0; index < latent_count; index++)
            latent[index] += dt * -velocity[index];
        double magnitude = 0.0;
        for (size_t index = 0; index < latent_count; index++)
            magnitude += fabs((double)latent[index]);
        printf("  step %d/%d  sigma %.4f -> %.4f  |latent| %.4f  %.1f s\n",
               step + 1, steps, sigmas[step], sigmas[step + 1],
               magnitude / (double)latent_count, now() - began);
        fflush(stdout);
    }
    zimage_dit_release(&dit);
    qwen_weights_close(transformer);
    /* The transformer's 6.17 GB of device tensors are dead once the last step
     * has run, and the decoder's buffers are the largest in the pipeline —
     * 256 channels at the full picture, five of them, which is 12 GB at 1536²
     * and 21 at 2048². Holding both at once is what puts a ceiling on the
     * canvas, so let the DiT go first. */
    if (device) {
        zimage_gpu_release(device);
        device = NULL;
    }

    /* ---- decode ------------------------------------------------------ */
    /* The decoder wants the latent raw: divide out the scaling, then add the
     * shift. Getting the order or the direction wrong leaves a picture with
     * the right structure and the wrong contrast. */
    for (size_t index = 0; index < latent_count; index++)
        latent[index] = latent[index] / ZIMAGE_VAE_SCALING + ZIMAGE_VAE_SHIFT;

    snprintf(path, sizeof(path), "%s/vae_decoder.safetensors", package);
    const int pixels = side * 8;
    float *image = malloc((size_t)3 * pixels * pixels * sizeof(float));
    if (!image) { fprintf(stderr, "out of memory\n"); return 1; }
    clock_start = now();
    if (on_device) {
        /* Shares the DiT's context rather than opening a second one: the
         * weights are already resident and the decoder only adds 99 MB. */
        zimage_vae_gpu *vae = zimage_vae_gpu_create(
            argv[7], path, NULL, side, problem, sizeof(problem));
        if (!vae || !zimage_vae_gpu_decode(vae, latent, side, image,
                                           problem, sizeof(problem))) {
            fprintf(stderr, "vae: %s\n", problem);
            return 1;
        }
        zimage_vae_gpu_release(vae);
    } else {
        qwen_weights *decoder = qwen_weights_open(path, problem, sizeof(problem));
        if (!decoder) { fprintf(stderr, "vae: %s\n", problem); return 1; }
        if (!zimage_vae_decode(decoder, latent, side, image, NULL, NULL)) {
            fprintf(stderr, "decode failed\n");
            return 1;
        }
        qwen_weights_close(decoder);
    }
    printf("decoded %d x %d in %.1f s\n", pixels, pixels, now() - clock_start);

    FILE *out = fopen(argv[6], "wb");
    if (!out) { fprintf(stderr, "cannot write %s\n", argv[6]); return 1; }
    fwrite(image, sizeof(float), (size_t)3 * pixels * pixels, out);
    fclose(out);
    double low = image[0], high = image[0];
    for (size_t index = 0; index < (size_t)3 * pixels * pixels; index++) {
        if (image[index] < low) low = image[index];
        if (image[index] > high) high = image[index];
    }
    printf("wrote %s, %d x %d x 3 f32, range [%.3f, %.3f]\n",
           argv[6], pixels, pixels, low, high);

    free(image); free(latent); free(velocity); free(sigmas); free(caption);
    return 0;
}
