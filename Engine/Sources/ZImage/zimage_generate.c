#include "zimage_generate.h"
#include "zimage_dit.h"
#include "zimage_encoder.h"
#include "zimage_gpu.h"
#include "zimage_vae.h"
#include "zimage_vae_gpu.h"
#include "h3_tokenizer.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SHIFT       3.0f
#define TRAIN_STEPS 1000.0f
#define VAE_SCALE   8

/* The encoder's per-layer tick, forwarded to the caller's progress callback.
 * Cancellation is deliberately not honoured here: `zimage_encode` has no way
 * to abandon a layer part-way, so a cancel during encoding takes effect at
 * the first sampler step instead. */
typedef struct {
    zimage_progress progress;
    void *context;
} encode_relay;

static void relay_encode_tick(int layer, int layers, void *context) {
    encode_relay *relay = context;
    if (relay->progress) relay->progress("text encoder", layer, layers,
                                         relay->context);
}

static int fail(char *error, size_t error_size, const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
    return 0;
}

int zimage_supports_canvas(int pixels) {
    if (pixels < 128 || pixels % (VAE_SCALE * ZIMAGE_PATCH)) return 0;
    const int side = pixels / VAE_SCALE;
    const int tokens_side = side / ZIMAGE_PATCH;
    /* The reference pads image tokens to a multiple of 32; this path asserts
     * instead, so canvases needing padding are refused rather than rendered
     * wrongly. 1440 gives 8100 tokens and does not qualify. */
    return (tokens_side * tokens_side) % ZIMAGE_SEQ_MULTIPLE == 0;
}

/* A seeded normal, so a seed and settings reproduce a picture exactly.
 * Box-Muller over xorshift64*: the sampler needs the distribution, not any
 * particular library's stream. */
static float normal(uint64_t *state) {
    double values[2];
    for (int index = 0; index < 2; index++) {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        const uint64_t scrambled = *state * 0x2545F4914F6CDD1DULL;
        values[index] = ((double)(scrambled >> 11) + 0.5) / 9007199254740992.0;
    }
    return (float)(sqrt(-2.0 * log(values[0])) * cos(2.0 * M_PI * values[1]));
}

int zimage_generate(const zimage_request *request, float *image,
                    zimage_progress progress, void *context,
                    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!request || !request->package || !request->prompt || !image)
        return fail(error, error_size, "a package, a prompt and somewhere to "
                                       "put the picture are all required");
    if (!zimage_supports_canvas(request->pixels))
        return fail(error, error_size,
                    "%d pixels square is not a canvas this build renders; "
                    "256, 512, 768, 1024, 1280, 1536 and 2048 are",
                    request->pixels);
    const int steps = request->steps > 0 ? request->steps : ZIMAGE_DEFAULT_STEPS;
    const int side = request->pixels / VAE_SCALE;
    char path[1200];

    /* ---- tokenize ---------------------------------------------------- */
    /* The caption is wrapped in the chat template the reference applies. No
     * think block is emitted despite enable_thinking, because nothing is
     * being generated. */
    const size_t templated_size = strlen(request->prompt) + 128;
    char *templated = malloc(templated_size);
    if (!templated) return fail(error, error_size, "out of memory");
    snprintf(templated, templated_size,
             "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
             request->prompt);

    snprintf(path, sizeof(path), "%s/tokenizer.json", request->package);
    h3_tokenizer *tokenizer = h3_tokenizer_load(path, error, error_size);
    if (!tokenizer) { free(templated); return 0; }
    uint32_t *ids = NULL;
    size_t count = 0;
    const int encoded = h3_tokenizer_encode(tokenizer, templated, 0, &ids,
                                            &count, error, error_size);
    h3_tokenizer_free(tokenizer);
    free(templated);
    if (!encoded) return 0;
    if (!count) {
        h3_tokenizer_ids_free(ids);
        return fail(error, error_size, "the prompt tokenized to nothing");
    }

    /* ---- encode ------------------------------------------------------ */
    snprintf(path, sizeof(path), "%s/text_encoder.safetensors", request->package);
    if (progress) progress("text encoder", 0, ZIMAGE_ENCODER_USED, context);
    qwen_weights *encoder = qwen_weights_open(path, error, error_size);
    float *caption = encoder ? malloc(count * ZIMAGE_CAP_DIM * sizeof(float)) : NULL;
    encode_relay relay = { .progress = progress, .context = context };
    if (!encoder || !caption ||
        !zimage_encode(encoder, ids, (int)count, caption, relay_encode_tick,
                       &relay, error, error_size)) {
        free(caption);
        if (encoder) qwen_weights_close(encoder);
        h3_tokenizer_ids_free(ids);
        return encoder ? 0 : fail(error, error_size, "cannot read the text encoder");
    }
    /* Closed before the transformer opens: between them they map over twenty
     * gigabytes, and nothing needs the encoder again. */
    qwen_weights_close(encoder);
    h3_tokenizer_ids_free(ids);

    /* ---- sample ------------------------------------------------------ */
    snprintf(path, sizeof(path), "%s/transformer.safetensors", request->package);
    /* Opening is a mapping and costs nothing; what follows is the GPU sizing
     * and the caption refiner, and together they are long enough to need
     * saying so. There is no inner counter to report, so this phase moves
     * once — which is still the difference between "working" and "hung". */
    if (progress) progress("transformer", 0, 1, context);
    qwen_weights *transformer = qwen_weights_open(path, error, error_size);
    if (!transformer) { free(caption); return 0; }

    zimage_gpu *device = NULL;
    if (request->shaders) {
        const int tokens_side = side / ZIMAGE_PATCH;
        const int sequence = tokens_side * tokens_side + 512;
        device = zimage_gpu_create(request->shaders, path, sequence,
                                   error, error_size);
        if (!device) {
            qwen_weights_close(transformer);
            free(caption);
            return 0;
        }
    }

    zimage_dit dit;
    if (!zimage_dit_init(&dit, transformer, side, caption, (int)count, device,
                         error, error_size)) {
        if (device) zimage_gpu_release(device);
        qwen_weights_close(transformer);
        free(caption);
        return 0;
    }
    free(caption);
    if (progress) progress("transformer", 1, 1, context);

    const size_t latent_count = (size_t)ZIMAGE_LATENT_CHANNELS * side * side;
    float *latent = malloc(latent_count * sizeof(float));
    float *velocity = malloc(latent_count * sizeof(float));
    float *sigmas = malloc((size_t)(steps + 1) * sizeof(float));
    if (!latent || !velocity || !sigmas) {
        free(latent); free(velocity); free(sigmas);
        zimage_dit_release(&dit);
        if (device) zimage_gpu_release(device);
        qwen_weights_close(transformer);
        return fail(error, error_size, "out of memory sizing the sampler");
    }
    uint64_t state = request->seed ? request->seed : 0x2545F4914F6CDD1DULL;
    for (size_t index = 0; index < latent_count; index++)
        latent[index] = normal(&state);

    /* The reference's linspace(1, 1/N, N), then a *static* shift of 3. The
     * `mu` it also computes is dead code under use_dynamic_shifting false. A
     * terminal zero is appended so the last step lands on the picture rather
     * than short of it. */
    for (int step = 0; step < steps; step++) {
        const float linear = steps > 1
            ? 1.0f + (1.0f / (float)steps - 1.0f) * (float)step / (float)(steps - 1)
            : 1.0f;
        sigmas[step] = SHIFT * linear / (1.0f + (SHIFT - 1.0f) * linear);
    }
    sigmas[steps] = 0.0f;

    int cancelled = 0, ok = 1;
    for (int step = 0; step < steps && ok && !cancelled; step++) {
        /* The model is fed 1 - sigma, not sigma. */
        const float timestep =
            (TRAIN_STEPS - sigmas[step] * TRAIN_STEPS) / TRAIN_STEPS;
        if (!zimage_dit_step(&dit, latent, timestep, velocity, NULL, NULL)) {
            ok = fail(error, error_size, "the transformer failed at step %d",
                      step + 1);
            break;
        }
        /* Euler on the flow, with the model's output negated. */
        const float dt = sigmas[step + 1] - sigmas[step];
        for (size_t index = 0; index < latent_count; index++)
            latent[index] += dt * -velocity[index];
        if (progress && !progress("denoise", step + 1, steps, context))
            cancelled = 1;
    }
    zimage_dit_release(&dit);
    qwen_weights_close(transformer);
    free(velocity); free(sigmas);
    /* The transformer's device tensors are dead now and the decoder's buffers
     * are the largest in the pipeline — 256 channels at the full picture, five
     * of them, which is 12 GB at 1536 pixels. Holding both is what puts a
     * ceiling on the canvas. */
    if (device) zimage_gpu_release(device);
    if (!ok || cancelled) { free(latent); return 0; }

    /* ---- decode ------------------------------------------------------ */
    /* The decoder wants the latent raw: divide out the scaling, then add the
     * shift. Getting the order or direction wrong leaves a picture with the
     * right structure and the wrong contrast. */
    for (size_t index = 0; index < latent_count; index++)
        latent[index] = latent[index] / ZIMAGE_VAE_SCALING + ZIMAGE_VAE_SHIFT;

    snprintf(path, sizeof(path), "%s/vae_decoder.safetensors", request->package);
    /* Tens of seconds at the larger canvases, and it lands after the last
     * step — so without this the bar sits full while the picture is still
     * being made. */
    if (progress) progress("image VAE", 0, 1, context);
    if (request->shaders) {
        zimage_vae_gpu *vae = zimage_vae_gpu_create(request->shaders, path, NULL,
                                                    side, error, error_size);
        ok = vae && zimage_vae_gpu_decode(vae, latent, side, image,
                                          error, error_size);
        if (vae) zimage_vae_gpu_release(vae);
    } else {
        qwen_weights *decoder = qwen_weights_open(path, error, error_size);
        ok = decoder && zimage_vae_decode(decoder, latent, side, image, NULL, NULL);
        if (decoder) qwen_weights_close(decoder);
        if (!ok && error && !*error)
            fail(error, error_size, "the decoder failed");
    }
    free(latent);
    return ok;
}
