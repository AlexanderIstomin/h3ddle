/* Prompt in, picture out: the whole Z-Image pipeline behind one call.
 *
 * Tokenize, encode, sample, decode. The stages are separately gated against
 * references and separately usable — see the README — but the service wants
 * one entry point, and the joins between them (the chat template, the encoder
 * tap one block short of the end, the latent's scaling) are decisions this
 * file owns rather than its caller.
 */
#ifndef ZIMAGE_GENERATE_H
#define ZIMAGE_GENERATE_H

#include <stddef.h>
#include <stdint.h>

/* Eight is what the turbo checkpoint was distilled for. */
#define ZIMAGE_DEFAULT_STEPS 8

typedef struct {
    /* Directory holding tokenizer.json, text_encoder.safetensors,
     * transformer.safetensors and vae_decoder.safetensors. */
    const char *package;
    /* h3_shaders.metal. NULL keeps everything on the CPU, which is correct
     * and roughly twenty times slower. */
    const char *shaders;
    const char *prompt;
    /* Both a multiple of 16, and their token count — (width/16) * (height/16)
     * — a multiple of 32. The two sides are independent: the reference takes
     * height and width separately and pads the token count, and this path
     * refuses what would need padding rather than rendering it wrongly. */
    int width, height;
    int steps;              /* 0 takes ZIMAGE_DEFAULT_STEPS */
    uint64_t seed;

    /* img2img. `source` is a picture to work from: 3 * width * height floats,
     * channel-major, in [-1, 1] — the same layout and range this call returns,
     * so a render can be fed straight back in. It must already be the
     * generation size; resampling belongs to the caller, which knows what the
     * user actually framed. NULL renders from the prompt alone.
     *
     * `strength` is how much of it to discard: 1 keeps nothing and is text to
     * image, 0 keeps everything and returns the picture back. It selects where
     * the schedule starts, so at eight steps it moves in eighths and the
     * effective value is rounded to the step it lands on. Ignored when
     * `source` is NULL. */
    const float *source;
    float strength;
} zimage_request;

/* Called as the run proceeds. Returning zero abandons the generation, which
 * is how the service cancels.
 *
 * `phase` names what is happening — "text encoder", "transformer", "denoise",
 * "image VAE" — because the sampler is only part of the wait: loading and
 * encoding take about two minutes before the first step, and a caller that
 * hears nothing until then shows a still bar and reads as hung.
 *
 * `step` and `steps` are within the phase, so they restart at each one. */
typedef int (*zimage_progress)(const char *phase, int step, int steps,
                               void *context);

/* `image` receives 3 * width * height floats, channel-major, in [-1, 1] —
 * the range the reference's own post-processing expects. Returns 0 on failure
 * with `error` set, and 0 with an empty `error` when the caller cancelled. */
int zimage_generate(const zimage_request *request, float *image,
                    zimage_progress progress, void *context,
                    char *error, size_t error_size);

/* Whether this build can render this frame, so a caller can refuse early
 * rather than after loading fourteen gigabytes. */
int zimage_supports_frame(int width, int height);

#endif
