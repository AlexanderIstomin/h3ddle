#include "sa3_generate.h"

#include "sa3_decoder.h"
#include "sa3_dit.h"
#include "sa3_dit_gpu.h"
#include "sa3_text.h"
#include "sa3_tokenizer.h"
#include "h3_safetensors.h"

#include <Accelerate/Accelerate.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH SA3_TEXT_WIDTH
#define PROMPT_TOKENS SA3_TEXT_MAX_TOKENS
#define CONTEXT_TOKENS (PROMPT_TOKENS + 1)  /* the prompt plus the duration */
#define FOURIER_DIM 256
#define FOURIER_MIN 0.5f
#define FOURIER_MAX 10000.0f
#define DURATION_MAX 384.0f
#define CHUNK 8
#define OVERLAP 2

/* The schedule is linear in time, then warped through log-SNR space, which
 * spends most of the eight steps where the sample is still mostly noise. */
#define LOGSNR_ANCHOR (-6.2f)
#define LOGSNR_END 2.0f

struct sa3 {
    sa3_tokenizer *tokenizer;
    sa3_text *text;
    sa3_dit *dit_cpu;
    sa3_dit_gpu *dit_gpu;
    sa3_decoder *decoder;

    /* Conditioner weights, which live alongside the transformer. */
    float *padding_embedding;   /* [WIDTH] */
    float *duration_weight;     /* [WIDTH, FOURIER_DIM] */
    float *duration_bias;       /* [WIDTH] */
    float fourier[FOURIER_DIM / 2];
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static float *load_conditioner(const h3_st_header *header, const char *name,
                               uint64_t expected, char *error,
                               size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor || h3_st_tensor_elements(tensor) != expected) {
        fail(error, error_size, "%s is missing from the transformer", name);
        return NULL;
    }
    float *values = malloc((size_t)expected * sizeof(*values));
    if (!values) {
        fail(error, error_size, "out of memory reading %s", name);
        return NULL;
    }
    if (tensor->dtype == H3_DTYPE_F32) {
        if (h3_st_read_data(header, tensor, values,
                            (size_t)expected * sizeof(*values), error,
                            error_size))
            return values;
        free(values);
        return NULL;
    }
    uint16_t *raw = malloc((size_t)expected * sizeof(*raw));
    if (!raw || !h3_st_read_data(header, tensor, raw,
                                 (size_t)expected * sizeof(*raw), error,
                                 error_size)) {
        free(raw);
        free(values);
        return NULL;
    }
    vImage_Buffer source = {raw, 1, (vImagePixelCount)expected,
                            (size_t)expected * sizeof(*raw)};
    vImage_Buffer destination = {values, 1, (vImagePixelCount)expected,
                                 (size_t)expected * sizeof(*values)};
    vImageConvert_Planar16FtoPlanarF(&source, &destination, 0);
    free(raw);
    return values;
}

sa3 *sa3_load(const char *package_directory, h3_gpu *gpu, char *error,
              size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!package_directory) {
        fail(error, error_size, "no package directory");
        return NULL;
    }
    sa3 *sa3 = calloc(1, sizeof(*sa3));
    if (!sa3) {
        fail(error, error_size, "out of memory");
        return NULL;
    }
    char path[1024];

    snprintf(path, sizeof(path), "%s/tokenizer.json", package_directory);
    sa3->tokenizer = sa3_tokenizer_load(path, error, error_size);
    if (!sa3->tokenizer) goto fail_out;

    snprintf(path, sizeof(path), "%s/text_encoder.safetensors",
             package_directory);
    sa3->text = sa3_text_load(path, error, error_size);
    if (!sa3->text) goto fail_out;

    snprintf(path, sizeof(path), "%s/decoder.safetensors", package_directory);
    sa3->decoder = sa3_decoder_load(path, error, error_size);
    if (!sa3->decoder) goto fail_out;

    snprintf(path, sizeof(path), "%s/dit.safetensors", package_directory);
    if (gpu) {
        sa3->dit_gpu = sa3_dit_gpu_load(gpu, path, error, error_size);
        if (!sa3->dit_gpu) goto fail_out;
    } else {
        sa3->dit_cpu = sa3_dit_load(path, error, error_size);
        if (!sa3->dit_cpu) goto fail_out;
    }

    h3_st_header header;
    if (!h3_st_read_header(path, &header, error, error_size)) goto fail_out;
    sa3->padding_embedding = load_conditioner(&header, "cond.padding_embedding",
                                              WIDTH, error, error_size);
    sa3->duration_weight = load_conditioner(&header, "cond.seconds_total_weight",
                                            (uint64_t)WIDTH * FOURIER_DIM,
                                            error, error_size);
    sa3->duration_bias = load_conditioner(&header, "cond.seconds_total_bias",
                                          WIDTH, error, error_size);
    h3_st_free_header(&header);
    if (!sa3->padding_embedding || !sa3->duration_weight || !sa3->duration_bias)
        goto fail_out;

    int half = FOURIER_DIM / 2;
    for (int index = 0; index < half; index++) {
        float ramp = half > 1 ? (float)index / (float)(half - 1) : 0.0f;
        sa3->fourier[index] =
            expf(ramp * (logf(FOURIER_MAX) - logf(FOURIER_MIN)) +
                 logf(FOURIER_MIN)) * 2.0f * (float)M_PI;
    }
    return sa3;

fail_out:
    sa3_free(sa3);
    return NULL;
}

void sa3_free(struct sa3 *sa3) {
    if (!sa3) return;
    sa3_tokenizer_free(sa3->tokenizer);
    sa3_text_free(sa3->text);
    sa3_dit_free(sa3->dit_cpu);
    sa3_dit_gpu_free(sa3->dit_gpu);
    sa3_decoder_free(sa3->decoder);
    free(sa3->padding_embedding);
    free(sa3->duration_weight);
    free(sa3->duration_bias);
    free(sa3);
}

/* ---- noise ------------------------------------------------------------- */

/* SplitMix64: a small, well-distributed generator, so a seed reproduces a
 * result without depending on the host's rand(). It will not reproduce the
 * reference implementation's noise, which uses a different generator; the
 * seeds simply name different draws. */
static uint64_t splitmix(uint64_t *state) {
    uint64_t value = (*state += 0x9E3779B97F4A7C15ull);
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

static void fill_normal(float *values, size_t count, uint64_t *state) {
    for (size_t index = 0; index < count; index += 2) {
        /* Box-Muller, nudged off zero so the logarithm stays finite. */
        float first = (float)((splitmix(state) >> 11) * 0x1.0p-53) + 1e-7f;
        float second = (float)((splitmix(state) >> 11) * 0x1.0p-53);
        float radius = sqrtf(-2.0f * logf(first));
        float angle = 2.0f * (float)M_PI * second;
        values[index] = radius * cosf(angle);
        if (index + 1 < count) values[index + 1] = radius * sinf(angle);
    }
}

/* ---- schedule ---------------------------------------------------------- */

static void build_schedule(float *sigmas, int steps) {
    for (int index = 0; index <= steps; index++) {
        float t = 1.0f - (float)index / (float)steps;
        if (t <= 0.0f) {
            sigmas[index] = 0.0f;
        } else if (t >= 1.0f) {
            sigmas[index] = 1.0f;
        } else {
            float logsnr = LOGSNR_END - t * (LOGSNR_END - LOGSNR_ANCHOR);
            sigmas[index] = 1.0f / (1.0f + expf(logsnr));
        }
    }
    /* The warp moves the first point; the run still starts at full noise. */
    sigmas[0] = 1.0f;
}

/* Continues the same bar the denoising steps were reported on. */
static void decode_relay(int completed, int total, void *opaque) {
    struct relay { sa3_progress fn; void *opaque; int offset; int total; };
    struct relay *relay = opaque;
    (void)total;
    if (relay && relay->fn)
        relay->fn(relay->offset + completed, relay->total, relay->opaque);
}

/* ---- generation -------------------------------------------------------- */

int sa3_generate(struct sa3 *sa3, const sa3_request *request,
                 sa3_progress progress, void *opaque, float **audio,
                 int *samples, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (audio) *audio = NULL;
    if (samples) *samples = 0;
    if (!sa3 || !request || !request->prompt || !audio || !samples) {
        fail(error, error_size, "invalid arguments for generation");
        return 0;
    }
    float seconds = request->seconds;
    if (!(seconds > 0.0f)) seconds = 5.0f;
    if (seconds > SA3_MAX_SECONDS) seconds = SA3_MAX_SECONDS;
    int steps = request->steps > 0 ? request->steps : SA3_DEFAULT_STEPS;

    /* One latent covers 4096 samples; round up and trim the tail later. */
    int frames = (int)ceilf(seconds * SA3_SAMPLE_RATE /
                            (float)SA3_SAMPLES_PER_LATENT);
    if (frames < 2) frames = 2;

    /* --- text --- */
    uint32_t ids[PROMPT_TOKENS];
    int token_count = 0;
    if (!sa3_tokenizer_encode(sa3->tokenizer, request->prompt, ids,
                              PROMPT_TOKENS, &token_count, error, error_size))
        return 0;
    if (token_count < 1) {
        /* An empty prompt is legitimate — the model generates unconditionally
         * — but attention needs at least one visible position. */
        ids[0] = 0;
        token_count = 1;
    }

    float *embeddings = malloc((size_t)PROMPT_TOKENS * WIDTH * sizeof(float));
    float *context = malloc((size_t)CONTEXT_TOKENS * WIDTH * sizeof(float));
    if (!embeddings || !context) {
        free(embeddings);
        free(context);
        fail(error, error_size, "out of memory conditioning the prompt");
        return 0;
    }
    if (!sa3_text_encode(sa3->text, ids, token_count, embeddings, error,
                         error_size)) {
        free(embeddings);
        free(context);
        return 0;
    }

    /* Real positions keep their embedding; padding is replaced wholesale by
     * the learned vector, which is what the model was trained to skip over. */
    for (int token = 0; token < PROMPT_TOKENS; token++) {
        const float *source = token < token_count
            ? embeddings + (size_t)token * WIDTH : sa3->padding_embedding;
        memcpy(context + (size_t)token * WIDTH, source, WIDTH * sizeof(float));
    }
    free(embeddings);

    /* The duration enters as its own conditioning token, and again as the
     * global vector that drives every block's modulation. */
    float normalized = seconds / DURATION_MAX;
    if (normalized < 0.0f) normalized = 0.0f;
    if (normalized > 1.0f) normalized = 1.0f;
    float features[FOURIER_DIM];
    int half = FOURIER_DIM / 2;
    for (int index = 0; index < half; index++) {
        float angle = normalized * sa3->fourier[index];
        features[index] = cosf(angle);
        features[half + index] = sinf(angle);
    }
    float *duration = context + (size_t)PROMPT_TOKENS * WIDTH;
    for (int channel = 0; channel < WIDTH; channel++) {
        const float *row = sa3->duration_weight + (size_t)channel * FOURIER_DIM;
        float sum = sa3->duration_bias[channel];
        for (int index = 0; index < FOURIER_DIM; index++)
            sum += row[index] * features[index];
        duration[channel] = sum;
    }

    /* --- denoise --- */
    float *sigmas = malloc((size_t)(steps + 1) * sizeof(float));
    size_t latent_count = (size_t)SA3_LATENT_CHANNELS * frames;
    float *x = malloc(latent_count * sizeof(float));
    float *velocity = malloc(latent_count * sizeof(float));
    float *injected = malloc(latent_count * sizeof(float));
    if (!sigmas || !x || !velocity || !injected) {
        free(sigmas); free(x); free(velocity); free(injected); free(context);
        fail(error, error_size, "out of memory denoising");
        return 0;
    }
    build_schedule(sigmas, steps);

    uint64_t state = request->seed ? request->seed : 0x5A3ull;
    fill_normal(x, latent_count, &state);

    // Decode is a real share of the wall time on long audio, so the bar
    // spans both stages: reaching 100% at the end of denoising left it
    // sitting there while the decoder worked.
    int windows = frames > (CHUNK + 2 * OVERLAP)
      ? sa3_decoder_window_count(frames, CHUNK, OVERLAP) : 1;
    int total_units = steps + windows;

    int ok = 1;
    if (sa3->dit_gpu)
        ok = sa3_dit_gpu_set_context(sa3->dit_gpu, context, CONTEXT_TOKENS,
                                     error, error_size);
    for (int step = 0; step < steps && ok; step++) {
        float sigma = sigmas[step], next = sigmas[step + 1];
        ok = sa3->dit_gpu
            ? sa3_dit_gpu_forward(sa3->dit_gpu, x, frames, duration, sigma,
                                  velocity, error, error_size)
            : sa3_dit_forward(sa3->dit_cpu, x, frames, context, CONTEXT_TOKENS,
                              duration, sigma, velocity, error, error_size);
        if (!ok) break;
        /* Rectified flow: step to the denoised estimate, then re-noise to the
         * next level rather than easing down. */
        for (size_t index = 0; index < latent_count; index++)
            x[index] -= sigma * velocity[index];
        if (step < steps - 1 && next > 0.0f) {
            fill_normal(injected, latent_count, &state);
            for (size_t index = 0; index < latent_count; index++)
                x[index] = (1.0f - next) * x[index] + next * injected[index];
        }
        if (progress) progress(step + 1, total_units, opaque);
    }
    free(sigmas);
    free(velocity);
    free(injected);
    free(context);
    if (!ok) {
        free(x);
        return 0;
    }

    /* --- decode --- */
    int patch_count = frames * SA3_PATCHES_PER_LATENT;
    float *patches = malloc((size_t)patch_count * SA3_PATCH_CHANNELS *
                            sizeof(float));
    if (!patches) {
        free(x);
        fail(error, error_size, "out of memory decoding");
        return 0;
    }
    struct { sa3_progress fn; void *opaque; int offset; int total; } relay = {
        progress, opaque, steps, total_units
    };
    int kernel = CHUNK + 2 * OVERLAP;
    if (frames > kernel)
        ok = sa3_decoder_run_chunked(sa3->decoder, x, frames, CHUNK, OVERLAP,
                                     patches, decode_relay, &relay,
                                     error, error_size);
    else if (frames % 2 == 0)
        ok = sa3_decoder_run(sa3->decoder, x, frames, patches, error,
                             error_size);
    else
        /* An odd count below the usual window still decodes if every window
         * stays even, which a smaller kernel guarantees. */
        ok = sa3_decoder_run_chunked(sa3->decoder, x, frames, 2, 2, patches,
                                     decode_relay, &relay, error, error_size);
    free(x);
    if (!ok) {
        free(patches);
        return 0;
    }

    int produced = patch_count * 256;
    float *result = malloc((size_t)2 * produced * sizeof(float));
    if (!result) {
        free(patches);
        fail(error, error_size, "out of memory unpacking audio");
        return 0;
    }
    sa3_decoder_unpatch(patches, patch_count, result);
    free(patches);

    /* Trim the rounding-up back off so the caller gets what it asked for. */
    int wanted = (int)(seconds * SA3_SAMPLE_RATE + 0.5f);
    if (wanted > produced) wanted = produced;
    if (wanted < produced) {
        float *trimmed = malloc((size_t)2 * wanted * sizeof(float));
        if (!trimmed) {
            free(result);
            fail(error, error_size, "out of memory trimming audio");
            return 0;
        }
        for (int channel = 0; channel < 2; channel++)
            memcpy(trimmed + (size_t)channel * wanted,
                   result + (size_t)channel * produced,
                   (size_t)wanted * sizeof(float));
        free(result);
        result = trimmed;
    }

    *audio = result;
    *samples = wanted;
    return 1;
}
