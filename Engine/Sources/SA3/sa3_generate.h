#ifndef SA3_GENERATE_H
#define SA3_GENERATE_H

#include <stddef.h>
#include <stdint.h>

/* Text to sound effect, end to end: tokenize, encode, denoise, decode.
 *
 * The model is distilled, so it wants eight steps and no classifier-free
 * guidance — there is no guidance scale to expose because turning one on makes
 * the output worse, not better. */

typedef struct h3_gpu h3_gpu;
typedef struct sa3 sa3;

#define SA3_SAMPLE_RATE 44100
#define SA3_MAX_SECONDS 120.0f
#define SA3_DEFAULT_STEPS 8

typedef struct {
    const char *prompt;
    float seconds;    /* trimmed to this length; the model rounds up internally */
    int steps;        /* 0 selects the distilled default */
    uint64_t seed;
} sa3_request;

/* Fires once per denoising step so a caller can show progress; the decode
 * that follows is a single further stage. */
typedef void (*sa3_progress)(int completed, int total, void *opaque);

/* `gpu` may be NULL, in which case the transformer runs on the CPU: correct
 * but roughly fourteen times slower. */
sa3 *sa3_load(const char *package_directory, h3_gpu *gpu, char *error,
              size_t error_size);
void sa3_free(sa3 *sa3);

/* Writes interleaved-free stereo [2, samples] the caller owns and frees.
 * Values are unclamped; a writer is expected to clip rather than normalise,
 * which is what the reference does. */
int sa3_generate(sa3 *sa3, const sa3_request *request, sa3_progress progress,
                 void *opaque, float **audio, int *samples, char *error,
                 size_t error_size);

#endif
