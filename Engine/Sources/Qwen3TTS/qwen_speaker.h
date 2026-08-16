#ifndef QWEN_SPEAKER_H
#define QWEN_SPEAKER_H

#include <stddef.h>
#include <stdint.h>

/* The speaker encoder: a reference clip in, 1024 numbers out.
 *
 * Since cloning conditions on this embedding alone rather than on the
 * reference clip's codes, these 1024 numbers are the entire carrier of the
 * voice — which is also why the Mimi codec encoder is not in the package.
 *
 * Two halves, and the first is the one with parameters that live in the
 * reference's Python rather than in any config file: a 128-band mel over a
 * 1024-point FFT, hop 256, fmax 12000, with a *slaney*-normalised filterbank,
 * reflect padding of (n_fft - hop) / 2, magnitude floored at 1e-9 inside the
 * square root and a log clamped at 1e-5. Then ECAPA-TDNN — res2net blocks,
 * squeeze-excitation, attentive statistics pooling — whose every convolution
 * pads by reflection rather than with zeros. */

#define QWEN_SPEAKER_DIM         1024
#define QWEN_SPEAKER_SAMPLE_RATE 24000

typedef struct qwen_speaker qwen_speaker;

qwen_speaker *qwen_speaker_load(const char *path, char *error,
                                size_t error_size);
void qwen_speaker_free(qwen_speaker *speaker);

/* `audio` is mono at QWEN_SPEAKER_SAMPLE_RATE; `embedding` receives
 * QWEN_SPEAKER_DIM floats. A few seconds of speech is enough — the pooling
 * averages over the whole clip. */
int qwen_speaker_embed(qwen_speaker *speaker, const float *audio, int samples,
                       float *embedding, char *error, size_t error_size);

/* Exposed so the mel can be checked on its own; `mel` receives
 * 128 * frames floats, band-major. Returns the frame count, or 0. */
int qwen_speaker_mel(const float *audio, int samples, float **mel,
                     char *error, size_t error_size);

#endif
