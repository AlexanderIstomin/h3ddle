#ifndef QWEN_CODEC_H
#define QWEN_CODEC_H

#include <stddef.h>
#include <stdint.h>

/* The Qwen3-TTS codec decoder: sixteen code groups per 12.5 Hz frame in,
 * 24 kHz audio out, 1920 samples per frame.
 *
 * Five stages, and the last is most of the code:
 *
 *   1. dequantise   sixteen codebook lookups summed, semantic and acoustic
 *                   projected separately, giving a 512-wide latent
 *   2. pre_conv     512 -> 1024
 *   3. transformer  8 layers, 512 wide with attention at 1024, layer-scaled
 *                   residuals and a 72-frame sliding window
 *   4. upsample     two ConvNeXt stages, x2 each
 *   5. vocoder      four transposed-conv stages at rates 8, 5, 4, 3 with
 *                   channels 1536 -> 768 -> 384 -> 192 -> 96, three residual
 *                   units apiece, SnakeBeta throughout, clamped to [-1, 1]
 *
 * 2*2*8*5*4*3 = 1920.
 *
 * Every convolution is causal — padded on the left only — so no output sample
 * depends on a later frame, which is what makes streaming possible later.
 *
 * The codebooks in the package are already folded from the released EMA sums;
 * see Scripts/convert-qwen3-tts-package.py. */

#define QWEN_CODEC_GROUPS      16
#define QWEN_CODEC_UPSAMPLE    1920
#define QWEN_CODEC_SAMPLE_RATE 24000

typedef struct qwen_codec qwen_codec;

qwen_codec *qwen_codec_load(const char *path, char *error, size_t error_size);
void qwen_codec_free(qwen_codec *codec);

/* Reports progress through the vocoder stages, which dominate a long decode. */
typedef void (*qwen_codec_progress)(int completed, int total, void *opaque);

/* Decodes `frames` frames of `codes`, laid out as QWEN_CODEC_GROUPS rows of
 * `frames` entries, into `audio` — frames * QWEN_CODEC_UPSAMPLE samples.
 *
 * Peak working memory is about 1.5 MB per frame, so a long utterance should be
 * decoded in chunks with a little left context rather than in one call. */
int qwen_codec_decode(qwen_codec *codec, const uint32_t *codes, int frames,
                      float *audio, qwen_codec_progress progress, void *opaque,
                      char *error, size_t error_size);

/* Reports a named intermediate stage, for checking against the reference. */
typedef void (*qwen_codec_probe)(const char *stage, const float *values,
                                 int channels, int length, void *opaque);
void qwen_codec_set_probe(qwen_codec *codec, qwen_codec_probe probe,
                          void *opaque);

#endif
