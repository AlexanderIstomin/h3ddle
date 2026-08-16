#ifndef QWEN_GENERATE_H
#define QWEN_GENERATE_H

#include <stddef.h>
#include <stdint.h>

/* Text and a reference clip in, 24 kHz speech out.
 *
 * The talker's input is the sum of two streams that advance at different
 * rates. The codec stream carries one summed set of sixteen group embeddings
 * per audio frame; the text stream feeds one token per frame and then pads
 * once it runs out. The prompt is the ten positions where they first meet:
 *
 *   0..2   chat-template role tokens          (text only)
 *   3..7   five pads, over think / think_bos / language / think_eos and the
 *          speaker embedding                  (both)
 *   8      tts_bos over codec_pad             (both)
 *   9      the first text token over codec_bos
 *
 * From there each frame samples group 0 from the talker, gets groups 1..15
 * from the code predictor, sums all sixteen embeddings with the next text
 * token, and feeds that back.
 *
 * Tokenising is the caller's job: pass the ids for
 * "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n", which
 * the engine's own tokenizer produces — H3 and Qwen3-TTS share a vocabulary
 * byte for byte. */

#define QWEN_LANGUAGE_ENGLISH    2050
#define QWEN_LANGUAGE_CHINESE    2055
#define QWEN_LANGUAGE_GERMAN     2053
#define QWEN_LANGUAGE_SPANISH    2054
#define QWEN_LANGUAGE_FRENCH     2061
#define QWEN_LANGUAGE_ITALIAN    2070
#define QWEN_LANGUAGE_PORTUGUESE 2071
#define QWEN_LANGUAGE_RUSSIAN    2069
#define QWEN_LANGUAGE_JAPANESE   2058
#define QWEN_LANGUAGE_KOREAN     2064

typedef struct qwen_tts qwen_tts;

qwen_tts *qwen_tts_load(const char *talker_path, const char *predictor_path,
                        const char *codec_path, const char *speaker_path,
                        int max_tokens, char *error, size_t error_size);
void qwen_tts_free(qwen_tts *tts);

typedef struct {
    const uint32_t *text_ids;   /* the chat-templated sequence */
    int text_count;
    const float *reference;     /* mono at 24 kHz */
    int reference_samples;
    uint32_t language_id;
    float temperature;          /* at or below zero takes the argmax */
    int top_k;                  /* 0 for no restriction */
    float repetition_penalty;   /* 1.0 for none; the reference uses 1.05 */
    uint64_t seed;
    int max_frames;             /* 12.5 per second of speech */
} qwen_tts_request;

/* Reports frames produced so far; `total` is the ceiling, not a prediction. */
typedef void (*qwen_tts_progress)(int frames, int total, void *opaque);

/* On success `audio` is a caller-owned buffer of `samples` floats at 24 kHz,
 * released with free(). */
int qwen_tts_generate(qwen_tts *tts, const qwen_tts_request *request,
                      float **audio, int *samples,
                      qwen_tts_progress progress, void *opaque,
                      char *error, size_t error_size);

#endif
