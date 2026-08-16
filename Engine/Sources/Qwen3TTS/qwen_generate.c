#include "qwen_generate.h"

#include "qwen_codec.h"
#include "qwen_gpu.h"
#include "qwen_predictor.h"
#include "qwen_speaker.h"
#include "qwen_talker.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* From the release's config.json. The tts_* ids index the text vocabulary,
 * the codec_* ids the talker's own 3072-entry table. */
#define TTS_PAD    151671u
#define TTS_BOS    151672u
#define TTS_EOS    151673u
#define CODEC_PAD  2148u
#define CODEC_BOS  2149u
#define CODEC_EOS  2150u
#define THINK      2154u
#define THINK_BOS  2156u
#define THINK_EOS  2157u

/* Group 0's head spans 3072 entries but only the first 2048 are codes; the
 * rest are control tokens the sampler must not choose. The reference
 * suppresses exactly that range, sparing the end-of-speech marker. */
#define CODE_LIMIT 2048
#define MIN_FRAMES 2          /* the reference's min_new_tokens */

#define WIDTH QWEN_TALKER_WIDTH

struct qwen_tts {
    qwen_talker *talker;
    qwen_predictor *predictor;
    qwen_codec *codec;
    qwen_speaker *speaker;
    /* Set together or not at all. When they are set the transformers run on the
     * GPU and the CPU talker is kept only for its embedding tables, which is
     * why its own cache is loaded at one token — calling qwen_talker_forward on
     * it would then fail with a clear message rather than run a second copy. */
    qwen_gpu_talker *gpu_talker;
    qwen_gpu_predictor *gpu_predictor;
};

/* The four calls the loop makes into whichever pair is in use. Everything else
 * — the embedding tables, the codec decoder, the speaker encoder — is the same
 * either way. */
static void talker_reset(qwen_tts *tts) {
    if (tts->gpu_talker) qwen_gpu_talker_reset(tts->gpu_talker);
    else qwen_talker_reset(tts->talker);
}

static int talker_forward(qwen_tts *tts, const float *embeddings, int tokens,
                          float *hidden, float *logits, char *error,
                          size_t error_size) {
    return tts->gpu_talker
        ? qwen_gpu_talker_forward(tts->gpu_talker, embeddings, tokens, hidden,
                                  logits, error, error_size)
        : qwen_talker_forward(tts->talker, embeddings, tokens, hidden, logits,
                              error, error_size);
}

static int predictor_run(qwen_tts *tts, const float *state, const float *next,
                         uint32_t *groups, float temperature, uint64_t *seed,
                         char *error, size_t error_size) {
    return tts->gpu_predictor
        ? qwen_gpu_predictor_run(tts->gpu_predictor, state, next, groups,
                                 temperature, seed, error, error_size)
        : qwen_predictor_run(tts->predictor, state, next, groups, temperature,
                             seed, error, error_size);
}

qwen_tts *qwen_tts_load(const char *talker_path, const char *predictor_path,
                        const char *codec_path, const char *speaker_path,
                        int max_tokens, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    qwen_tts *tts = calloc(1, sizeof(*tts));
    if (!tts) {
        snprintf(error, error_size, "out of memory");
        return NULL;
    }
    tts->talker = qwen_talker_load(talker_path, max_tokens, error, error_size);
    if (tts->talker)
        tts->predictor = qwen_predictor_load(predictor_path, error, error_size);
    if (tts->predictor)
        tts->codec = qwen_codec_load(codec_path, error, error_size);
    if (tts->codec)
        tts->speaker = qwen_speaker_load(speaker_path, error, error_size);
    if (!tts->speaker) {
        qwen_tts_free(tts);
        return NULL;
    }
    return tts;
}

qwen_tts *qwen_tts_load_metal(const char *talker_path,
                              const char *predictor_path,
                              const char *codec_path, const char *speaker_path,
                              const char *shader_path, int max_tokens,
                              char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    /* One token, because this copy is only ever asked for embeddings — see the
     * note on the struct. It saves the 224 KB a token costs across 28 layers,
     * which for a long utterance is most of a hundred megabytes. */
    qwen_tts *tts = qwen_tts_load(talker_path, predictor_path, codec_path,
                                  speaker_path, 1, error, error_size);
    if (!tts) return NULL;
    tts->gpu_talker = qwen_gpu_talker_load(talker_path, shader_path, max_tokens,
                                           error, error_size);
    if (tts->gpu_talker)
        tts->gpu_predictor = qwen_gpu_predictor_load(predictor_path, shader_path,
                                                     error, error_size);
    if (!tts->gpu_predictor ||
        !qwen_codec_use_metal(tts->codec, shader_path, error, error_size)) {
        qwen_tts_free(tts);
        return NULL;
    }
    return tts;
}

void qwen_tts_free(qwen_tts *tts) {
    if (!tts) return;
    qwen_gpu_talker_free(tts->gpu_talker);
    qwen_gpu_predictor_free(tts->gpu_predictor);
    qwen_talker_free(tts->talker);
    qwen_predictor_free(tts->predictor);
    qwen_codec_free(tts->codec);
    qwen_speaker_free(tts->speaker);
    free(tts);
}

static float next_uniform(uint64_t *state) {
    uint64_t value = *state ? *state : 0x9E3779B97F4A7C15ull;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    return (float)((value >> 11) * (1.0 / 9007199254740992.0));
}

/* Chooses group 0 from the talker's logits.
 *
 * Only the 2048 codes and the end marker are eligible; everything else in the
 * 3072-wide head is a control token. A code already used is penalised, which
 * is what keeps the model from settling into a buzz. */
static uint32_t sample_group0(const float *logits, const uint8_t *used,
                              int allow_end, float temperature, int top_k,
                              float penalty, uint64_t *seed) {
    float scratch[QWEN_TALKER_VOCAB];   /* local: two generations may overlap */
    int count = 0;
    int indices[QWEN_TALKER_VOCAB];
    for (int index = 0; index < CODE_LIMIT; index++) {
        float value = logits[index];
        if (penalty > 1.0f && used[index])
            value = value > 0.0f ? value / penalty : value * penalty;
        scratch[count] = value;
        indices[count++] = index;
    }
    if (allow_end) {
        scratch[count] = logits[CODEC_EOS];
        indices[count++] = (int)CODEC_EOS;
    }

    int best = 0;
    for (int index = 1; index < count; index++)
        if (scratch[index] > scratch[best]) best = index;
    if (temperature <= 0.0f) return (uint32_t)indices[best];

    /* top-k by repeated selection: k is 50 by default, so a partial scan
     * costs less than sorting 2049 entries. */
    int limit = top_k > 0 && top_k < count ? top_k : count;
    float threshold = -INFINITY;
    if (limit < count) {
        float *copy = malloc((size_t)count * sizeof(float));
        memcpy(copy, scratch, (size_t)count * sizeof(float));
        for (int round = 0; round < limit; round++) {
            int pick = 0;
            for (int index = 1; index < count; index++)
                if (copy[index] > copy[pick]) pick = index;
            threshold = copy[pick];
            copy[pick] = -INFINITY;
        }
        free(copy);
    }

    const float highest = scratch[best];
    float sum = 0.0f;
    for (int index = 0; index < count; index++) {
        if (scratch[index] < threshold) { scratch[index] = 0.0f; continue; }
        scratch[index] = expf((scratch[index] - highest) / temperature);
        sum += scratch[index];
    }
    float target = next_uniform(seed) * sum, running = 0.0f;
    for (int index = 0; index < count; index++) {
        running += scratch[index];
        if (running >= target) return (uint32_t)indices[index];
    }
    return (uint32_t)indices[best];
}

int qwen_tts_generate(qwen_tts *tts, const qwen_tts_request *request,
                      float **audio, int *samples,
                      qwen_tts_progress progress, void *opaque,
                      char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tts || !request || !audio || !samples || !request->text_ids ||
        !request->reference) {
        snprintf(error, error_size, "invalid arguments");
        return 0;
    }
    /* three role tokens, one first text token, five closing template tokens */
    if (request->text_count < 9) {
        snprintf(error, error_size, "the prompt is too short to be templated");
        return 0;
    }
    const int max_frames = request->max_frames > 0 ? request->max_frames : 512;

    float *speaker = malloc(WIDTH * sizeof(float));
    if (!speaker || !qwen_speaker_embed(tts->speaker, request->reference,
                                        request->reference_samples, speaker,
                                        error, error_size)) {
        free(speaker);
        return 0;
    }

    /* ---- the prompt ---------------------------------------------------- */
    float *pad_bos = malloc(2 * WIDTH * sizeof(float));
    float *codec7 = malloc(7 * WIDTH * sizeof(float));
    float *prefill = malloc(10 * WIDTH * sizeof(float));
    if (!pad_bos || !codec7 || !prefill) {
        free(speaker); free(pad_bos); free(codec7); free(prefill);
        snprintf(error, error_size, "out of memory");
        return 0;
    }
    const uint32_t pad_bos_ids[2] = {TTS_PAD, TTS_BOS};
    const uint32_t head_ids[4] = {THINK, THINK_BOS, request->language_id,
                                  THINK_EOS};
    const uint32_t tail_ids[2] = {CODEC_PAD, CODEC_BOS};
    int fine = qwen_talker_embed_text(tts->talker, pad_bos_ids, 2, pad_bos,
                                      error, error_size) &&
               qwen_talker_embed_codec(tts->talker, head_ids, 4, codec7,
                                       error, error_size) &&
               qwen_talker_embed_codec(tts->talker, tail_ids, 2,
                                       codec7 + 5 * WIDTH, error, error_size) &&
               qwen_talker_embed_text(tts->talker, request->text_ids, 3,
                                      prefill, error, error_size) &&
               qwen_talker_embed_text(tts->talker, request->text_ids + 3, 1,
                                      prefill + 9 * WIDTH, error, error_size);
    if (!fine) {
        free(speaker); free(pad_bos); free(codec7); free(prefill);
        return 0;
    }
    memcpy(codec7 + 4 * WIDTH, speaker, WIDTH * sizeof(float));

    /* five pads then bos, over the first six codec positions */
    for (int position = 0; position < 6; position++) {
        const float *text = position < 5 ? pad_bos : pad_bos + WIDTH;
        float *target = prefill + (size_t)(3 + position) * WIDTH;
        for (int channel = 0; channel < WIDTH; channel++)
            target[channel] = text[channel] +
                codec7[(size_t)position * WIDTH + channel];
    }
    /* the tenth position already holds the first text token; add codec_bos */
    for (int channel = 0; channel < WIDTH; channel++)
        prefill[9 * WIDTH + channel] += codec7[6 * WIDTH + channel];

    /* the text stream after the first token, then the end marker */
    const int trailing_count = request->text_count - 5 - 4 + 1;
    float *trailing = malloc((size_t)(trailing_count > 0 ? trailing_count : 1) *
                             WIDTH * sizeof(float));
    if (!trailing) {
        free(speaker); free(pad_bos); free(codec7); free(prefill);
        snprintf(error, error_size, "out of memory");
        return 0;
    }
    if (trailing_count > 1 &&
        !qwen_talker_embed_text(tts->talker, request->text_ids + 4,
                                trailing_count - 1, trailing, error, error_size)) {
        free(speaker); free(pad_bos); free(codec7); free(prefill); free(trailing);
        return 0;
    }
    {
        const uint32_t end[1] = {TTS_EOS};
        if (!qwen_talker_embed_text(tts->talker, end, 1,
                                    trailing + (size_t)(trailing_count - 1) * WIDTH,
                                    error, error_size)) {
            free(speaker); free(pad_bos); free(codec7); free(prefill); free(trailing);
            return 0;
        }
    }

    /* ---- the loop ------------------------------------------------------ */
    talker_reset(tts);
    float *hidden = malloc((size_t)10 * WIDTH * sizeof(float));
    float *logits = malloc((size_t)10 * QWEN_TALKER_VOCAB * sizeof(float));
    uint32_t *codes = malloc((size_t)QWEN_CODEC_GROUPS * max_frames *
                             sizeof(uint32_t));
    uint8_t *used = calloc(CODE_LIMIT, sizeof(uint8_t));
    float *next = malloc(WIDTH * sizeof(float));
    if (!hidden || !logits || !codes || !used || !next) {
        free(speaker); free(pad_bos); free(codec7); free(prefill); free(trailing);
        free(hidden); free(logits); free(codes); free(used); free(next);
        snprintf(error, error_size, "out of memory");
        return 0;
    }

    int ok = talker_forward(tts, prefill, 10, hidden, logits, error, error_size);
    /* the prefill's last position is the one that predicts the first frame */
    float *state = hidden + 9 * WIDTH;
    float *scores = logits + (size_t)9 * QWEN_TALKER_VOCAB;
    uint64_t seed = request->seed;

    int frames = 0;
    while (ok && frames < max_frames) {
        const uint32_t group0 = sample_group0(scores, used, frames >= MIN_FRAMES,
                                              request->temperature,
                                              request->top_k,
                                              request->repetition_penalty, &seed);
        if (group0 == CODEC_EOS) break;
        used[group0] = 1;

        if (!qwen_talker_embed_codec(tts->talker, &group0, 1, next,
                                     error, error_size)) { ok = 0; break; }
        uint32_t groups[QWEN_CODE_GROUPS - 1];
        if (!predictor_run(tts, state, next, groups, request->temperature,
                           &seed, error, error_size)) {
            ok = 0;
            break;
        }
        codes[(size_t)0 * max_frames + frames] = group0;
        for (int group = 1; group < QWEN_CODEC_GROUPS; group++)
            codes[(size_t)group * max_frames + frames] = groups[group - 1];

        /* the next input: all sixteen groups summed, plus the text stream */
        if (!qwen_predictor_accumulate(tts->predictor, groups, next,
                                       error, error_size)) { ok = 0; break; }
        const float *text = frames < trailing_count
            ? trailing + (size_t)frames * WIDTH : pad_bos;
        for (int channel = 0; channel < WIDTH; channel++)
            next[channel] += text[channel];

        frames++;
        if (progress) progress(frames, max_frames, opaque);
        ok = talker_forward(tts, next, 1, hidden, logits, error, error_size);
        state = hidden;
        scores = logits;
    }

    if (ok && frames < 1) {
        snprintf(error, error_size, "the model produced no audio frames");
        ok = 0;
    }

    if (ok) {
        /* the codec wants the groups contiguous, and the loop wrote them into
         * a buffer sized for max_frames */
        uint32_t *packed = malloc((size_t)QWEN_CODEC_GROUPS * frames *
                                  sizeof(uint32_t));
        if (packed) {
            for (int group = 0; group < QWEN_CODEC_GROUPS; group++)
                memcpy(packed + (size_t)group * frames,
                       codes + (size_t)group * max_frames,
                       (size_t)frames * sizeof(uint32_t));
            *samples = frames * QWEN_CODEC_UPSAMPLE;
            *audio = malloc((size_t)*samples * sizeof(float));
            ok = *audio && qwen_codec_decode(tts->codec, packed, frames, *audio,
                                             NULL, NULL, error, error_size);
            free(packed);
        } else {
            snprintf(error, error_size, "out of memory");
            ok = 0;
        }
    }

    free(speaker); free(pad_bos); free(codec7); free(prefill); free(trailing);
    free(hidden); free(logits); free(codes); free(used); free(next);
    return ok;
}
