#include "H3Native.h"
#include "h3_gpu.h"
#include "h3_host.h"
#include "h3_tokenizer.h"
#include "qwen_codec.h"
#include "qwen_generate.h"
#include "qwen_speaker.h"
#include "sa3_generate.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

h3_params h3ddle_h3_default_params(void) {
    h3_params params = H3_PARAMS_DEFAULT;
    return params;
}

int h3ddle_h3_prepare_metal(const char *shader_source_path, char *error,
                            size_t error_size) {
    return h3_gpu_prepare(
        shader_source_path && *shader_source_path ? shader_source_path
                                                  : "h3_shaders.metal",
        error, error_size);
}

int h3ddle_h3_frames_for_seconds(double seconds) {
    if (!isfinite(seconds) || seconds <= 0.0) return H3_DEFAULT_FRAMES;
    double requested = ceil(seconds * H3_FPS);
    if (requested > INT_MAX) requested = INT_MAX;
    return h3_align_frame_count((int)requested);
}

const char *h3ddle_h3_version(void) {
    return H3_VERSION;
}

const char *h3ddle_h3_device_name(const h3_device_info *device) {
    return device ? device->name : "";
}

const char *h3ddle_h3_device_architecture(const h3_device_info *device) {
    return device ? device->architecture : "";
}

const char *h3ddle_h3_model_layout_name(const h3_model_info *model) {
    if (!model) return "unknown";
    switch (model->layout) {
        case H3_MODEL_LAYOUT_RELEASED_DIRECTORY: return "releasedDirectory";
        case H3_MODEL_LAYOUT_OPTIMIZED_INT8_SINGLE_FILE:
            return "optimizedINT8SingleFile";
        case H3_MODEL_LAYOUT_UNKNOWN: return "unknown";
    }
    return "unknown";
}

int h3ddle_h3_model_supports_generation(const h3_model_info *model) {
    return model ? model->generation_supported : 0;
}

/* Writes 16-bit stereo PCM. The samples arrive channel-major and unclamped;
 * clipping rather than normalising matches the reference, which would
 * otherwise quietly change the level of every generation. */
static int sa3_write_wav(const char *path, const float *audio, int samples) {
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    unsigned int rate = 44100, data_bytes = (unsigned int)samples * 4u;
    unsigned int header[] = {36u + data_bytes, 16u, rate, rate * 4u, data_bytes};
    unsigned short fields[] = {1, 2, 16, 4};
    int ok = fwrite("RIFF", 1, 4, file) == 4;
    ok = ok && fwrite(&header[0], 4, 1, file) == 1;
    ok = ok && fwrite("WAVEfmt ", 1, 8, file) == 8;
    ok = ok && fwrite(&header[1], 4, 1, file) == 1;
    ok = ok && fwrite(&fields[0], 2, 1, file) == 1;   /* PCM */
    ok = ok && fwrite(&fields[1], 2, 1, file) == 1;   /* stereo */
    ok = ok && fwrite(&header[2], 4, 1, file) == 1;
    ok = ok && fwrite(&header[3], 4, 1, file) == 1;
    ok = ok && fwrite(&fields[3], 2, 1, file) == 1;   /* block align */
    ok = ok && fwrite(&fields[2], 2, 1, file) == 1;   /* bits */
    ok = ok && fwrite("data", 1, 4, file) == 4;
    ok = ok && fwrite(&header[4], 4, 1, file) == 1;
    for (int index = 0; ok && index < samples; index++)
        for (int channel = 0; channel < 2; channel++) {
            float value = audio[(size_t)channel * samples + index];
            if (value > 1.0f) value = 1.0f;
            if (value < -1.0f) value = -1.0f;
            short pcm = (short)(value * 32767.0f);
            ok = fwrite(&pcm, 2, 1, file) == 1;
        }
    fclose(file);
    return ok;
}

/* The package is kept between generations so a run of effects pays the
 * 1.8-second load once instead of every time. It is dropped the moment the
 * video model wants memory: at tens of gigabytes that model decides what
 * fits, and 1.7 GB of sound effects is not worth crowding it. */
static sa3 *sa3_cached;
static char sa3_cached_directory[1024];
static h3_gpu *sa3_cached_gpu;

void h3ddle_sa3_release(void) {
    if (!sa3_cached) return;
    sa3_free(sa3_cached);
    h3_gpu_free(sa3_cached_gpu);
    sa3_cached = NULL;
    sa3_cached_gpu = NULL;
    sa3_cached_directory[0] = '\0';
}

int h3ddle_sa3_generate(const char *package_directory, const char *prompt,
                        double seconds, int steps, unsigned long long seed,
                        const char *output_path, h3ddle_sa3_step on_step,
                        void *opaque, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!package_directory || !prompt || !output_path) {
        if (error && error_size)
            snprintf(error, error_size, "a package, prompt and destination are "
                     "required");
        return 0;
    }
    h3_gpu *gpu = sa3_cached_gpu;
    sa3 *model = sa3_cached;
    if (!model || strcmp(sa3_cached_directory, package_directory)) {
        h3ddle_sa3_release();
        /* Sharing h3.c's Metal context keeps both models on one device and
         * one allocator; without it the transformer falls back to the CPU. */
        gpu = h3_gpu_create(NULL, error, error_size);
        model = sa3_load(package_directory, gpu, error, error_size);
        if (!model) {
            h3_gpu_free(gpu);
            return 0;
        }
        sa3_cached = model;
        sa3_cached_gpu = gpu;
        snprintf(sa3_cached_directory, sizeof(sa3_cached_directory), "%s",
                 package_directory);
    }

    sa3_request request = {
        .prompt = prompt,
        .seconds = (float)seconds,
        .steps = steps,
        .seed = seed,
    };
    float *audio = NULL;
    int samples = 0;
    int ok = sa3_generate(model, &request, on_step, opaque, &audio, &samples,
                          error, error_size);
    if (ok) {
        ok = sa3_write_wav(output_path, audio, samples);
        if (!ok && error && error_size)
            snprintf(error, error_size, "cannot write %s", output_path);
    }
    free(audio);
    return ok;
}

/* ---- Qwen3-TTS ---------------------------------------------------------- */

/* Writes 16-bit mono PCM. Already clamped by the codec, so this only
 * quantises. */
static int qwen_write_wav(const char *path, const float *audio, int samples,
                          int rate) {
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    unsigned int data_bytes = (unsigned int)samples * 2u;
    unsigned int header[] = {36u + data_bytes, 16u, (unsigned int)rate,
                             (unsigned int)rate * 2u, data_bytes};
    unsigned short fields[] = {1, 1, 16, 2};
    int ok = fwrite("RIFF", 1, 4, file) == 4;
    ok = ok && fwrite(&header[0], 4, 1, file) == 1;
    ok = ok && fwrite("WAVEfmt ", 1, 8, file) == 8;
    ok = ok && fwrite(&header[1], 4, 1, file) == 1;
    ok = ok && fwrite(&fields[0], 2, 1, file) == 1;   /* PCM */
    ok = ok && fwrite(&fields[1], 2, 1, file) == 1;   /* mono */
    ok = ok && fwrite(&header[2], 4, 1, file) == 1;
    ok = ok && fwrite(&header[3], 4, 1, file) == 1;
    ok = ok && fwrite(&fields[3], 2, 1, file) == 1;   /* block align */
    ok = ok && fwrite(&fields[2], 2, 1, file) == 1;   /* bits */
    ok = ok && fwrite("data", 1, 4, file) == 4;
    ok = ok && fwrite(&header[4], 4, 1, file) == 1;
    for (int index = 0; ok && index < samples; index++) {
        float value = audio[index];
        if (value > 1.0f) value = 1.0f;
        if (value < -1.0f) value = -1.0f;
        short pcm = (short)lrintf(value * 32767.0f);
        ok = fwrite(&pcm, 2, 1, file) == 1;
    }
    fclose(file);
    return ok;
}

/* An unknown code is refused rather than defaulted: the wrong language token
 * still produces fluent speech, just in the wrong accent, so a silent
 * fallback would be a bug nobody reports. */
static int qwen_language_id(const char *code, uint32_t *language) {
    static const struct { const char *code; uint32_t id; } table[] = {
        {"en", QWEN_LANGUAGE_ENGLISH},    {"zh", QWEN_LANGUAGE_CHINESE},
        {"de", QWEN_LANGUAGE_GERMAN},     {"es", QWEN_LANGUAGE_SPANISH},
        {"fr", QWEN_LANGUAGE_FRENCH},     {"it", QWEN_LANGUAGE_ITALIAN},
        {"pt", QWEN_LANGUAGE_PORTUGUESE}, {"ru", QWEN_LANGUAGE_RUSSIAN},
        {"ja", QWEN_LANGUAGE_JAPANESE},   {"ko", QWEN_LANGUAGE_KOREAN},
    };
    if (!code) return 0;
    for (size_t index = 0; index < sizeof(table) / sizeof(table[0]); index++)
        if (!strcmp(table[index].code, code)) {
            *language = table[index].id;
            return 1;
        }
    return 0;
}

/* Held between generations on the same terms as the sound-effect package:
 * loading costs seconds, and speech is usually asked for in runs. */
static qwen_tts *qwen_cached;
static h3_tokenizer *qwen_cached_tokenizer;
static char qwen_cached_directory[1024];

void h3ddle_qwen_release(void) {
    if (!qwen_cached) return;
    qwen_tts_free(qwen_cached);
    h3_tokenizer_free(qwen_cached_tokenizer);
    qwen_cached = NULL;
    qwen_cached_tokenizer = NULL;
    qwen_cached_directory[0] = '\0';
}

/* 12.5 frames a second, and the ceiling is a stop rather than a target: the
 * model emits EOS when the text runs out, so a generous bound costs nothing. */
#define QWEN_FRAMES_PER_SECOND 12.5
#define QWEN_REFERENCE_LIMIT   (QWEN_SPEAKER_SAMPLE_RATE * 60)

int h3ddle_qwen_generate(const char *package_directory, const char *text,
                         const char *language, const char *reference_path,
                         double max_seconds, double temperature, int top_k,
                         double repetition_penalty, unsigned long long seed,
                         const char *output_path, h3ddle_qwen_frame on_frame,
                         void *opaque, double *produced_seconds,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (produced_seconds) *produced_seconds = 0.0;
    if (!package_directory || !text || !output_path) {
        if (error && error_size)
            snprintf(error, error_size, "a package, text and destination are "
                     "required");
        return 0;
    }
    if (!reference_path) {
        if (error && error_size)
            snprintf(error, error_size, "a reference clip is required: this "
                     "model has no default voice");
        return 0;
    }
    uint32_t language_id = 0;
    if (!qwen_language_id(language, &language_id)) {
        if (error && error_size)
            snprintf(error, error_size, "unsupported language \"%s\"",
                     language ? language : "");
        return 0;
    }

    char talker[1200], predictor[1200], codec[1200], speaker[1200];
    char tokenizer_path[1200];
    snprintf(talker, sizeof(talker), "%s/talker.safetensors", package_directory);
    snprintf(predictor, sizeof(predictor), "%s/code_predictor.safetensors",
             package_directory);
    snprintf(codec, sizeof(codec), "%s/codec_decoder.safetensors",
             package_directory);
    snprintf(speaker, sizeof(speaker), "%s/speaker_encoder.safetensors",
             package_directory);
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json",
             package_directory);

    const int max_frames = max_seconds > 0.0
        ? (int)(max_seconds * QWEN_FRAMES_PER_SECOND + 0.5) : 400;
    if (!qwen_cached || strcmp(qwen_cached_directory, package_directory)) {
        h3ddle_qwen_release();
        h3_tokenizer *tokenizer = h3_tokenizer_load(tokenizer_path, error,
                                                    error_size);
        if (!tokenizer) return 0;
        /* The talker's cache is sized once at load, so give it the ceiling the
         * app allows rather than this request's. */
        qwen_tts *tts = qwen_tts_load_metal(talker, predictor, codec, speaker,
                                            NULL, 2048, error, error_size);
        if (!tts) {
            h3_tokenizer_free(tokenizer);
            return 0;
        }
        qwen_cached = tts;
        qwen_cached_tokenizer = tokenizer;
        snprintf(qwen_cached_directory, sizeof(qwen_cached_directory), "%s",
                 package_directory);
    }

    /* The talker reads a chat-templated turn; h3.c's tokenizer produces the
     * ids because H3 and Qwen3-TTS share a vocabulary byte for byte. */
    size_t templated_size = strlen(text) + 128;
    char *templated = malloc(templated_size);
    if (!templated) {
        if (error && error_size) snprintf(error, error_size, "out of memory");
        return 0;
    }
    snprintf(templated, templated_size,
             "<|im_start|>assistant\n%s<|im_end|>\n<|im_start|>assistant\n",
             text);
    uint32_t *ids = NULL;
    size_t id_count = 0;
    int ok = h3_tokenizer_encode(qwen_cached_tokenizer, templated, 0, &ids,
                                 &id_count, error, error_size);
    free(templated);
    if (!ok) return 0;

    float *reference = NULL;
    int reference_samples = 0;
    if (!h3ddle_read_mono_f32(reference_path, QWEN_SPEAKER_SAMPLE_RATE,
                              QWEN_REFERENCE_LIMIT, &reference,
                              &reference_samples, error, error_size)) {
        h3_tokenizer_ids_free(ids);
        return 0;
    }

    qwen_tts_request request = {
        .text_ids = ids,
        .text_count = (int)id_count,
        .reference = reference,
        .reference_samples = reference_samples,
        .language_id = language_id,
        .temperature = (float)temperature,
        .top_k = top_k,
        .repetition_penalty = (float)repetition_penalty,
        .seed = seed,
        .max_frames = max_frames,
    };
    float *audio = NULL;
    int samples = 0;
    ok = qwen_tts_generate(qwen_cached, &request, &audio, &samples, on_frame,
                           opaque, error, error_size);
    h3_tokenizer_ids_free(ids);
    free(reference);
    if (ok) {
        ok = qwen_write_wav(output_path, audio, samples,
                            QWEN_CODEC_SAMPLE_RATE);
        if (!ok && error && error_size)
            snprintf(error, error_size, "cannot write %s", output_path);
        else if (produced_seconds)
            *produced_seconds = (double)samples / QWEN_CODEC_SAMPLE_RATE;
    }
    free(audio);
    return ok;
}
