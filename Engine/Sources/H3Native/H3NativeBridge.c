#include "H3Native.h"
#include "zimage_generate.h"
#include "h3_gpu.h"
#include "h3_host.h"
#include "h3_tokenizer.h"
#include "qwen_codec.h"
#include "qwen_generate.h"
#include "qwen_speaker.h"
#include "ltx_audio.h"
#include "ltx_generate.h"
#include "h3_avwriter.h"
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

/* Reads a saved voice. Wrong-sized files are refused rather than padded: a
 * truncated one would speak in a voice nobody chose. */
static int qwen_read_embedding(const char *path, float *embedding, char *error,
                               size_t error_size) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        if (error && error_size)
            snprintf(error, error_size, "cannot read the voice %s", path);
        return 0;
    }
    const size_t read = fread(embedding, sizeof(float), QWEN_SPEAKER_DIM, file);
    int extra = fgetc(file) != EOF;
    fclose(file);
    if (read != QWEN_SPEAKER_DIM || extra) {
        if (error && error_size)
            snprintf(error, error_size, "%s is not a voice: expected %d floats",
                     path, QWEN_SPEAKER_DIM);
        return 0;
    }
    return 1;
}

int h3ddle_qwen_write_embedding(const char *package_directory,
                                const char *reference_path,
                                const char *embedding_path,
                                char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!package_directory || !reference_path || !embedding_path) {
        if (error && error_size)
            snprintf(error, error_size, "a package, clip and destination are "
                     "required");
        return 0;
    }
    char speaker[1200];
    snprintf(speaker, sizeof(speaker), "%s/speaker_encoder.safetensors",
             package_directory);
    /* Loaded and dropped per call: 35 MB, used once when a voice is added. */
    qwen_speaker *encoder = qwen_speaker_load(speaker, error, error_size);
    if (!encoder) return 0;

    float *reference = NULL;
    int reference_samples = 0;
    if (!h3ddle_read_mono_f32(reference_path, QWEN_SPEAKER_SAMPLE_RATE,
                              QWEN_REFERENCE_LIMIT, &reference,
                              &reference_samples, error, error_size)) {
        qwen_speaker_free(encoder);
        return 0;
    }
    float embedding[QWEN_SPEAKER_DIM];
    const int ok = qwen_speaker_embed(encoder, reference, reference_samples,
                                      embedding, error, error_size);
    free(reference);
    qwen_speaker_free(encoder);
    if (!ok) return 0;

    FILE *file = fopen(embedding_path, "wb");
    if (!file || fwrite(embedding, sizeof(float), QWEN_SPEAKER_DIM, file) !=
        QWEN_SPEAKER_DIM) {
        if (file) fclose(file);
        if (error && error_size)
            snprintf(error, error_size, "cannot write %s", embedding_path);
        return 0;
    }
    fclose(file);
    return 1;
}

int h3ddle_qwen_generate(const char *package_directory, const char *text,
                         const char *language, const char *reference_path,
                         const char *embedding_path,
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

    /* A saved voice, a clip to take one from, or neither — the model's own
     * unconditioned voice, which is what "no reference" has to mean for the
     * request to be answerable at all. */
    float embedding[QWEN_SPEAKER_DIM] = {0};
    int haveEmbedding = 0;
    float *reference = NULL;
    int reference_samples = 0;
    if (embedding_path && *embedding_path) {
        if (!qwen_read_embedding(embedding_path, embedding, error, error_size)) {
            h3_tokenizer_ids_free(ids);
            return 0;
        }
        haveEmbedding = 1;
    } else if (reference_path && *reference_path) {
        if (!h3ddle_read_mono_f32(reference_path, QWEN_SPEAKER_SAMPLE_RATE,
                                  QWEN_REFERENCE_LIMIT, &reference,
                                  &reference_samples, error, error_size)) {
            h3_tokenizer_ids_free(ids);
            return 0;
        }
    } else {
        haveEmbedding = 1;   /* the zeros above */
    }

    qwen_tts_request request = {
        .text_ids = ids,
        .text_count = (int)id_count,
        .reference = reference,
        .reference_samples = reference_samples,
        .speaker_embedding = haveEmbedding ? embedding : NULL,
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

/* ---- Z-Image-Turbo ------------------------------------------------- */

typedef struct {
    h3ddle_zimage_step on_step;
    void *opaque;
} zimage_bridge_context;

static int zimage_bridge_step(const char *phase, int step, int steps,
                              void *context) {
    zimage_bridge_context *bridge = context;
    if (!bridge->on_step) return 1;
    return bridge->on_step(phase, step, steps, bridge->opaque);
}

int h3ddle_zimage_supports_frame(int width, int height) {
    return zimage_supports_frame(width, height);
}

int h3ddle_zimage_generate(const char *package_directory, const char *shaders,
                           const char *prompt, int width, int height, int steps,
                           unsigned long long seed,
                           const unsigned char *source_rgb, float strength,
                           unsigned char *rgb,
                           h3ddle_zimage_step on_step, void *opaque,
                           char *error, size_t error_size) {
    if (!rgb) {
        snprintf(error, error_size, "somewhere to put the picture is required");
        return 0;
    }
    const size_t area = (size_t)width * height;
    const size_t count = (size_t)3 * area;

    /* Interleaved 8-bit up to channel-major [-1, 1]: the exact inverse of the
     * post-processing below, so a picture handed back in arrives as the
     * generator left it. */
    float *source = NULL;
    if (source_rgb) {
        source = malloc(count * sizeof(float));
        if (!source) {
            snprintf(error, error_size, "out of memory for the source picture");
            return 0;
        }
        for (size_t index = 0; index < area; index++)
            for (int channel = 0; channel < 3; channel++)
                source[(size_t)channel * area + index] =
                    (float)source_rgb[index * 3 + channel] / 255.0f * 2.0f - 1.0f;
    }

    const zimage_request request = {
        .package = package_directory,
        .shaders = shaders,
        .prompt = prompt,
        .width = width,
        .height = height,
        .steps = steps,
        .seed = seed,
        .source = source,
        .strength = strength,
    };
    float *planes = malloc(count * sizeof(float));
    if (!planes) {
        free(source);
        snprintf(error, error_size, "out of memory for a %dx%d picture",
                 width, height);
        return 0;
    }
    zimage_bridge_context bridge = {on_step, opaque};
    const int rendered = zimage_generate(&request, planes, zimage_bridge_step,
                                        &bridge, error, error_size);
    free(source);
    if (!rendered) {
        free(planes);
        return 0;
    }
    /* Channel-major [-1, 1] to interleaved 8-bit, which is the reference's own
     * post-processing: halve, centre, clamp. Clamping after the shift rather
     * than before is the difference between a correct picture and a blown-out
     * one. */
    for (size_t index = 0; index < area; index++)
        for (int channel = 0; channel < 3; channel++) {
            float value = planes[(size_t)channel * area + index] * 0.5f + 0.5f;
            if (value < 0.0f) value = 0.0f;
            if (value > 1.0f) value = 1.0f;
            rgb[index * 3 + channel] = (unsigned char)(value * 255.0f + 0.5f);
        }
    free(planes);
    return 1;
}

/* ----------------------------------------------------------------- LTX-2.5 */

typedef struct {
    h3ddle_ltx_step on_step;
    void *opaque;
} ltx_bridge_context;

static int ltx_bridge_step(const char *phase, int step, int steps,
                           void *context) {
    ltx_bridge_context *bridge = context;
    if (!bridge->on_step) return 1;
    return bridge->on_step(phase, step, steps, bridge->opaque);
}

/* The vocoder makes 16 kHz and the system muxer's AAC encoder will not take
 * it. Measured rather than assumed, by handing the writer a second of tone at
 * each rate: 16000, 22050 and 24000 are all refused; 32000, 44100 and 48000
 * are accepted. So the track is resampled on the way to the container.
 *
 * 48 kHz is exactly 3x, which keeps the arithmetic honest -- every third
 * output sample is an input sample untouched. This adds no bandwidth and is
 * not the 16 -> 48 kHz extender, which is a learned model and is not ported;
 * it moves a band-limited signal into a container that will carry it.
 *
 * Windowed-sinc interpolation at 32 taps a side. Linear interpolation would be
 * cheaper and would put a first-order hold's droop across the top of the band,
 * which on a track that is *already* band-limited to 8 kHz is exactly where
 * what little brightness it has lives. */
enum { LTX_CONTAINER_RATE = 48000, LTX_RESAMPLE_TAPS = 32 };

static float *resample_stereo(const float *in, uint32_t in_frames, int in_rate,
                              int out_rate, uint32_t *out_frames) {
    const double ratio = (double)out_rate / (double)in_rate;
    const uint32_t frames = (uint32_t)((double)in_frames * ratio);
    float *out = malloc((size_t)frames * 2 * sizeof(*out));
    if (!out) return NULL;
    for (uint32_t frame = 0; frame < frames; frame++) {
        const double at = (double)frame / ratio;
        const long centre = (long)floor(at);
        double left = 0.0, right = 0.0, weight = 0.0;
        for (long tap = -LTX_RESAMPLE_TAPS + 1; tap <= LTX_RESAMPLE_TAPS; tap++) {
            const long index = centre + tap;
            if (index < 0 || index >= (long)in_frames) continue;
            const double distance = at - (double)index;
            double kernel;
            if (distance == 0.0) {
                kernel = 1.0;
            } else {
                const double x = M_PI * distance;
                /* Hann over the tap span, which is what keeps the stopband
                 * from ringing across a transient. */
                const double window =
                    0.5 + 0.5 * cos(M_PI * distance / (double)LTX_RESAMPLE_TAPS);
                kernel = sin(x) / x * window;
            }
            left += kernel * (double)in[(size_t)index * 2];
            right += kernel * (double)in[(size_t)index * 2 + 1];
            weight += kernel;
        }
        /* Normalizing by the realized weight rather than trusting the kernel
         * sum keeps the first and last few frames from fading, where the
         * window runs off the end of the signal. */
        if (weight > 1e-9) { left /= weight; right /= weight; }
        out[(size_t)frame * 2] = (float)left;
        out[(size_t)frame * 2 + 1] = (float)right;
    }
    *out_frames = frames;
    return out;
}

int h3ddle_ltx_plan(int width, int height, int frames, int fps,
                    double *seconds, char *error, size_t error_size) {
    ltx_request request = {0};
    request.package = "";
    request.shaders = "";
    request.prompt = "";
    request.width = width;
    request.height = height;
    request.frames = frames;
    request.fps = fps;
    ltx_shape shape;
    if (!ltx_plan(&request, &shape, error, error_size)) return 0;
    if (seconds)
        *seconds = (double)shape.frames /
                   (fps > 0 ? (double)fps : (double)LTX_DEFAULT_FPS);
    return 1;
}

int h3ddle_ltx_generate(const char *package_directory, const char *shaders,
                        const char *prompt, int width, int height,
                        int frames, int fps,
                        int steps, unsigned long long seed,
                        const char *first_frame, const char *last_frame,
                        const char *const *references, int reference_count,
                        const char *output_path, h3ddle_ltx_step on_step,
                        void *opaque, char *error, size_t error_size) {
    if (!output_path) {
        snprintf(error, error_size, "somewhere to put the clip is required");
        return 0;
    }
    /* Anchors first and at the ends, then references spread through what is
     * left. The last frame is pinned to `frames - 1` rather than to a
     * duration, because a rounded duration and the rendered length differ. */
    ltx_conditioning conditioning[LTX_MAX_CONDITIONING];
    int conditioning_count = 0;
    if (first_frame && *first_frame) {
        conditioning[conditioning_count].path = first_frame;
        conditioning[conditioning_count].frame_index = 0;
        conditioning[conditioning_count].strength = 1.0f;
        conditioning_count++;
    }
    if (last_frame && *last_frame && conditioning_count < LTX_MAX_CONDITIONING) {
        conditioning[conditioning_count].path = last_frame;
        conditioning[conditioning_count].frame_index = frames - 1;
        conditioning[conditioning_count].strength = 1.0f;
        conditioning_count++;
    }
    for (int index = 0; index < reference_count &&
                        conditioning_count < LTX_MAX_CONDITIONING; index++) {
        if (!references || !references[index] || !*references[index]) continue;
        conditioning[conditioning_count].path = references[index];
        conditioning[conditioning_count].frame_index =
            (int)((long)(index + 1) * frames / (reference_count + 1));
        conditioning[conditioning_count].strength = 1.0f;
        conditioning_count++;
    }
    ltx_request request = {0};
    request.package = package_directory;
    request.shaders = shaders;
    request.prompt = prompt;
    request.width = width;
    request.height = height;
    request.frames = frames;
    request.fps = fps;
    request.steps = steps;
    request.seed = seed;
    request.conditioning = conditioning_count ? conditioning : NULL;
    request.conditioning_count = conditioning_count;
    ltx_shape shape;
    if (!ltx_plan(&request, &shape, error, error_size)) return 0;

    float *planes = malloc(shape.video_floats * sizeof(*planes));
    float *audio = malloc(shape.audio_floats * sizeof(*audio));
    if (!planes || !audio) {
        free(planes); free(audio);
        snprintf(error, error_size, "out of memory for %d frames of %dx%d",
                 shape.frames, shape.width, shape.height);
        return 0;
    }
    ltx_bridge_context bridge = {on_step, opaque};
    if (!ltx_generate(&request, planes, audio, ltx_bridge_step, &bridge,
                      error, error_size)) {
        free(planes); free(audio);
        return 0;
    }

    /* Channel-major [-1, 1] to tightly packed RGB24, which is what the muxer
     * takes. Same halve-centre-clamp as Z-Image's, and the same reason to
     * clamp *after* the shift rather than before. The planes are per frame
     * here, so the channel stride is one frame's area rather than the whole
     * clip's. */
    const size_t area = (size_t)shape.width * shape.height;
    unsigned char *rgb = malloc((size_t)shape.frames * area * 3);
    if (!rgb) {
        free(planes); free(audio);
        snprintf(error, error_size, "out of memory packing %d frames",
                 shape.frames);
        return 0;
    }
    for (int frame = 0; frame < shape.frames; frame++)
        for (size_t index = 0; index < area; index++)
            for (int channel = 0; channel < 3; channel++) {
                const size_t at = ((size_t)channel * (size_t)shape.frames +
                                   (size_t)frame) * area + index;
                float value = planes[at] * 0.5f + 0.5f;
                if (value < 0.0f) value = 0.0f;
                if (value > 1.0f) value = 1.0f;
                rgb[((size_t)frame * area + index) * 3 + channel] =
                    (unsigned char)(value * 255.0f + 0.5f);
            }
    free(planes);

    /* The soundtrack is the point of this engine, so it is muxed in rather
     * than written beside the clip. */
    uint32_t track_frames = shape.audio_frames;
    float *track = resample_stereo(audio, shape.audio_frames,
                                   LTX_AUDIO_SAMPLE_RATE, LTX_CONTAINER_RATE,
                                   &track_frames);
    free(audio);
    if (!track) {
        free(rgb);
        snprintf(error, error_size, "out of memory resampling the soundtrack");
        return 0;
    }
    const int wrote = h3_avwriter_write_av_rgb24_f32(
        output_path, rgb, shape.frames, shape.width, shape.height,
        fps > 0 ? fps : LTX_DEFAULT_FPS, track, (int)track_frames, 2,
        LTX_CONTAINER_RATE, error, error_size);
    free(rgb);
    free(track);
    return wrote;
}
