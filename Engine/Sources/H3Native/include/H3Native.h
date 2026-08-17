#ifndef H3DDLE_H3_NATIVE_H
#define H3DDLE_H3_NATIVE_H

#include "../../../Vendor/h3.c/h3.h"

#ifdef __cplusplus
extern "C" {
#endif

h3_params h3ddle_h3_default_params(void);
int h3ddle_h3_frames_for_seconds(double seconds);
int h3ddle_h3_prepare_metal(const char *shader_source_path, char *error,
                            size_t error_size);
const char *h3ddle_h3_version(void);
const char *h3ddle_h3_device_name(const h3_device_info *device);
const char *h3ddle_h3_device_architecture(const h3_device_info *device);
const char *h3ddle_h3_model_layout_name(const h3_model_info *model);
int h3ddle_h3_model_supports_generation(const h3_model_info *model);

/* Stable Audio 3 sound effects. The package is loaded and released per
 * call: at 1.7 GB it is cheap to reload, and holding it resident would
 * compete with H3 for the memory that actually constrains this app.
 *
 * `on_step` may be NULL; it fires once per denoising pass. Writes a
 * 16-bit stereo WAV at 44.1 kHz to `output_path`. */
typedef void (*h3ddle_sa3_step)(int completed, int total, void *opaque);
/* Releases any cached sound-effect package. Called when the video model is
 * about to load, so the two never occupy memory at once. */
void h3ddle_sa3_release(void);
int h3ddle_sa3_generate(const char *package_directory, const char *prompt,
                        double seconds, int steps, unsigned long long seed,
                        const char *output_path, h3ddle_sa3_step on_step,
                        void *opaque, char *error, size_t error_size);

/* Z-Image-Turbo: a prompt in, a square picture out.
 *
 * The package is loaded and released per call. At 14 GB that is not cheap,
 * but holding it resident would compete with H3 for exactly the memory that
 * constrains this app, and the decoder alone wants 12 GB of working buffers
 * at 1536 pixels.
 *
 * `rgb` receives pixels * pixels * 3 bytes, interleaved and 8-bit, which is
 * what the service's PNG encoder already takes. `shaders` may be NULL to stay
 * on the CPU, which is correct and roughly twenty times slower.
 *
 * `on_step` fires once per sampler step and returns zero to abandon the
 * generation; that is how cancellation reaches the sampler. A cancelled call
 * returns zero with an empty `error`. */
typedef int (*h3ddle_zimage_step)(int completed, int total, void *opaque);
int h3ddle_zimage_generate(const char *package_directory, const char *shaders,
                           const char *prompt, int pixels, int steps,
                           unsigned long long seed, unsigned char *rgb,
                           h3ddle_zimage_step on_step, void *opaque,
                           char *error, size_t error_size);
/* Whether this build renders `pixels` square; the token count must be a
 * multiple of 32, so 1440 is out where 1024 and 1536 are in. */
int h3ddle_zimage_supports_canvas(int pixels);

/* Decodes any audio file AVFoundation reads into mono float at
 * `sample_rate`, which for a Qwen3-TTS reference clip must be 24 kHz — its
 * mel filterbank is defined at that rate. The caller owns *pcm and frees it. */
int h3ddle_read_mono_f32(const char *path, int sample_rate, int max_samples,
                         float **pcm, int *samples, char *error,
                         size_t error_size);

/* Qwen3-TTS: text and a reference clip in, 24 kHz mono speech out.
 *
 * `language` is a two-letter code — en, zh, de, es, fr, it, pt, ru, ja, ko —
 * and an unknown one is an error rather than a silent fallback, because the
 * wrong language token produces fluent speech in the wrong accent.
 *
 * The voice comes from exactly one of three places, tried in that order:
 * `embedding_path`, a file of QWEN_SPEAKER_DIM floats saved earlier;
 * `reference_path`, any audio file the system can decode, from which one is
 * computed; or neither, which speaks in the model's own unconditioned voice.
 * A few seconds of clean speech is enough for a reference — the encoder pools
 * over the whole clip, so a longer one adds nothing.
 *
 * `temperature` at or below zero takes the argmax, which makes a generation
 * reproducible and is a poor default: greedy decoding loops, and a six-word
 * line measured here ran to a thirty-second ceiling where sampling at 0.7 or
 * 0.9 stopped at 2.3 seconds. `on_frame` may be NULL and fires per 80 ms
 * frame, with `total` the ceiling rather than a prediction. */
typedef void (*h3ddle_qwen_frame)(int frames, int total, void *opaque);
/* Releases any cached speech package, on the same terms as the sound-effect
 * one: the video model decides what fits. */
void h3ddle_qwen_release(void);
int h3ddle_qwen_generate(const char *package_directory, const char *text,
                         const char *language, const char *reference_path,
                         const char *embedding_path,
                         double max_seconds, double temperature, int top_k,
                         double repetition_penalty, unsigned long long seed,
                         const char *output_path, h3ddle_qwen_frame on_frame,
                         void *opaque, double *produced_seconds,
                         char *error, size_t error_size);

/* Turns a reference clip into the voice it carries and writes it to
 * `embedding_path` — 1024 little-endian floats.
 *
 * Saving this rather than the clip is what lets a voice be chosen once and
 * reused: the model conditions on these numbers and on nothing else from the
 * recording, so they are the whole of it, at four kilobytes. */
int h3ddle_qwen_write_embedding(const char *package_directory,
                                const char *reference_path,
                                const char *embedding_path,
                                char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
