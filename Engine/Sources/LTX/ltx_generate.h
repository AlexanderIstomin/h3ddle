/* Prompt in, clip out: the whole LTX-2.5 pipeline behind one call.
 *
 * Tokenize, encode, connect, denoise both streams, decode both. The stages are
 * separately gated against the released weights and separately usable — see
 * the README — but the service wants one entry point, and the joins between
 * them are decisions this file owns rather than its caller: the rope's time
 * axis in seconds, the audio length derived from the video's, the context span
 * taken from the connector rather than assumed.
 */
#ifndef LTX_GENERATE_H
#define LTX_GENERATE_H

#include <stddef.h>
#include <stdint.h>

/* Eight is what the distilled checkpoint was made for. */
#define LTX_DEFAULT_STEPS 8
/* LTX's conventional rate. Not in the checkpoint — `VideoPixelShape.fps` is
 * supplied by the caller — but not free either: it sets the scale of the
 * video rope's time axis and, through that, the audio length. */
#define LTX_DEFAULT_FPS 24

/* A picture the clip is conditioned on.
 *
 * The frame index is in **pixel** frames: 0 opens the clip, `frames - 1` ends
 * it, and anything between pins a moment in the middle. Strength 1 holds the
 * picture exactly.
 *
 * These are appended to the DiT's sequence rather than pasted into the output,
 * so the rendered frame at that index resembles the picture rather than being
 * it — which is what lets the model carry motion through it. */
typedef struct {
    const char *path;       /* any image ImageIO reads */
    int frame_index;
    float strength;         /* 0 takes 1.0 */
} ltx_conditioning;

/* Conditioning pictures are square like the clip, and there are not many:
 * every one is a full VAE encode and a permanent addition to the sequence
 * every block of every step reads. */
#define LTX_MAX_CONDITIONING 4

typedef struct {
    /* The model snapshot: diffusion_models/, text_encoders/, vae/. */
    const char *package;
    /* h3_shaders.metal. This engine has no CPU path — the DiT is 21.5 GB of
     * int8 and a CPU fallback would be days, not minutes. */
    const char *shaders;
    const char *prompt;
    /* Both a multiple of 32, the video VAE's spatial factor, and otherwise
     * free. Square is allowed but is not what this model is demonstrated at:
     * the released example renders 960x544 and doubles it in a second stage
     * this engine does not run. */
    int width, height;
    /* Pixel frames. The VAE compresses 8x in time and cannot produce the seven
     * leading frames, so this must be 8k + 1: 17, 33, 65, 97. */
    int frames;
    int fps;                /* 0 takes LTX_DEFAULT_FPS */
    int steps;              /* 0 takes LTX_DEFAULT_STEPS */
    uint64_t seed;
    /* Optional; up to LTX_MAX_CONDITIONING. */
    const ltx_conditioning *conditioning;
    int conditioning_count;
} ltx_request;

/* What a request will produce, so a caller can allocate — and refuse — before
 * loading thirty-eight gigabytes. */
typedef struct {
    int frames;             /* pixel frames, echoing the request */
    int width, height;
    /* 16 kHz stereo, interleaved. Derived from the video duration rather than
     * chosen: round(frames / fps * 25) latent rows, each 4 mel frames less the
     * 3 the causal stack cannot make, each 160 samples. */
    uint32_t audio_frames;
    size_t video_floats;    /* 3 * frames * width * height */
    size_t audio_floats;    /* 2 * audio_frames */
} ltx_shape;

/* Returns 0 and sets `error` when the request is not renderable — a frame
 * count that is not 8k + 1, a side that is not a multiple of 32. */
int ltx_plan(const ltx_request *request, ltx_shape *shape,
             char *error, size_t error_size);

/* Called as the run proceeds. Returning zero abandons the generation, which is
 * how the service cancels.
 *
 * `phase` names what is happening — "text encoder", "connector", "denoise",
 * "video VAE", "vocoder" — because the sampler is only part of the wait. The
 * tower alone is minutes before the first step, and a caller that hears
 * nothing until then shows a still bar and reads as hung.
 *
 * `step` and `steps` are within the phase. Denoise restarts the block counter
 * for each named step; video VAE counts decoder operations across all tiles
 * and never goes backwards. */
typedef int (*ltx_progress)(const char *phase, int step, int steps,
                            void *context);

/* `video` receives `shape.video_floats`, channel-major, in [-1, 1] — the range
 * the VAE works in. `audio` receives `shape.audio_floats`, interleaved stereo
 * in [-1, 1], already clamped as the vocoder's own tail does.
 *
 * Returns 0 on failure with `error` set, and 0 with an empty `error` when the
 * caller cancelled. */
int ltx_generate(const ltx_request *request, float *video, float *audio,
                 ltx_progress progress, void *context,
                 char *error, size_t error_size);

#endif
