/* LTX-2.5's dual-stream DiT: conditioning in, two latents out.
 *
 * 48 blocks, six attentions and two feed-forwards each, video at 4096 wide
 * over 32 heads of 128 and audio at 2048 over 32 of 64. Both streams denoise
 * together and attend to each other, which is what makes the soundtrack match
 * the picture rather than merely share a prompt.
 *
 * The geometry is a runtime argument here where the driver takes it at compile
 * time. That is the one part of this stage that is not a lift: the service
 * cannot ship a binary per clip length. It turns out to cost little, because
 * the driver sizes everything with computed lengths and holds exactly one
 * array on the stack.
 */
#ifndef LTX_DIT_H
#define LTX_DIT_H

#include "h3_gpu.h"
#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>

#define LTX_DIT_VIDEO_DIM 4096
#define LTX_DIT_AUDIO_DIM 2048
#define LTX_DIT_LATENT 128
#define LTX_DIT_BLOCKS 48
#define LTX_DIT_MAX_STEPS 32

/* A picture the clip is conditioned on, already in the DiT's latent space —
 * `ltx_video_encode` is what puts it there.
 *
 * Conditioning here *appends* tokens rather than freezing existing ones. The
 * keyframe's tokens join the sequence with their own positions, and the model
 * matches the denoising tokens to them; the extra tokens are dropped on the
 * way out. `VideoConditionByKeyframeIndex` is the reference for all of it.
 */
typedef struct {
    /* `[frames][height][width][128]` channels-last, at the request's spatial
     * size. One latent frame is a single picture. */
    const float *latent;
    uint32_t frames;
    /* Where this sits in the clip, in **pixel** frames — 0 is the first frame,
     * and the causal shift that gives the opening frame its own [0, 1) span
     * applies only there. */
    int frame_index;
    /* 1 holds the picture exactly; below that lets the sampler move it. The
     * per-token timestep is `sigma * (1 - strength)`, so 1 pins the token at
     * timestep zero and its velocity contributes nothing. */
    float strength;
} ltx_dit_condition;

typedef struct {
    /* Latent geometry. Pixel frames are 8(frames-1)+1 and each spatial cell is
     * 32 pixels; `ltx_plan` is what turns a request into these. */
    uint32_t frames, height, width;
    /* Audio latent rows. Derived from the video duration -- see
     * `ltx_audio_rows_for` -- and not free: the two streams cross-attend
     * through positions measured in seconds, so a mismatch desynchronizes them
     * rather than merely truncating one. */
    uint32_t audio_rows;
    /* The rate the clip is played at, which sets the scale of the video rope's
     * time axis. Wrong here and the picture still looks right while the
     * soundtrack stops matching it. */
    int fps;
    int steps;
    uint64_t seed;
    /* Optional. Each adds its tokens to the video stream for the whole run. */
    const ltx_dit_condition *conditions;
    uint32_t condition_count;
} ltx_dit_request;

/* Returning zero abandons the run. `completed` counts completed transformer
 * blocks across the whole sampler and `total` is steps * LTX_DIT_BLOCKS, so
 * the value runs 0..total and never goes backwards.
 *
 * Asked between blocks rather than between steps: a step is 48 blocks and can
 * take minutes at a large geometry. This makes the first step measurable and
 * lets cancellation land without waiting for the entire pass. */
typedef int (*ltx_dit_tick)(int completed, int total, void *context);

/* Called after each completed Euler step, while the updated video latent is
 * already present on the host. `video_latent` excludes appended conditioning
 * rows. Returning zero cancels on the same terms as `ltx_dit_tick`. */
typedef int (*ltx_dit_preview)(int completed_steps, int total_steps,
                               const float *video_latent,
                               size_t video_elements, void *context);

/* `video_context` is `[span][4096]` and `audio_context` `[span][2048]`, as
 * `ltx_connector_run` leaves them; `span` is the tokenizer's, not the register
 * count.
 *
 * `video_latent` receives `frames * height * width * 128` floats and
 * `audio_latent` `audio_rows * 128`, both in the normalized latent space the
 * VAEs denormalize out of.
 *
 * Returns 0 with `error` set on failure, and 0 with an empty `error` when the
 * tick cancelled. */
int ltx_dit_sample(h3_gpu *gpu, const h3_weight_store *dit,
                   const ltx_dit_request *request, const float *video_context,
                   const float *audio_context, uint32_t span,
                   float *video_latent, float *audio_latent,
                   ltx_dit_tick tick, void *tick_context,
                   ltx_dit_preview preview, void *preview_context,
                   char *error, size_t error_size);

#endif
