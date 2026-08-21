/* Prompt in, clip out: the LTX-2.5 pipeline behind one call.
 *
 * The stages are separately gated against the released weights and separately
 * usable. What this adds is the sequencing, and the sequencing is the part
 * with a constraint in it: **the Gemma tower and the DiT do not fit in memory
 * together** -- about 37 GB -- so the tower runs, hands back its features and
 * is freed before the DiT store is opened. That ordering is a requirement,
 * not scaffolding, and it is why this file exists rather than a caller simply
 * calling the five stages itself.
 *
 * `Vendor/h3.c/tests/ltx_clip.sh` is the specification: it chains the same
 * five stages as separate processes, which is how the two-phase split was
 * discovered in the first place.
 *
 * The joins are decisions this file owns rather than its caller's:
 *
 *   - the tokenizer left-pads to 256 and the pads are **not** stripped -- the
 *     connector's registers replace them, and the span the DiT cross-attends
 *     to is that padded length rather than the register count;
 *
 *   - the audio length follows the video's duration in seconds, through
 *     `ltx_audio_rows_for`, and is not a free parameter; and
 *
 *   - fps reaches the DiT explicitly, because the rope's time axis is measured
 *     in seconds and the two streams attend to each other through it.
 */
#include "ltx_generate.h"

#include "ltx_audio.h"
#include "ltx_connector.h"
#include "ltx_dit.h"
#include "ltx_text.h"
#include "ltx_video.h"

#include "h3_avreader.h"
#include "h3_ffmpeg.h"
#include "h3_gpu.h"
#include "h3_safetensors.h"
#include "h3_tokenizer.h"
#include "h3_weights.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* `LTXVGemmaTokenizer(max_length=256)`. The span is what the DiT
 * cross-attends to; the prompt itself is whatever it tokenized to, and the
 * connector's registers make up the difference. */
enum { SPAN = 256, TEMPORAL = 8, SPATIAL = LTX_VIDEO_SPATIAL };

/* The snapshot's own layout. The three directories are what `ltx_request.
 * package` names; the filenames are the published ones. */
#define DIT_PATH "diffusion_models/" \
    "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
#define DIT_INPUT_MAJOR_PATH "diffusion_models/" \
    "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot_input_major.safetensors"
#define ENCODER_PATH "text_encoders/" \
    "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
#define VIDEO_VAE_PATH "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
#define AUDIO_VAE_PATH "vae/ltx-2.5-audio-vae-bf16.safetensors"

static int fail(char *error, size_t error_size, const char *format, ...) {
    if (error && error_size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return 0;
}

/* A local repack can sit beside the released checkpoint for an A/B without
 * renaming or overwriting 21 GB files. Managed releases install their selected
 * source under DIT_PATH, so this is diagnostic compatibility rather than a
 * second production dependency. `H3_LTX_INPUT_MAJOR=0` forces the original;
 * `=1` requires the sibling candidate; unset selects the candidate when it is
 * present. The tensor marker, not the filename, still decides the GPU layout. */
static int transformer_path(const ltx_request *request, char *path, size_t size,
                            char *error, size_t error_size) {
    const char *setting = getenv("H3_LTX_INPUT_MAJOR");
    const int force_original = setting && !strcmp(setting, "0");
    const int require_candidate = setting && !strcmp(setting, "1");
    if (!force_original) {
        snprintf(path, size, "%s/" DIT_INPUT_MAJOR_PATH, request->package);
        if (access(path, R_OK) == 0) {
            if (getenv("H3_PROFILE"))
                fprintf(stderr, "ltx: using input-major candidate %s\n", path);
            return 1;
        }
        if (require_candidate)
            return fail(error, error_size,
                        "H3_LTX_INPUT_MAJOR=1 but the candidate is missing: %s",
                        path);
    }
    snprintf(path, size, "%s/" DIT_PATH, request->package);
    if (getenv("H3_PROFILE"))
        fprintf(stderr, "ltx: using managed transformer %s\n", path);
    return 1;
}

/* ------------------------------------------------------------------- plan */

int ltx_plan(const ltx_request *request, ltx_shape *shape,
             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!request || !shape)
        return fail(error, error_size, "ltx_plan wants a request and "
                                       "somewhere to put the shape");
    const int fps = request->fps > 0 ? request->fps : LTX_DEFAULT_FPS;
    if (request->width < SPATIAL || request->width % SPATIAL ||
        request->height < SPATIAL || request->height % SPATIAL)
        return fail(error, error_size, "%dx%d is not a frame this engine "
                                       "renders; the video VAE expands %dx in "
                                       "space, so both sides must be multiples "
                                       "of %d", request->width,
                    request->height, SPATIAL, SPATIAL);
    /* A causal encoder builds the first latent frame from a single pixel
     * frame, so n latent frames decode to 8(n-1)+1 and nothing else. */
    if (request->frames < 1 || (request->frames - 1) % TEMPORAL)
        return fail(error, error_size, "%d frames is not renderable; the video "
                                       "VAE compresses %dx in time and cannot "
                                       "produce the seven leading frames, so "
                                       "8k+1 -- 17, 33, 65, 97 -- is what it "
                                       "makes", request->frames, TEMPORAL);
    memset(shape, 0, sizeof(*shape));
    shape->frames = request->frames;
    shape->width = request->width;
    shape->height = request->height;
    shape->audio_frames =
        ltx_audio_frames_for(ltx_audio_rows_for(request->frames, fps));
    shape->video_floats = (size_t)LTX_VIDEO_CHANNELS * (size_t)request->frames *
                          (size_t)request->width * (size_t)request->height;
    shape->audio_floats = 2u * (size_t)shape->audio_frames;
    return 1;
}

/* -------------------------------------------------------------- the relays */

/* Each stage reports within itself, so the phase name is what tells a caller
 * which of the five it is waiting on. Without it the tower alone is minutes
 * of silence before the first denoise step, which reads as a hang. */
typedef struct {
    ltx_progress progress;
    void *context;
} relay;

static int text_tick(int layer, int layers, void *context) {
    relay *r = context;
    if (!r->progress) return 1;
    return r->progress("text encoder", layer, layers, r->context);
}

static int denoise_tick(int completed, int total, void *context) {
    relay *r = context;
    if (!r->progress) return 1;
    const int steps = total / LTX_DIT_BLOCKS;
    int step;
    int block;
    if (completed >= total) {
        step = steps;
        block = LTX_DIT_BLOCKS;
    } else {
        step = completed / LTX_DIT_BLOCKS + 1;
        block = completed % LTX_DIT_BLOCKS;
    }
    char phase[64];
    snprintf(phase, sizeof(phase), "denoise step %d/%d", step, steps);
    return r->progress(phase, block, LTX_DIT_BLOCKS, r->context);
}

static int video_tick(int completed, int total, void *context) {
    relay *r = context;
    if (!r->progress) return 1;
    return r->progress("video VAE", completed, total, r->context);
}

/* A phase with nothing to report inside it still says it started. Returning
 * zero here cancels, the same as anywhere else. */
static int announce(const relay *r, const char *phase) {
    if (!r->progress) return 1;
    return r->progress(phase, 0, 1, r->context);
}

/* ---------------------------------------------------------- the tokenizer */

/* Gemma's tokenizer rides in the text encoder checkpoint as a `tokenizer_json`
 * tensor rather than beside it as a file, so it is read out and parsed from
 * memory. It is also *not* the stock Gemma tokenizer: the post-processor in
 * this copy is a no-op, so no <bos> is prepended, and h3.c's SentencePiece
 * path reproduces its ids exactly -- checked against the reference
 * `tokenizers` library in `h3_gemma_tokenizer_test`.
 *
 * The ids come back **bare**, not padded to the span, which is deliberate and
 * is the subtlest decision in this file. See `ltx_generate` below. */
static int tokenize(const h3_weight_store *encoder, const char *prompt,
                    int32_t *ids, uint32_t *tokens,
                    char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor =
        h3_weight_find(encoder, "tokenizer_json", &header);
    if (!tensor)
        return fail(error, error_size, "the text encoder carries no "
                                       "tokenizer_json tensor");
    if (tensor->dtype != H3_DTYPE_U8)
        return fail(error, error_size, "tokenizer_json is %s, expected bytes",
                    h3_dtype_name(tensor->dtype));
    const size_t bytes = (size_t)(tensor->data_end - tensor->data_begin);
    char *json = malloc(bytes);
    if (!json)
        return fail(error, error_size, "cannot allocate the tokenizer");
    if (!h3_st_read_data(header, tensor, json, bytes, error, error_size)) {
        free(json);
        return 0;
    }
    h3_tokenizer *tokenizer = h3_tokenizer_load_json(json, bytes, error,
                                                     error_size);
    if (!tokenizer) { free(json); return 0; }
    uint32_t *encoded = NULL;
    size_t count = 0;
    const int ok = h3_tokenizer_encode(tokenizer, prompt, 0, &encoded, &count,
                                       error, error_size);
    h3_tokenizer_free(tokenizer);
    free(json);
    if (!ok) return 0;
    if (!count) {
        h3_tokenizer_ids_free(encoded);
        return fail(error, error_size, "the prompt tokenized to nothing");
    }
    /* Truncate as the reference does, keeping the front. */
    if (count > SPAN) count = SPAN;
    for (size_t index = 0; index < count; index++)
        ids[index] = (int32_t)encoded[index];
    h3_tokenizer_ids_free(encoded);
    *tokens = (uint32_t)count;
    return 1;
}

/* Why the tower is run on the bare prompt rather than the padded span.
 *
 * `LTXVGemmaTokenizer` left-pads to 256 and hands the tower an attention mask
 * beside the ids; `base_encoder.py` passes that mask straight into the HF
 * model, so the pads are excluded from attention. This engine has no masked
 * GQA kernel -- `h3_gpu_gqa_causal_bf16` is causal and nothing else -- so
 * feeding it the padded span would let every real token attend to two hundred
 * and thirty-odd pad tokens the reference never shows it.
 *
 * Running the bare tokens instead is *equivalent*, not merely close, and the
 * reason is that Gemma's positions enter only through RoPE. RoPE is relative:
 * a query at i and a key at j interact through i - j alone, so translating
 * every position by the pad length leaves every attention logit unchanged.
 * Gemma carries no learned absolute position embedding to break that. The
 * sliding layers' window is far wider than any prompt at this length, so the
 * same keys are in range either way.
 *
 * Two consequences fall out. The aggregation zeroes padded rows before the
 * connector sees them (`norm_and_concat_per_token_rms` is per token, so no
 * statistic crosses tokens and nothing else is affected), and those rows are
 * then discarded anyway -- so not producing them costs nothing. And the tower
 * runs on a few dozen rows instead of 256, which is most of a minute back.
 *
 * The span still exists; it is just the connector's job. See below. */

/* Read each conditioning picture and encode it into the DiT's latent space.
 *
 * The video VAE is opened for this and released again before the tower loads,
 * which keeps the 37 GB rule intact: at no point are the encoder, the tower and
 * the DiT all resident. It costs a second open of a 1.4 GB file, against
 * holding it across the whole run.
 *
 * `h3_avreader_read_image_f32` hands back [0, 1] and the VAE works in [-1, 1],
 * which is the same halve-centre convention the decoder inverts on the way
 * out. */
static int encode_conditioning(const ltx_request *request, h3_gpu *gpu,
                               uint32_t cells_wide, uint32_t cells_high,
                               float *latents, ltx_dit_condition *items,
                               char *error, size_t error_size) {
    char path[1200];
    snprintf(path, sizeof(path), "%s/" VIDEO_VAE_PATH, request->package);
    h3_weight_store *vae = h3_weight_store_open(path, error, error_size);
    if (!vae) return 0;
    const uint32_t pixel_width = cells_wide * SPATIAL;
    const uint32_t pixel_height = cells_high * SPATIAL;
    const size_t cells = (size_t)cells_wide * cells_high;
    int ok = 1;
    size_t at = 0;
    for (int index = 0; index < request->conditioning_count && ok; index++) {
        const ltx_conditioning *item = &request->conditioning[index];
        float *rgb = NULL;
        /* Cover rather than stretch: a reference of a different shape is
         * cropped to the clip's, not squashed into it. */
        ok = h3_avreader_read_image_f32(item->path, (int)pixel_width,
                                        (int)pixel_height, H3_IMAGE_FIT_COVER,
                                        &rgb, error, error_size);
        if (!ok) break;
        const size_t count =
            (size_t)LTX_VIDEO_CHANNELS * pixel_width * pixel_height;
        for (size_t value = 0; value < count; value++)
            rgb[value] = rgb[value] * 2.0f - 1.0f;
        ok = ltx_video_encode(gpu, vae, rgb, 1, pixel_height, pixel_width,
                              latents + at * LTX_DIT_LATENT, error, error_size);
        free(rgb);
        if (!ok) break;
        items[index].latent = latents + at * LTX_DIT_LATENT;
        items[index].frames = 1;
        items[index].frame_index = item->frame_index;
        items[index].strength = item->strength > 0.0f ? item->strength : 1.0f;
        at += cells;
    }
    h3_weight_store_free(vae);
    return ok;
}

/* ------------------------------------------------------------------- entry */

int ltx_generate(const ltx_request *request, float *video, float *audio,
                 ltx_progress progress, void *context,
                 char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!request || !request->package || !request->prompt || !video || !audio)
        return fail(error, error_size, "a package, a prompt and somewhere to "
                                       "put both streams are all required");
    ltx_shape shape;
    if (!ltx_plan(request, &shape, error, error_size)) return 0;

    const int fps = request->fps > 0 ? request->fps : LTX_DEFAULT_FPS;
    const int steps = request->steps > 0 ? request->steps : LTX_DEFAULT_STEPS;
    const uint32_t latent_frames =
        (uint32_t)((request->frames - 1) / TEMPORAL) + 1;
    const uint32_t cells_wide = (uint32_t)(request->width / SPATIAL);
    const uint32_t cells_high = (uint32_t)(request->height / SPATIAL);
    const uint32_t audio_rows = ltx_audio_rows_for(request->frames, fps);
    relay report = { .progress = progress, .context = context };
    char path[1200];

    h3_gpu *gpu = h3_gpu_create(request->shaders ? request->shaders
                                                 : "h3_shaders.metal",
                                error, error_size);
    if (!gpu) return 0;

    int32_t *ids = malloc(SPAN * sizeof(*ids));
    float *video_features = malloc((size_t)SPAN * LTX_TEXT_VIDEO_DIM *
                                   sizeof(float));
    float *audio_features = malloc((size_t)SPAN * LTX_TEXT_AUDIO_DIM *
                                   sizeof(float));
    float *video_context = malloc((size_t)SPAN * LTX_DIT_VIDEO_DIM *
                                  sizeof(float));
    float *audio_context = malloc((size_t)SPAN * LTX_DIT_AUDIO_DIM *
                                  sizeof(float));
    const size_t video_cells = (size_t)latent_frames * cells_wide * cells_high;
    float *video_latent = malloc(video_cells * LTX_DIT_LATENT * sizeof(float));
    float *audio_latent = malloc((size_t)audio_rows * LTX_DIT_LATENT *
                                 sizeof(float));
    int ok = ids && video_features && audio_features && video_context &&
             audio_context && video_latent && audio_latent;
    if (!ok) fail(error, error_size, "cannot allocate the pipeline buffers");

    /* ---- tokenize and encode ----------------------------------------- */

    /* The tower's store is opened once and used for both: the tokenizer rides
     * inside it. */
    /* Before the tower, so the encoder's 1.4 GB is gone by the time 15 GB of
     * Gemma arrives. */
    float *conditioning_latents = NULL;
    ltx_dit_condition conditions[LTX_MAX_CONDITIONING];
    memset(conditions, 0, sizeof(conditions));
    const int conditioning_count =
        request->conditioning_count < LTX_MAX_CONDITIONING
            ? request->conditioning_count : LTX_MAX_CONDITIONING;
    if (ok && conditioning_count > 0) {
        conditioning_latents =
            malloc((size_t)conditioning_count * cells_wide * cells_high *
                   LTX_DIT_LATENT * sizeof(*conditioning_latents));
        if (!conditioning_latents)
            ok = fail(error, error_size, "cannot allocate the conditioning "
                                         "latents");
        if (ok) ok = announce(&report, "reference frames");
        if (ok)
            ok = encode_conditioning(request, gpu, cells_wide, cells_high,
                                     conditioning_latents, conditions, error,
                                     error_size);
    }

    uint32_t tokens = 0;
    if (ok) {
        snprintf(path, sizeof(path), "%s/" ENCODER_PATH, request->package);
        h3_weight_store *encoder = h3_weight_store_open(path, error, error_size);
        ok = encoder != NULL;
        if (ok) ok = tokenize(encoder, request->prompt, ids, &tokens, error,
                              error_size);
        if (ok) ok = announce(&report, "text encoder");
        if (ok)
            ok = ltx_text_encode(gpu, encoder, ids, tokens, video_features,
                                 audio_features, text_tick, &report, error,
                                 error_size);
        /* Freed before the DiT opens, which is the whole shape of this file. */
        h3_weight_store_free(encoder);
    }

    /* ---- connect and denoise ----------------------------------------- */

    if (ok) {
        ok = transformer_path(request, path, sizeof(path), error, error_size);
    }
    if (ok) {
        h3_weight_store *dit = h3_weight_store_open(path, error, error_size);
        ok = dit != NULL;
        if (ok) ok = announce(&report, "connector");
        if (ok)
            /* The prompt's own tokens, then the connector's learnable
             * registers out to the span the DiT cross-attends to. This is
             * `_replace_padded_with_learnable_registers` read literally: it
             * compacts the unmasked rows to the front and then **flips the
             * mask** before choosing between them and the registers, so the
             * registers land on the tail rather than back on the left pads.
             *
             *     adjusted = pad(hidden[mask], (0, 0, 0, pad_length))
             *     flipped  = flip(mask, dims=[1])
             *     hidden   = flipped * adjusted + (1 - flipped) * registers
             *
             * Handing it a left-padded span instead -- which every path here
             * did until now -- puts the prompt at rope positions pad..255
             * instead of 0..N-1 and feeds the tower's output for pad tokens
             * where the registers belong. It conditions plausibly either way,
             * which is exactly why it survived being looked at. */
            ok = ltx_connector_run(gpu, dit, video_features, audio_features,
                                   tokens, SPAN, video_context, audio_context,
                                   error, error_size);
        if (ok) {
            ltx_dit_request denoise = {0};
            denoise.frames = latent_frames;
            denoise.height = cells_high;
            denoise.width = cells_wide;
            denoise.audio_rows = audio_rows;
            denoise.fps = fps;
            denoise.steps = steps;
            denoise.seed = request->seed;
            denoise.conditions = conditioning_count ? conditions : NULL;
            denoise.condition_count = (uint32_t)conditioning_count;
            ok = ltx_dit_sample(gpu, dit, &denoise, video_context,
                                audio_context, SPAN, video_latent,
                                audio_latent, denoise_tick, &report, error,
                                error_size);
        }
        h3_weight_store_free(dit);
    }
    free(ids); free(video_features); free(audio_features);
    free(video_context); free(audio_context);

    /* ---- decode both streams ----------------------------------------- */

    if (ok) {
        snprintf(path, sizeof(path), "%s/" VIDEO_VAE_PATH, request->package);
        h3_weight_store *vae = h3_weight_store_open(path, error, error_size);
        ok = vae != NULL;
        if (ok)
            ok = ltx_video_decode_progress(
                gpu, vae, video_latent, latent_frames, cells_high, cells_wide,
                video, video_tick, &report, error, error_size);
        h3_weight_store_free(vae);
    }
    if (ok) {
        snprintf(path, sizeof(path), "%s/" AUDIO_VAE_PATH, request->package);
        h3_weight_store *vae = h3_weight_store_open(path, error, error_size);
        ok = vae != NULL;
        if (ok) ok = announce(&report, "vocoder");
        if (ok)
            ok = ltx_audio_decode(gpu, vae, audio_latent, audio_rows, audio,
                                  error, error_size);
        h3_weight_store_free(vae);
    }

    free(video_latent); free(audio_latent);
    free(conditioning_latents);
    h3_gpu_free(gpu);
    return ok;
}
