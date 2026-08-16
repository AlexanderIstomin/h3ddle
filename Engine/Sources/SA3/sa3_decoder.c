#include "sa3_decoder.h"

#include "h3_safetensors.h"

#include <Accelerate/Accelerate.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CPU reference for SAME-S, checked stage by stage against tensors dumped
 * from Stability's MLX implementation. It runs on 34-token windows, so the
 * matrices stay small and Accelerate carries the cost; the Metal path can
 * follow once this is known correct, with this as the oracle. */

#define DIM 768
#define HEADS 12
#define HEAD_DIM 64
#define ROPE_DIMS 32
#define BLOCKS 6
#define FF_INNER 2304
#define SUB_CHUNK 17  /* one real token plus sixteen placeholders */
#define WINDOW 34     /* two latents' worth, the attention window */
#define SHIFT 17      /* half a window, the offset for blocks 3-5 */
#define ROPE_BASE 10000.0f

typedef struct {
    float alpha;
    float *gamma;
    float *beta;
} sa3_dyt;

typedef struct {
    sa3_dyt pre_norm;
    sa3_dyt ff_norm;
    sa3_dyt q_norm;
    sa3_dyt k_norm;
    float *to_qkv;   /* [5 * DIM, DIM] */
    float *to_out;   /* [DIM, DIM] */
    float *glu_weight; /* [2 * FF_INNER, DIM] */
    float *glu_bias;   /* [2 * FF_INNER] */
    float *out_weight; /* [DIM, FF_INNER] */
    float *out_bias;   /* [DIM] */
} sa3_block;

struct sa3_decoder {
    float running_std;
    float *project_in_weight; /* [DIM, SA3_LATENT_CHANNELS] */
    float *project_in_bias;   /* [DIM] */
    float *new_tokens;        /* [DIM] */
    float *mapping_weight;    /* [SA3_PATCH_CHANNELS, DIM, 3] */
    float *mapping_bias;      /* [SA3_PATCH_CHANNELS] */
    sa3_block blocks[BLOCKS];

    /* Rotary tables for a single window, built once. */
    float rope_cos[WINDOW * ROPE_DIMS / 2];
    float rope_sin[WINDOW * ROPE_DIMS / 2];

    /* Per-window scratch, sized at load so the forward never allocates. */
    float *window;   /* [WINDOW, DIM] */
    float *normed;   /* [WINDOW, DIM] */
    float *qkv;      /* [WINDOW, 5 * DIM] */
    float *attn_out; /* [WINDOW, DIM] */
    float *scores;   /* [WINDOW, WINDOW] */
    float *glu;      /* [WINDOW, 2 * FF_INNER] */
    float *gated;    /* [WINDOW, FF_INNER] */
    float *proj;     /* [WINDOW, DIM] */
    float *heads;    /* [5][HEADS, WINDOW, HEAD_DIM] */
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* ---- weight loading ---------------------------------------------------- */

static float *load_tensor(const h3_st_header *header, const char *name,
                          uint64_t expected, char *error, size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor) {
        fail(error, error_size, "%s is missing from the decoder weights", name);
        return NULL;
    }
    if (tensor->dtype != H3_DTYPE_F32) {
        fail(error, error_size, "%s is %s, expected F32", name,
             h3_dtype_name(tensor->dtype));
        return NULL;
    }
    uint64_t count = h3_st_tensor_elements(tensor);
    if (expected && count != expected) {
        fail(error, error_size, "%s holds %llu values, expected %llu", name,
             (unsigned long long)count, (unsigned long long)expected);
        return NULL;
    }
    float *values = malloc((size_t)count * sizeof(*values));
    if (!values) {
        fail(error, error_size, "out of memory reading %s", name);
        return NULL;
    }
    if (!h3_st_read_data(header, tensor, values, (size_t)count * sizeof(*values),
                         error, error_size)) {
        free(values);
        return NULL;
    }
    return values;
}

static int load_dyt(const h3_st_header *header, const char *prefix,
                    const char *field, int dim, sa3_dyt *norm,
                    char *error, size_t error_size) {
    char name[256];
    snprintf(name, sizeof(name), "%s%s.alpha", prefix, field);
    float *alpha = load_tensor(header, name, 1, error, error_size);
    if (!alpha) return 0;
    norm->alpha = alpha[0];
    free(alpha);

    snprintf(name, sizeof(name), "%s%s.gamma", prefix, field);
    norm->gamma = load_tensor(header, name, (uint64_t)dim, error, error_size);
    if (!norm->gamma) return 0;

    snprintf(name, sizeof(name), "%s%s.beta", prefix, field);
    norm->beta = load_tensor(header, name, (uint64_t)dim, error, error_size);
    return norm->beta != NULL;
}

static void build_rope(sa3_decoder *decoder) {
    int pairs = ROPE_DIMS / 2;
    for (int position = 0; position < WINDOW; position++) {
        for (int index = 0; index < pairs; index++) {
            float inverse = powf(ROPE_BASE,
                                 -2.0f * (float)index / (float)ROPE_DIMS);
            float angle = (float)position * inverse;
            decoder->rope_cos[position * pairs + index] = cosf(angle);
            decoder->rope_sin[position * pairs + index] = sinf(angle);
        }
    }
}

sa3_decoder *sa3_decoder_load(const char *path, char *error,
                              size_t error_size) {
    if (error && error_size) error[0] = '\0';
    h3_st_header header;
    if (!h3_st_read_header(path, &header, error, error_size)) return NULL;

    sa3_decoder *decoder = calloc(1, sizeof(*decoder));
    if (!decoder) {
        h3_st_free_header(&header);
        fail(error, error_size, "out of memory allocating the decoder");
        return NULL;
    }

    float *std = load_tensor(&header, "running_std", 1, error, error_size);
    if (!std) goto fail_out;
    decoder->running_std = std[0];
    free(std);

    decoder->project_in_weight = load_tensor(&header, "project_in.weight",
                                             DIM * SA3_LATENT_CHANNELS,
                                             error, error_size);
    decoder->project_in_bias = load_tensor(&header, "project_in.bias", DIM,
                                           error, error_size);
    decoder->new_tokens = load_tensor(&header, "new_tokens", DIM,
                                      error, error_size);
    decoder->mapping_weight = load_tensor(&header, "mapping.weight",
                                          SA3_PATCH_CHANNELS * DIM * 3,
                                          error, error_size);
    decoder->mapping_bias = load_tensor(&header, "mapping.bias",
                                        SA3_PATCH_CHANNELS, error, error_size);
    if (!decoder->project_in_weight || !decoder->project_in_bias ||
        !decoder->new_tokens || !decoder->mapping_weight ||
        !decoder->mapping_bias)
        goto fail_out;

    for (int index = 0; index < BLOCKS; index++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "blocks.%d.", index);
        sa3_block *block = &decoder->blocks[index];
        char name[256];

        if (!load_dyt(&header, prefix, "pre_norm", DIM, &block->pre_norm,
                      error, error_size) ||
            !load_dyt(&header, prefix, "ff_norm", DIM, &block->ff_norm,
                      error, error_size) ||
            !load_dyt(&header, prefix, "attn.q_norm", HEAD_DIM, &block->q_norm,
                      error, error_size) ||
            !load_dyt(&header, prefix, "attn.k_norm", HEAD_DIM, &block->k_norm,
                      error, error_size))
            goto fail_out;

        snprintf(name, sizeof(name), "%sattn.to_qkv.weight", prefix);
        block->to_qkv = load_tensor(&header, name, (uint64_t)5 * DIM * DIM,
                                    error, error_size);
        snprintf(name, sizeof(name), "%sattn.to_out.weight", prefix);
        block->to_out = load_tensor(&header, name, (uint64_t)DIM * DIM,
                                    error, error_size);
        snprintf(name, sizeof(name), "%sff.glu_proj.weight", prefix);
        block->glu_weight = load_tensor(&header, name,
                                        (uint64_t)2 * FF_INNER * DIM,
                                        error, error_size);
        snprintf(name, sizeof(name), "%sff.glu_proj.bias", prefix);
        block->glu_bias = load_tensor(&header, name, (uint64_t)2 * FF_INNER,
                                      error, error_size);
        snprintf(name, sizeof(name), "%sff.proj_out.weight", prefix);
        block->out_weight = load_tensor(&header, name,
                                        (uint64_t)DIM * FF_INNER,
                                        error, error_size);
        snprintf(name, sizeof(name), "%sff.proj_out.bias", prefix);
        block->out_bias = load_tensor(&header, name, DIM, error, error_size);

        if (!block->to_qkv || !block->to_out || !block->glu_weight ||
            !block->glu_bias || !block->out_weight || !block->out_bias)
            goto fail_out;
    }

    decoder->window = malloc((size_t)WINDOW * DIM * sizeof(float));
    decoder->normed = malloc((size_t)WINDOW * DIM * sizeof(float));
    decoder->qkv = malloc((size_t)WINDOW * 5 * DIM * sizeof(float));
    decoder->attn_out = malloc((size_t)WINDOW * DIM * sizeof(float));
    decoder->scores = malloc((size_t)WINDOW * WINDOW * sizeof(float));
    decoder->glu = malloc((size_t)WINDOW * 2 * FF_INNER * sizeof(float));
    decoder->gated = malloc((size_t)WINDOW * FF_INNER * sizeof(float));
    decoder->proj = malloc((size_t)WINDOW * DIM * sizeof(float));
    decoder->heads = malloc((size_t)5 * HEADS * WINDOW * HEAD_DIM * sizeof(float));
    if (!decoder->window || !decoder->normed || !decoder->qkv ||
        !decoder->attn_out || !decoder->scores || !decoder->glu ||
        !decoder->gated || !decoder->proj || !decoder->heads) {
        fail(error, error_size, "out of memory allocating decoder scratch");
        goto fail_out;
    }

    build_rope(decoder);
    h3_st_free_header(&header);
    return decoder;

fail_out:
    h3_st_free_header(&header);
    sa3_decoder_free(decoder);
    return NULL;
}

static void free_dyt(sa3_dyt *norm) {
    free(norm->gamma);
    free(norm->beta);
}

void sa3_decoder_free(sa3_decoder *decoder) {
    if (!decoder) return;
    free(decoder->project_in_weight);
    free(decoder->project_in_bias);
    free(decoder->new_tokens);
    free(decoder->mapping_weight);
    free(decoder->mapping_bias);
    for (int index = 0; index < BLOCKS; index++) {
        sa3_block *block = &decoder->blocks[index];
        free_dyt(&block->pre_norm);
        free_dyt(&block->ff_norm);
        free_dyt(&block->q_norm);
        free_dyt(&block->k_norm);
        free(block->to_qkv);
        free(block->to_out);
        free(block->glu_weight);
        free(block->glu_bias);
        free(block->out_weight);
        free(block->out_bias);
    }
    free(decoder->window);
    free(decoder->normed);
    free(decoder->qkv);
    free(decoder->attn_out);
    free(decoder->scores);
    free(decoder->glu);
    free(decoder->gated);
    free(decoder->proj);
    free(decoder->heads);
    free(decoder);
}

/* ---- primitives -------------------------------------------------------- */

/* rows x in @ weight[out, in]^T + bias, row-major throughout. */
static void linear(const float *input, const float *weight, const float *bias,
                   int rows, int in_features, int out_features, float *output) {
    if (bias) {
        for (int row = 0; row < rows; row++)
            memcpy(output + (size_t)row * out_features, bias,
                   (size_t)out_features * sizeof(float));
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, out_features,
                in_features, 1.0f, input, in_features, weight, in_features,
                bias ? 1.0f : 0.0f, output, out_features);
}

static void apply_dyt(float *values, int rows, int dim, const sa3_dyt *norm) {
    for (int row = 0; row < rows; row++) {
        float *line = values + (size_t)row * dim;
        for (int index = 0; index < dim; index++)
            line[index] = norm->gamma[index] * tanhf(norm->alpha * line[index]) +
                          norm->beta[index];
    }
}

/* Rotates the leading ROPE_DIMS of each head in halves, leaving the rest as
 * they are — the split-half convention, not the interleaved one. */
static void apply_rope(float *heads, int tokens, const float *cos_table,
                       const float *sin_table) {
    int pairs = ROPE_DIMS / 2;
    for (int head = 0; head < HEADS; head++) {
        for (int token = 0; token < tokens; token++) {
            float *vector = heads + ((size_t)head * tokens + token) * HEAD_DIM;
            const float *cosines = cos_table + (size_t)token * pairs;
            const float *sines = sin_table + (size_t)token * pairs;
            for (int index = 0; index < pairs; index++) {
                float low = vector[index];
                float high = vector[index + pairs];
                vector[index] = low * cosines[index] - high * sines[index];
                vector[index + pairs] = low * sines[index] + high * cosines[index];
            }
        }
    }
}

/* [tokens, DIM] with heads side by side -> [HEADS, tokens, HEAD_DIM]. */
static void split_heads(const float *packed, int tokens, float *heads) {
    for (int head = 0; head < HEADS; head++)
        for (int token = 0; token < tokens; token++)
            memcpy(heads + ((size_t)head * tokens + token) * HEAD_DIM,
                   packed + (size_t)token * DIM + head * HEAD_DIM,
                   HEAD_DIM * sizeof(float));
}

static void softmax_row(float *row, int count) {
    float largest = row[0];
    for (int index = 1; index < count; index++)
        if (row[index] > largest) largest = row[index];
    float total = 0.0f;
    for (int index = 0; index < count; index++) {
        row[index] = expf(row[index] - largest);
        total += row[index];
    }
    float inverse = 1.0f / total;
    for (int index = 0; index < count; index++) row[index] *= inverse;
}

/* Accumulates softmax(q k^T / sqrt(d)) v into output with the given sign, so
 * the differential pair costs one pass each rather than a second buffer. */
static void attend(const float *q, const float *k, const float *v, int tokens,
                   float sign, float *scores, float *output) {
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (int head = 0; head < HEADS; head++) {
        const float *qh = q + (size_t)head * tokens * HEAD_DIM;
        const float *kh = k + (size_t)head * tokens * HEAD_DIM;
        const float *vh = v + (size_t)head * tokens * HEAD_DIM;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, tokens, tokens,
                    HEAD_DIM, scale, qh, HEAD_DIM, kh, HEAD_DIM, 0.0f, scores,
                    tokens);
        for (int token = 0; token < tokens; token++)
            softmax_row(scores + (size_t)token * tokens, tokens);
        /* Scatter straight into the head's slice of the packed output. */
        for (int token = 0; token < tokens; token++) {
            const float *weights = scores + (size_t)token * tokens;
            float *destination = output + (size_t)token * DIM + head * HEAD_DIM;
            for (int channel = 0; channel < HEAD_DIM; channel++) {
                float sum = 0.0f;
                for (int source = 0; source < tokens; source++)
                    sum += weights[source] * vh[(size_t)source * HEAD_DIM + channel];
                destination[channel] += sign * sum;
            }
        }
    }
}

static void run_block(sa3_decoder *decoder, const sa3_block *block,
                      float *tokens_buffer, int tokens) {
    /* --- differential attention --- */
    memcpy(decoder->normed, tokens_buffer,
           (size_t)tokens * DIM * sizeof(float));
    apply_dyt(decoder->normed, tokens, DIM, &block->pre_norm);
    linear(decoder->normed, block->to_qkv, NULL, tokens, DIM, 5 * DIM,
           decoder->qkv);

    /* to_qkv packs q, k, v, q_diff, k_diff in that order. */
    size_t stride = (size_t)HEADS * tokens * HEAD_DIM;
    float *heads[5];
    for (int part = 0; part < 5; part++) {
        heads[part] = decoder->heads + (size_t)part * stride;
        /* Gather this projection's DIM-wide slice out of the packed rows. */
        for (int token = 0; token < tokens; token++)
            memcpy(decoder->attn_out + (size_t)token * DIM,
                   decoder->qkv + (size_t)token * 5 * DIM + part * DIM,
                   DIM * sizeof(float));
        split_heads(decoder->attn_out, tokens, heads[part]);
    }

    /* q and q_diff share q_norm; k and k_diff share k_norm. */
    for (int part = 0; part < 5; part++) {
        if (part == 2) continue; /* v is neither normalised nor rotated */
        const sa3_dyt *norm = (part == 0 || part == 3) ? &block->q_norm
                                                       : &block->k_norm;
        apply_dyt(heads[part], HEADS * tokens, HEAD_DIM, norm);
        apply_rope(heads[part], tokens, decoder->rope_cos, decoder->rope_sin);
    }

    memset(decoder->attn_out, 0, (size_t)tokens * DIM * sizeof(float));
    attend(heads[0], heads[1], heads[2], tokens, 1.0f, decoder->scores,
           decoder->attn_out);
    attend(heads[3], heads[4], heads[2], tokens, -1.0f, decoder->scores,
           decoder->attn_out);

    linear(decoder->attn_out, block->to_out, NULL, tokens, DIM, DIM,
           decoder->proj);
    for (int index = 0; index < tokens * DIM; index++)
        tokens_buffer[index] += decoder->proj[index];

    /* --- gated feed-forward --- */
    memcpy(decoder->normed, tokens_buffer,
           (size_t)tokens * DIM * sizeof(float));
    apply_dyt(decoder->normed, tokens, DIM, &block->ff_norm);
    linear(decoder->normed, block->glu_weight, block->glu_bias, tokens, DIM,
           2 * FF_INNER, decoder->glu);
    for (int token = 0; token < tokens; token++) {
        const float *line = decoder->glu + (size_t)token * 2 * FF_INNER;
        float *out = decoder->gated + (size_t)token * FF_INNER;
        for (int index = 0; index < FF_INNER; index++) {
            float gate = line[FF_INNER + index];
            out[index] = line[index] * (gate / (1.0f + expf(-gate)));
        }
    }
    linear(decoder->gated, block->out_weight, block->out_bias, tokens,
           FF_INNER, DIM, decoder->proj);
    for (int index = 0; index < tokens * DIM; index++)
        tokens_buffer[index] += decoder->proj[index];
}

/* Runs one stage of three blocks over a sequence cut into whole windows. */
static void run_stage(sa3_decoder *decoder, float *sequence, int windows,
                      int first_block) {
    for (int window = 0; window < windows; window++) {
        float *slice = sequence + (size_t)window * WINDOW * DIM;
        for (int index = 0; index < 3; index++)
            run_block(decoder, &decoder->blocks[first_block + index], slice,
                      WINDOW);
    }
}

int sa3_decoder_run(sa3_decoder *decoder, const float *latents, int frames,
                    float *patches, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!decoder || !latents || !patches || frames < 2) {
        fail(error, error_size, "invalid arguments for the SAME-S decoder");
        return 0;
    }
    if (frames % 2 != 0) {
        fail(error, error_size,
             "SAME-S needs an even latent count, got %d", frames);
        return 0;
    }

    int internal = frames * SUB_CHUNK;
    /* The shifted stage adds one window of context on each side. */
    float *sequence = malloc((size_t)(internal + WINDOW) * DIM * sizeof(float));
    float *projected = malloc((size_t)frames * DIM * sizeof(float));
    if (!sequence || !projected) {
        free(sequence);
        free(projected);
        fail(error, error_size, "out of memory decoding %d latents", frames);
        return 0;
    }

    /* Latents arrive channel-major; the projection wants one row per frame. */
    float *rows = malloc((size_t)frames * SA3_LATENT_CHANNELS * sizeof(float));
    if (!rows) {
        free(sequence);
        free(projected);
        fail(error, error_size, "out of memory transposing latents");
        return 0;
    }
    for (int frame = 0; frame < frames; frame++)
        for (int channel = 0; channel < SA3_LATENT_CHANNELS; channel++)
            rows[(size_t)frame * SA3_LATENT_CHANNELS + channel] =
                latents[(size_t)channel * frames + frame] * decoder->running_std;

    linear(rows, decoder->project_in_weight, decoder->project_in_bias, frames,
           SA3_LATENT_CHANNELS, DIM, projected);
    free(rows);

    /* Each latent contributes its own token followed by sixteen copies of the
     * learned placeholder, which is what the sixteen output patches grow from. */
    for (int frame = 0; frame < frames; frame++) {
        float *group = sequence + (size_t)frame * SUB_CHUNK * DIM;
        memcpy(group, projected + (size_t)frame * DIM, DIM * sizeof(float));
        for (int slot = 1; slot < SUB_CHUNK; slot++)
            memcpy(group + (size_t)slot * DIM, decoder->new_tokens,
                   DIM * sizeof(float));
    }
    free(projected);

    run_stage(decoder, sequence, internal / WINDOW, 0);

    /* Shift by half a window: mirror SHIFT tokens onto each end, run, trim. */
    memmove(sequence + (size_t)SHIFT * DIM, sequence,
            (size_t)internal * DIM * sizeof(float));
    memcpy(sequence, sequence + (size_t)SHIFT * DIM,
           (size_t)SHIFT * DIM * sizeof(float));
    memcpy(sequence + (size_t)(internal + SHIFT) * DIM,
           sequence + (size_t)internal * DIM,
           (size_t)SHIFT * DIM * sizeof(float));

    run_stage(decoder, sequence, (internal + WINDOW) / WINDOW, 3);

    memmove(sequence, sequence + (size_t)SHIFT * DIM,
            (size_t)internal * DIM * sizeof(float));

    /* Drop each group's leading latent token, keeping the sixteen patches. */
    int kept = frames * SA3_PATCHES_PER_LATENT;
    for (int frame = 0; frame < frames; frame++)
        memmove(sequence + (size_t)frame * SA3_PATCHES_PER_LATENT * DIM,
                sequence + ((size_t)frame * SUB_CHUNK + 1) * DIM,
                (size_t)SA3_PATCHES_PER_LATENT * DIM * sizeof(float));

    /* mapping is a width-3 convolution with zero padding at both ends; the
     * weights are [out, in, tap], so each tap is a strided matrix. */
    for (int position = 0; position < kept; position++) {
        float *out = patches + (size_t)position * SA3_PATCH_CHANNELS;
        memcpy(out, decoder->mapping_bias,
               SA3_PATCH_CHANNELS * sizeof(float));
    }
    for (int tap = 0; tap < 3; tap++) {
        int offset = tap - 1;
        for (int position = 0; position < kept; position++) {
            int source = position + offset;
            if (source < 0 || source >= kept) continue;
            const float *vector = sequence + (size_t)source * DIM;
            float *out = patches + (size_t)position * SA3_PATCH_CHANNELS;
            for (int channel = 0; channel < SA3_PATCH_CHANNELS; channel++) {
                const float *weights =
                    decoder->mapping_weight + ((size_t)channel * DIM) * 3 + tap;
                float sum = 0.0f;
                for (int index = 0; index < DIM; index++)
                    sum += vector[index] * weights[(size_t)index * 3];
                out[channel] += sum;
            }
        }
    }

    /* Hand back channel-major patches to match the reference's [512, N]. */
    float *transposed = malloc((size_t)kept * SA3_PATCH_CHANNELS * sizeof(float));
    if (!transposed) {
        free(sequence);
        fail(error, error_size, "out of memory transposing patches");
        return 0;
    }
    for (int position = 0; position < kept; position++)
        for (int channel = 0; channel < SA3_PATCH_CHANNELS; channel++)
            transposed[(size_t)channel * kept + position] =
                patches[(size_t)position * SA3_PATCH_CHANNELS + channel];
    memcpy(patches, transposed,
           (size_t)kept * SA3_PATCH_CHANNELS * sizeof(float));
    free(transposed);
    free(sequence);
    return 1;
}

int sa3_decoder_window_count(int frames, int chunk, int overlap) {
    int kernel = chunk + 2 * overlap;
    if (frames <= kernel) return 1;
    int windows = 1;                      /* the opening window */
    int position = chunk + overlap;
    while (position + chunk + overlap <= frames) {
        windows++;
        position += chunk;
    }
    if (frames - position > 0) windows++; /* the closing window */
    return windows;
}

int sa3_decoder_run_chunked(sa3_decoder *decoder, const float *latents,
                            int frames, int chunk, int overlap, float *patches,
                            sa3_decoder_progress progress, void *opaque,
                            char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    int kernel = chunk + 2 * overlap;
    if (kernel % 2 != 0) {
        fail(error, error_size,
             "chunk + 2 * overlap must be even, got %d", kernel);
        return 0;
    }
    if (frames <= kernel)
        return sa3_decoder_run(decoder, latents, frames, patches, error,
                               error_size);

    int total = frames * SA3_PATCHES_PER_LATENT;
    float *window = malloc((size_t)kernel * SA3_PATCH_CHANNELS *
                           SA3_PATCHES_PER_LATENT * sizeof(float));
    float *slice = malloc((size_t)kernel * SA3_LATENT_CHANNELS * sizeof(float));
    if (!window || !slice) {
        free(window);
        free(slice);
        fail(error, error_size, "out of memory decoding in windows");
        return 0;
    }

    /* Copies `kernel` frames starting at `start` into channel-major order. */
    #define TAKE(start) \
        do { \
            for (int channel = 0; channel < SA3_LATENT_CHANNELS; channel++) \
                memcpy(slice + (size_t)channel * kernel, \
                       latents + (size_t)channel * frames + (start), \
                       (size_t)kernel * sizeof(float)); \
        } while (0)

    /* Copies `count` output positions from the window into the result. */
    #define EMIT(from, to, count) \
        do { \
            for (int channel = 0; channel < SA3_PATCH_CHANNELS; channel++) \
                memcpy(patches + (size_t)channel * total + (to), \
                       window + (size_t)channel * kernel * \
                           SA3_PATCHES_PER_LATENT + (from), \
                       (size_t)(count) * sizeof(float)); \
        } while (0)

    int ok = 1;
    int windows = sa3_decoder_window_count(frames, chunk, overlap);
    int done = 0;
    /* The opening window needs no left context, so its head is already valid. */
    TAKE(0);
    ok = sa3_decoder_run(decoder, slice, kernel, window, error, error_size);
    int valid = chunk + overlap;
    if (ok) EMIT(0, 0, valid * SA3_PATCHES_PER_LATENT);
    if (ok && progress) progress(++done, windows, opaque);

    int position = valid;
    while (ok && position + chunk + overlap <= frames) {
        TAKE(position - overlap);
        ok = sa3_decoder_run(decoder, slice, kernel, window, error, error_size);
        if (ok)
            EMIT(overlap * SA3_PATCHES_PER_LATENT,
                 position * SA3_PATCHES_PER_LATENT,
                 chunk * SA3_PATCHES_PER_LATENT);
        position += chunk;
        if (ok && progress) progress(++done, windows, opaque);
    }

    /* The final window is anchored to the end, so its tail is valid. */
    int remaining = frames - position;
    if (ok && remaining > 0) {
        TAKE(frames - kernel);
        ok = sa3_decoder_run(decoder, slice, kernel, window, error, error_size);
        if (ok)
            EMIT((kernel - remaining) * SA3_PATCHES_PER_LATENT,
                 position * SA3_PATCHES_PER_LATENT,
                 remaining * SA3_PATCHES_PER_LATENT);
        if (ok && progress) progress(++done, windows, opaque);
    }

    #undef TAKE
    #undef EMIT
    free(window);
    free(slice);
    return ok;
}

void sa3_decoder_unpatch(const float *patches, int patch_count, float *audio) {
    /* [2 * 256, N] holds one channel's 256 samples per patch, so the sample at
     * (channel, patch, tap) sits at row channel * 256 + tap. */
    int samples = patch_count * 256;
    for (int channel = 0; channel < 2; channel++)
        for (int patch = 0; patch < patch_count; patch++)
            for (int tap = 0; tap < 256; tap++)
                audio[(size_t)channel * samples + (size_t)patch * 256 + tap] =
                    patches[((size_t)channel * 256 + tap) * patch_count + patch];
}
