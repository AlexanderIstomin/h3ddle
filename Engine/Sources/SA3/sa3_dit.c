#include "sa3_dit.h"

#include "h3_safetensors.h"

#include <Accelerate/Accelerate.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CPU reference for the SA3 DiT, checked against tensors dumped from
 * Stability's MLX implementation. The weights ship as f16 and are widened once
 * at load so Accelerate can drive them; that costs memory but keeps this
 * honest as the oracle the Metal path will be measured against. */

#define EMBED 1024
#define DEPTH 20
#define HEADS 16
#define HEAD_DIM 64
#define ROPE_DIMS 32
#define FF_INNER 4096
#define MEMORY_TOKENS 64
#define LOCAL_DIM 257
#define TIMESTEP_DIM 256
#define NORM_EPS 1e-5f
#define QK_NORM_EPS 1e-6f
#define ROPE_BASE 10000.0f
#define FOURIER_MIN 0.5f
#define FOURIER_MAX 10000.0f

typedef struct {
    float *pre_norm;         /* [EMBED] */
    float *cross_attend_norm;/* [EMBED] */
    float *ff_norm;          /* [EMBED] */
    float *scale_shift_gate; /* [6 * EMBED] */

    float *self_qkv;         /* [3 * EMBED, EMBED] */
    float *self_out;         /* [EMBED, EMBED] */
    float *self_q_norm;      /* [HEAD_DIM] */
    float *self_k_norm;      /* [HEAD_DIM] */

    float *cross_q;          /* [EMBED, EMBED] */
    float *cross_kv;         /* [2 * EMBED, EMBED] */
    float *cross_out;        /* [EMBED, EMBED] */
    float *cross_q_norm;     /* [HEAD_DIM] */
    float *cross_k_norm;     /* [HEAD_DIM] */

    float *ff_in;            /* [2 * FF_INNER, EMBED] */
    float *ff_in_bias;       /* [2 * FF_INNER] */
    float *ff_out;           /* [EMBED, FF_INNER] */
    float *ff_out_bias;      /* [EMBED] */

    float *local_in;         /* [EMBED, LOCAL_DIM] */
    float *local_in_bias;    /* [EMBED] */
    float *local_out;        /* [EMBED, EMBED] */
    float *local_out_bias;   /* [EMBED] */
} sa3_dit_block;

struct sa3_dit {
    float *preprocess;   /* [CHANNELS, CHANNELS] */
    float *postprocess;  /* [CHANNELS, CHANNELS] */
    float *cond_in;      /* [EMBED, CONTEXT] */
    float *cond_out;     /* [EMBED, EMBED] */
    float *global_in;    /* [EMBED, CONTEXT] */
    float *global_out;   /* [EMBED, EMBED] */
    float *timestep_in;  /* [EMBED, TIMESTEP_DIM] */
    float *timestep_in_bias;
    float *timestep_out; /* [EMBED, EMBED] */
    float *timestep_out_bias;
    float *project_in;   /* [EMBED, CHANNELS] */
    float *project_out;  /* [CHANNELS, EMBED] */
    float *memory;       /* [MEMORY_TOKENS, EMBED] */
    float *modulation_in;      /* [EMBED, EMBED] */
    float *modulation_in_bias; /* [EMBED] */
    float *modulation_out;     /* [6 * EMBED, EMBED] */
    float *modulation_out_bias;/* [6 * EMBED] */
    sa3_dit_block blocks[DEPTH];

    float fourier[TIMESTEP_DIM / 2];

    /* Scratch grown to fit the longest sequence seen so far. */
    int capacity_tokens;
    int capacity_context;
    float *sequence;   /* [tokens, EMBED] */
    float *normed;     /* [tokens, EMBED] */
    float *qkv;        /* [tokens, 3 * EMBED] */
    float *heads_q;    /* [HEADS, tokens, HEAD_DIM] */
    float *heads_k;    /* [HEADS, max(tokens,context), HEAD_DIM] */
    float *heads_v;    /* [HEADS, max(tokens,context), HEAD_DIM] */
    float *attn;       /* [tokens, EMBED] */
    float *scores;     /* [tokens, max(tokens,context)] */
    float *ff;         /* [tokens, 2 * FF_INNER] */
    float *gated;      /* [tokens, FF_INNER] */
    float *proj;       /* [tokens, EMBED] */
    float *context_kv; /* [context, 2 * EMBED] */
    float *context_out;/* [context, EMBED] */
    float *local;      /* [tokens, EMBED] */
    float *rope_cos;   /* [tokens, ROPE_DIMS / 2] */
    float *rope_sin;
    float modulation[6 * EMBED];
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* ---- loading ----------------------------------------------------------- */

/* Widens f16 to f32 on the way in; the archive is f16 but Accelerate is not. */
static float *load(const h3_st_header *header, const char *name,
                   uint64_t expected, char *error, size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor) {
        fail(error, error_size, "%s is missing from the transformer", name);
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
    if (tensor->dtype == H3_DTYPE_F32) {
        if (!h3_st_read_data(header, tensor, values,
                             (size_t)count * sizeof(*values), error,
                             error_size)) {
            free(values);
            return NULL;
        }
        return values;
    }
    if (tensor->dtype != H3_DTYPE_F16) {
        fail(error, error_size, "%s is %s, expected F16 or F32", name,
             h3_dtype_name(tensor->dtype));
        free(values);
        return NULL;
    }
    uint16_t *raw = malloc((size_t)count * sizeof(*raw));
    if (!raw) {
        free(values);
        fail(error, error_size, "out of memory reading %s", name);
        return NULL;
    }
    if (!h3_st_read_data(header, tensor, raw, (size_t)count * sizeof(*raw),
                         error, error_size)) {
        free(raw);
        free(values);
        return NULL;
    }
    vImage_Buffer source = {raw, 1, (vImagePixelCount)count,
                            (size_t)count * sizeof(*raw)};
    vImage_Buffer destination = {values, 1, (vImagePixelCount)count,
                                 (size_t)count * sizeof(*values)};
    vImageConvert_Planar16FtoPlanarF(&source, &destination, 0);
    free(raw);
    return values;
}

#define LOAD(field, name, count)                                    \
    do {                                                            \
        dit->field = load(&header, (name), (count), error, error_size); \
        if (!dit->field) goto fail_out;                             \
    } while (0)

#define LOAD_BLOCK(field, suffix, count)                            \
    do {                                                            \
        snprintf(name, sizeof(name), "%s" suffix, prefix);          \
        block->field = load(&header, name, (count), error, error_size); \
        if (!block->field) goto fail_out;                           \
    } while (0)

sa3_dit *sa3_dit_load(const char *path, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    h3_st_header header;
    if (!h3_st_read_header(path, &header, error, error_size)) return NULL;

    sa3_dit *dit = calloc(1, sizeof(*dit));
    if (!dit) {
        h3_st_free_header(&header);
        fail(error, error_size, "out of memory allocating the transformer");
        return NULL;
    }

    const uint64_t channels = SA3_DIT_CHANNELS;
    LOAD(preprocess, "preprocess_conv.weight", channels * channels);
    LOAD(postprocess, "postprocess_conv.weight", channels * channels);
    LOAD(cond_in, "to_cond_embed.0.weight", (uint64_t)EMBED * SA3_DIT_CONTEXT);
    LOAD(cond_out, "to_cond_embed.2.weight", (uint64_t)EMBED * EMBED);
    LOAD(global_in, "to_global_embed.0.weight",
         (uint64_t)EMBED * SA3_DIT_CONTEXT);
    LOAD(global_out, "to_global_embed.2.weight", (uint64_t)EMBED * EMBED);
    LOAD(timestep_in, "to_timestep_embed.0.weight",
         (uint64_t)EMBED * TIMESTEP_DIM);
    LOAD(timestep_in_bias, "to_timestep_embed.0.bias", EMBED);
    LOAD(timestep_out, "to_timestep_embed.2.weight", (uint64_t)EMBED * EMBED);
    LOAD(timestep_out_bias, "to_timestep_embed.2.bias", EMBED);
    LOAD(project_in, "transformer.project_in.weight",
         (uint64_t)EMBED * channels);
    LOAD(project_out, "transformer.project_out.weight",
         (uint64_t)channels * EMBED);
    LOAD(memory, "transformer.memory_tokens", (uint64_t)MEMORY_TOKENS * EMBED);
    LOAD(modulation_in, "transformer.global_cond_embedder.0.weight",
         (uint64_t)EMBED * EMBED);
    LOAD(modulation_in_bias, "transformer.global_cond_embedder.0.bias", EMBED);
    LOAD(modulation_out, "transformer.global_cond_embedder.2.weight",
         (uint64_t)6 * EMBED * EMBED);
    LOAD(modulation_out_bias, "transformer.global_cond_embedder.2.bias",
         6 * EMBED);

    for (int index = 0; index < DEPTH; index++) {
        char prefix[64], name[192];
        snprintf(prefix, sizeof(prefix), "transformer.layers.%d.", index);
        sa3_dit_block *block = &dit->blocks[index];

        LOAD_BLOCK(pre_norm, "pre_norm.weight", EMBED);
        LOAD_BLOCK(cross_attend_norm, "cross_attend_norm.weight", EMBED);
        LOAD_BLOCK(ff_norm, "ff_norm.weight", EMBED);
        LOAD_BLOCK(scale_shift_gate, "to_scale_shift_gate", 6 * EMBED);
        LOAD_BLOCK(self_qkv, "self_attn.to_qkv.weight",
                   (uint64_t)3 * EMBED * EMBED);
        LOAD_BLOCK(self_out, "self_attn.to_out.weight",
                   (uint64_t)EMBED * EMBED);
        LOAD_BLOCK(self_q_norm, "self_attn.q_norm.weight", HEAD_DIM);
        LOAD_BLOCK(self_k_norm, "self_attn.k_norm.weight", HEAD_DIM);
        LOAD_BLOCK(cross_q, "cross_attn.to_q.weight", (uint64_t)EMBED * EMBED);
        LOAD_BLOCK(cross_kv, "cross_attn.to_kv.weight",
                   (uint64_t)2 * EMBED * EMBED);
        LOAD_BLOCK(cross_out, "cross_attn.to_out.weight",
                   (uint64_t)EMBED * EMBED);
        LOAD_BLOCK(cross_q_norm, "cross_attn.q_norm.weight", HEAD_DIM);
        LOAD_BLOCK(cross_k_norm, "cross_attn.k_norm.weight", HEAD_DIM);
        LOAD_BLOCK(ff_in, "ff.ff.0.proj.weight",
                   (uint64_t)2 * FF_INNER * EMBED);
        LOAD_BLOCK(ff_in_bias, "ff.ff.0.proj.bias", 2 * FF_INNER);
        LOAD_BLOCK(ff_out, "ff.ff.2.weight", (uint64_t)EMBED * FF_INNER);
        LOAD_BLOCK(ff_out_bias, "ff.ff.2.bias", EMBED);
        LOAD_BLOCK(local_in, "to_local_embed.seq.0.weight",
                   (uint64_t)EMBED * LOCAL_DIM);
        LOAD_BLOCK(local_in_bias, "to_local_embed.seq.0.bias", EMBED);
        LOAD_BLOCK(local_out, "to_local_embed.seq.2.weight",
                   (uint64_t)EMBED * EMBED);
        LOAD_BLOCK(local_out_bias, "to_local_embed.seq.2.bias", EMBED);
    }

    /* Frequencies are spaced evenly in log space between the two bounds. */
    int half = TIMESTEP_DIM / 2;
    for (int index = 0; index < half; index++) {
        float ramp = half > 1 ? (float)index / (float)(half - 1) : 0.0f;
        dit->fourier[index] =
            expf(ramp * (logf(FOURIER_MAX) - logf(FOURIER_MIN)) +
                 logf(FOURIER_MIN)) * 2.0f * (float)M_PI;
    }

    h3_st_free_header(&header);
    return dit;

fail_out:
    h3_st_free_header(&header);
    sa3_dit_free(dit);
    return NULL;
}

#undef LOAD
#undef LOAD_BLOCK

void sa3_dit_free(sa3_dit *dit) {
    if (!dit) return;
    free(dit->preprocess); free(dit->postprocess);
    free(dit->cond_in); free(dit->cond_out);
    free(dit->global_in); free(dit->global_out);
    free(dit->timestep_in); free(dit->timestep_in_bias);
    free(dit->timestep_out); free(dit->timestep_out_bias);
    free(dit->project_in); free(dit->project_out); free(dit->memory);
    free(dit->modulation_in); free(dit->modulation_in_bias);
    free(dit->modulation_out); free(dit->modulation_out_bias);
    for (int index = 0; index < DEPTH; index++) {
        sa3_dit_block *block = &dit->blocks[index];
        free(block->pre_norm); free(block->cross_attend_norm);
        free(block->ff_norm); free(block->scale_shift_gate);
        free(block->self_qkv); free(block->self_out);
        free(block->self_q_norm); free(block->self_k_norm);
        free(block->cross_q); free(block->cross_kv); free(block->cross_out);
        free(block->cross_q_norm); free(block->cross_k_norm);
        free(block->ff_in); free(block->ff_in_bias);
        free(block->ff_out); free(block->ff_out_bias);
        free(block->local_in); free(block->local_in_bias);
        free(block->local_out); free(block->local_out_bias);
    }
    free(dit->sequence); free(dit->normed); free(dit->qkv);
    free(dit->heads_q); free(dit->heads_k); free(dit->heads_v);
    free(dit->attn); free(dit->scores); free(dit->ff); free(dit->gated);
    free(dit->proj); free(dit->context_kv); free(dit->context_out);
    free(dit->local); free(dit->rope_cos); free(dit->rope_sin);
    free(dit);
}

/* ---- primitives -------------------------------------------------------- */

static void linear(const float *input, const float *weight, const float *bias,
                   int rows, int in_features, int out_features, float *output) {
    if (bias)
        for (int row = 0; row < rows; row++)
            memcpy(output + (size_t)row * out_features, bias,
                   (size_t)out_features * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, out_features,
                in_features, 1.0f, input, in_features, weight, in_features,
                bias ? 1.0f : 0.0f, output, out_features);
}

static void rms_norm(float *values, int rows, int dim, const float *weight,
                     float eps) {
    for (int row = 0; row < rows; row++) {
        float *line = values + (size_t)row * dim;
        float sum = 0.0f;
        for (int index = 0; index < dim; index++) sum += line[index] * line[index];
        float inverse = 1.0f / sqrtf(sum / (float)dim + eps);
        for (int index = 0; index < dim; index++)
            line[index] = line[index] * inverse * weight[index];
    }
}

static void silu_inplace(float *values, size_t count) {
    for (size_t index = 0; index < count; index++)
        values[index] = values[index] / (1.0f + expf(-values[index]));
}

static void split_heads(const float *packed, int tokens, int stride,
                        int offset, float *heads) {
    for (int head = 0; head < HEADS; head++)
        for (int token = 0; token < tokens; token++)
            memcpy(heads + ((size_t)head * tokens + token) * HEAD_DIM,
                   packed + (size_t)token * stride + offset + head * HEAD_DIM,
                   HEAD_DIM * sizeof(float));
}

static void head_rms_norm(float *heads, int tokens, const float *weight) {
    rms_norm(heads, HEADS * tokens, HEAD_DIM, weight, QK_NORM_EPS);
}

static void apply_rope(float *heads, int tokens, const float *cos_table,
                       const float *sin_table) {
    int pairs = ROPE_DIMS / 2;
    for (int head = 0; head < HEADS; head++)
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

static void attend(const float *q, const float *k, const float *v,
                   int queries, int keys, float *scores, float *output) {
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (int head = 0; head < HEADS; head++) {
        const float *qh = q + (size_t)head * queries * HEAD_DIM;
        const float *kh = k + (size_t)head * keys * HEAD_DIM;
        const float *vh = v + (size_t)head * keys * HEAD_DIM;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, queries, keys,
                    HEAD_DIM, scale, qh, HEAD_DIM, kh, HEAD_DIM, 0.0f, scores,
                    keys);
        for (int query = 0; query < queries; query++) {
            float *row = scores + (size_t)query * keys;
            float largest = row[0];
            for (int index = 1; index < keys; index++)
                if (row[index] > largest) largest = row[index];
            float total = 0.0f;
            for (int index = 0; index < keys; index++) {
                row[index] = expf(row[index] - largest);
                total += row[index];
            }
            float inverse = 1.0f / total;
            float *destination =
                output + (size_t)query * EMBED + head * HEAD_DIM;
            for (int channel = 0; channel < HEAD_DIM; channel++) {
                float sum = 0.0f;
                for (int index = 0; index < keys; index++)
                    sum += row[index] * vh[(size_t)index * HEAD_DIM + channel];
                destination[channel] = sum * inverse;
            }
        }
    }
}

static int reserve(sa3_dit *dit, int tokens, int context_tokens,
                   char *error, size_t error_size) {
    if (tokens <= dit->capacity_tokens &&
        context_tokens <= dit->capacity_context)
        return 1;
    int wide = tokens > context_tokens ? tokens : context_tokens;
    free(dit->sequence); free(dit->normed); free(dit->qkv);
    free(dit->heads_q); free(dit->heads_k); free(dit->heads_v);
    free(dit->attn); free(dit->scores); free(dit->ff); free(dit->gated);
    free(dit->proj); free(dit->context_kv); free(dit->context_out);
    free(dit->local); free(dit->rope_cos); free(dit->rope_sin);

    size_t embed = (size_t)tokens * EMBED;
    dit->sequence = malloc(embed * sizeof(float));
    dit->normed = malloc(embed * sizeof(float));
    dit->qkv = malloc((size_t)tokens * 3 * EMBED * sizeof(float));
    dit->heads_q = malloc((size_t)HEADS * tokens * HEAD_DIM * sizeof(float));
    dit->heads_k = malloc((size_t)HEADS * wide * HEAD_DIM * sizeof(float));
    dit->heads_v = malloc((size_t)HEADS * wide * HEAD_DIM * sizeof(float));
    dit->attn = malloc(embed * sizeof(float));
    dit->scores = malloc((size_t)tokens * wide * sizeof(float));
    dit->ff = malloc((size_t)tokens * 2 * FF_INNER * sizeof(float));
    dit->gated = malloc((size_t)tokens * FF_INNER * sizeof(float));
    dit->proj = malloc(embed * sizeof(float));
    dit->context_kv = malloc((size_t)context_tokens * 2 * EMBED * sizeof(float));
    dit->context_out = malloc((size_t)context_tokens * EMBED * sizeof(float));
    dit->local = malloc(embed * sizeof(float));
    dit->rope_cos = malloc((size_t)tokens * (ROPE_DIMS / 2) * sizeof(float));
    dit->rope_sin = malloc((size_t)tokens * (ROPE_DIMS / 2) * sizeof(float));

    if (!dit->sequence || !dit->normed || !dit->qkv || !dit->heads_q ||
        !dit->heads_k || !dit->heads_v || !dit->attn || !dit->scores ||
        !dit->ff || !dit->gated || !dit->proj || !dit->context_kv ||
        !dit->context_out || !dit->local || !dit->rope_cos || !dit->rope_sin) {
        fail(error, error_size, "out of memory sizing the transformer for %d "
             "tokens", tokens);
        dit->capacity_tokens = dit->capacity_context = 0;
        return 0;
    }

    int pairs = ROPE_DIMS / 2;
    for (int token = 0; token < tokens; token++)
        for (int index = 0; index < pairs; index++) {
            float inverse = powf(ROPE_BASE,
                                 -2.0f * (float)index / (float)ROPE_DIMS);
            float angle = (float)token * inverse;
            dit->rope_cos[token * pairs + index] = cosf(angle);
            dit->rope_sin[token * pairs + index] = sinf(angle);
        }

    dit->capacity_tokens = tokens;
    dit->capacity_context = context_tokens;
    return 1;
}

int sa3_dit_forward(sa3_dit *dit, const float *latents, int frames,
                    const float *context, int context_tokens,
                    const float *global, float sigma, float *velocity,
                    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !latents || !context || !global || !velocity || frames < 1) {
        fail(error, error_size, "invalid arguments for the transformer");
        return 0;
    }
    int tokens = MEMORY_TOKENS + frames;
    if (!reserve(dit, tokens, context_tokens, error, error_size)) return 0;

    /* --- conditioning --- */
    linear(context, dit->cond_in, NULL, context_tokens, SA3_DIT_CONTEXT, EMBED,
           dit->context_out);
    silu_inplace(dit->context_out, (size_t)context_tokens * EMBED);
    float *projected_context =
        malloc((size_t)context_tokens * EMBED * sizeof(float));
    if (!projected_context) {
        fail(error, error_size, "out of memory projecting the context");
        return 0;
    }
    linear(dit->context_out, dit->cond_out, NULL, context_tokens, EMBED, EMBED,
           projected_context);

    float global_hidden[EMBED], global_embed[EMBED];
    linear(global, dit->global_in, NULL, 1, SA3_DIT_CONTEXT, EMBED,
           global_hidden);
    silu_inplace(global_hidden, EMBED);
    linear(global_hidden, dit->global_out, NULL, 1, EMBED, EMBED, global_embed);

    /* The timestep enters as paired cosine and sine of log-spaced frequencies. */
    float features[TIMESTEP_DIM];
    int half = TIMESTEP_DIM / 2;
    for (int index = 0; index < half; index++) {
        float angle = sigma * dit->fourier[index];
        features[index] = cosf(angle);
        features[half + index] = sinf(angle);
    }
    float timestep_hidden[EMBED], timestep_embed[EMBED];
    linear(features, dit->timestep_in, dit->timestep_in_bias, 1, TIMESTEP_DIM,
           EMBED, timestep_hidden);
    silu_inplace(timestep_hidden, EMBED);
    linear(timestep_hidden, dit->timestep_out, dit->timestep_out_bias, 1, EMBED,
           EMBED, timestep_embed);
    for (int index = 0; index < EMBED; index++)
        global_embed[index] += timestep_embed[index];

    float modulation_hidden[EMBED];
    linear(global_embed, dit->modulation_in, dit->modulation_in_bias, 1, EMBED,
           EMBED, modulation_hidden);
    silu_inplace(modulation_hidden, EMBED);
    linear(modulation_hidden, dit->modulation_out, dit->modulation_out_bias, 1,
           EMBED, 6 * EMBED, dit->modulation);

    /* --- input --- */
    float *rows = malloc((size_t)frames * SA3_DIT_CHANNELS * sizeof(float));
    float *preprocessed = malloc((size_t)frames * SA3_DIT_CHANNELS *
                                 sizeof(float));
    if (!rows || !preprocessed) {
        free(rows); free(preprocessed); free(projected_context);
        fail(error, error_size, "out of memory preparing latents");
        return 0;
    }
    for (int frame = 0; frame < frames; frame++)
        for (int channel = 0; channel < SA3_DIT_CHANNELS; channel++)
            rows[(size_t)frame * SA3_DIT_CHANNELS + channel] =
                latents[(size_t)channel * frames + frame];

    /* A width-1 convolution with a residual, so a plain matrix plus the input. */
    linear(rows, dit->preprocess, NULL, frames, SA3_DIT_CHANNELS,
           SA3_DIT_CHANNELS, preprocessed);
    for (int index = 0; index < frames * SA3_DIT_CHANNELS; index++)
        preprocessed[index] += rows[index];

    for (int token = 0; token < MEMORY_TOKENS; token++)
        memcpy(dit->sequence + (size_t)token * EMBED,
               dit->memory + (size_t)token * EMBED, EMBED * sizeof(float));
    linear(preprocessed, dit->project_in, NULL, frames, SA3_DIT_CHANNELS, EMBED,
           dit->sequence + (size_t)MEMORY_TOKENS * EMBED);

    /* --- blocks --- */
    const float *scale_self = dit->modulation;
    const float *shift_self = dit->modulation + EMBED;
    const float *gate_self = dit->modulation + 2 * EMBED;
    const float *scale_ff = dit->modulation + 3 * EMBED;
    const float *shift_ff = dit->modulation + 4 * EMBED;
    const float *gate_ff = dit->modulation + 5 * EMBED;

    for (int index = 0; index < DEPTH; index++) {
        sa3_dit_block *block = &dit->blocks[index];

        /* Each block folds its own learned offset into the shared modulation. */
        float self_scale[EMBED], self_shift[EMBED], self_gate[EMBED];
        float ff_scale[EMBED], ff_shift[EMBED], ff_gate[EMBED];
        for (int channel = 0; channel < EMBED; channel++) {
            self_scale[channel] = block->scale_shift_gate[channel] + scale_self[channel];
            self_shift[channel] = block->scale_shift_gate[EMBED + channel] + shift_self[channel];
            self_gate[channel] = block->scale_shift_gate[2 * EMBED + channel] + gate_self[channel];
            ff_scale[channel] = block->scale_shift_gate[3 * EMBED + channel] + scale_ff[channel];
            ff_shift[channel] = block->scale_shift_gate[4 * EMBED + channel] + shift_ff[channel];
            ff_gate[channel] = block->scale_shift_gate[5 * EMBED + channel] + gate_ff[channel];
        }

        /* self-attention */
        memcpy(dit->normed, dit->sequence,
               (size_t)tokens * EMBED * sizeof(float));
        rms_norm(dit->normed, tokens, EMBED, block->pre_norm, NORM_EPS);
        for (int token = 0; token < tokens; token++) {
            float *line = dit->normed + (size_t)token * EMBED;
            for (int channel = 0; channel < EMBED; channel++)
                line[channel] = line[channel] * (1.0f + self_scale[channel]) +
                                self_shift[channel];
        }
        linear(dit->normed, block->self_qkv, NULL, tokens, EMBED, 3 * EMBED,
               dit->qkv);
        split_heads(dit->qkv, tokens, 3 * EMBED, 0, dit->heads_q);
        split_heads(dit->qkv, tokens, 3 * EMBED, EMBED, dit->heads_k);
        split_heads(dit->qkv, tokens, 3 * EMBED, 2 * EMBED, dit->heads_v);
        head_rms_norm(dit->heads_q, tokens, block->self_q_norm);
        head_rms_norm(dit->heads_k, tokens, block->self_k_norm);
        apply_rope(dit->heads_q, tokens, dit->rope_cos, dit->rope_sin);
        apply_rope(dit->heads_k, tokens, dit->rope_cos, dit->rope_sin);
        attend(dit->heads_q, dit->heads_k, dit->heads_v, tokens, tokens,
               dit->scores, dit->attn);
        linear(dit->attn, block->self_out, NULL, tokens, EMBED, EMBED,
               dit->proj);
        /* The gate is sigmoid(1 - g), so a zero gate leaves most of it through. */
        for (int token = 0; token < tokens; token++) {
            float *line = dit->proj + (size_t)token * EMBED;
            float *destination = dit->sequence + (size_t)token * EMBED;
            for (int channel = 0; channel < EMBED; channel++)
                destination[channel] +=
                    line[channel] / (1.0f + expf(self_gate[channel] - 1.0f));
        }

        /* cross-attention onto the conditioning */
        memcpy(dit->normed, dit->sequence,
               (size_t)tokens * EMBED * sizeof(float));
        rms_norm(dit->normed, tokens, EMBED, block->cross_attend_norm,
                 NORM_EPS);
        linear(dit->normed, block->cross_q, NULL, tokens, EMBED, EMBED,
               dit->attn);
        split_heads(dit->attn, tokens, EMBED, 0, dit->heads_q);
        linear(projected_context, block->cross_kv, NULL, context_tokens, EMBED,
               2 * EMBED, dit->context_kv);
        split_heads(dit->context_kv, context_tokens, 2 * EMBED, 0,
                    dit->heads_k);
        split_heads(dit->context_kv, context_tokens, 2 * EMBED, EMBED,
                    dit->heads_v);
        head_rms_norm(dit->heads_q, tokens, block->cross_q_norm);
        head_rms_norm(dit->heads_k, context_tokens, block->cross_k_norm);
        attend(dit->heads_q, dit->heads_k, dit->heads_v, tokens,
               context_tokens, dit->scores, dit->attn);
        linear(dit->attn, block->cross_out, NULL, tokens, EMBED, EMBED,
               dit->proj);
        for (int index2 = 0; index2 < tokens * EMBED; index2++)
            dit->sequence[index2] += dit->proj[index2];

        /* The local conditioning is zero for text-to-audio, but the layers
         * still contribute their biases, so it cannot be skipped. */
        float *local_rows = calloc((size_t)frames * LOCAL_DIM, sizeof(float));
        if (!local_rows) {
            free(rows); free(preprocessed); free(projected_context);
            fail(error, error_size, "out of memory building local conditioning");
            return 0;
        }
        linear(local_rows, block->local_in, block->local_in_bias, frames,
               LOCAL_DIM, EMBED, dit->local);
        free(local_rows);
        silu_inplace(dit->local, (size_t)frames * EMBED);
        linear(dit->local, block->local_out, block->local_out_bias, frames,
               EMBED, EMBED, dit->proj);
        for (int frame = 0; frame < frames; frame++) {
            float *destination =
                dit->sequence + (size_t)(MEMORY_TOKENS + frame) * EMBED;
            const float *line = dit->proj + (size_t)frame * EMBED;
            for (int channel = 0; channel < EMBED; channel++)
                destination[channel] += line[channel];
        }

        /* feed-forward */
        memcpy(dit->normed, dit->sequence,
               (size_t)tokens * EMBED * sizeof(float));
        rms_norm(dit->normed, tokens, EMBED, block->ff_norm, NORM_EPS);
        for (int token = 0; token < tokens; token++) {
            float *line = dit->normed + (size_t)token * EMBED;
            for (int channel = 0; channel < EMBED; channel++)
                line[channel] = line[channel] * (1.0f + ff_scale[channel]) +
                                ff_shift[channel];
        }
        linear(dit->normed, block->ff_in, block->ff_in_bias, tokens, EMBED,
               2 * FF_INNER, dit->ff);
        for (int token = 0; token < tokens; token++) {
            const float *line = dit->ff + (size_t)token * 2 * FF_INNER;
            float *out = dit->gated + (size_t)token * FF_INNER;
            for (int channel = 0; channel < FF_INNER; channel++) {
                float gate = line[FF_INNER + channel];
                out[channel] = line[channel] * (gate / (1.0f + expf(-gate)));
            }
        }
        linear(dit->gated, block->ff_out, block->ff_out_bias, tokens, FF_INNER,
               EMBED, dit->proj);
        for (int token = 0; token < tokens; token++) {
            float *line = dit->proj + (size_t)token * EMBED;
            float *destination = dit->sequence + (size_t)token * EMBED;
            for (int channel = 0; channel < EMBED; channel++)
                destination[channel] +=
                    line[channel] / (1.0f + expf(ff_gate[channel] - 1.0f));
        }
    }
    free(projected_context);

    /* --- output --- */
    linear(dit->sequence + (size_t)MEMORY_TOKENS * EMBED, dit->project_out,
           NULL, frames, EMBED, SA3_DIT_CHANNELS, rows);
    linear(rows, dit->postprocess, NULL, frames, SA3_DIT_CHANNELS,
           SA3_DIT_CHANNELS, preprocessed);
    for (int frame = 0; frame < frames; frame++)
        for (int channel = 0; channel < SA3_DIT_CHANNELS; channel++)
            velocity[(size_t)channel * frames + frame] =
                preprocessed[(size_t)frame * SA3_DIT_CHANNELS + channel] +
                rows[(size_t)frame * SA3_DIT_CHANNELS + channel];

    free(rows);
    free(preprocessed);
    return 1;
}
