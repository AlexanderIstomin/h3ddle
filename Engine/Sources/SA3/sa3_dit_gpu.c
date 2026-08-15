#include "sa3_dit_gpu.h"

#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <Accelerate/Accelerate.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Metal path for the SA3 transformer, built entirely from kernels h3.c
 * already has. Three mismatches between what those kernels compute and what
 * this model wants are resolved by reshaping data rather than by writing new
 * kernels, which keeps the engine's shader library untouched:
 *
 *  - h3_gate_bf16 multiplies by the raw gate, but SA3 wants sigmoid(1 - gate).
 *    The gate slots are derived once per step on the CPU, so the sigmoid is
 *    folded in before upload.
 *  - h3_swiglu_bf16 computes silu(first half) * second half; SA3 wants the
 *    opposite pairing, so the two halves of the feed-forward weight and bias
 *    are swapped at load.
 *  - The per-layer local conditioning is driven by an all-zero input for
 *    text-to-audio, which makes its output a constant vector per layer. It is
 *    evaluated once at load instead of twenty times per step.
 *
 * Cross-attention keys and values depend only on the prompt, so they are built
 * once per generation rather than once per step. */

#define EMBED 1024
#define DEPTH 20
#define HEADS 16
#define HEAD_DIM 64
#define ROPE_HALF 16     /* rotary covers the leading 32 of 64 dimensions */
#define FF_INNER 4096
#define MEMORY_TOKENS 64
#define LOCAL_DIM 257
#define TIMESTEP_DIM 256
#define NORM_EPS 1e-5f
#define QK_NORM_EPS 1e-6f
#define ROPE_BASE 10000.0f
#define FOURIER_MIN 0.5f
#define FOURIER_MAX 10000.0f
#define SLOTS 6

typedef struct {
    h3_gpu_tensor *pre_norm;
    h3_gpu_tensor *cross_attend_norm;
    h3_gpu_tensor *ff_norm;
    h3_gpu_tensor *self_qkv;
    h3_gpu_tensor *self_out;
    h3_gpu_tensor *self_q_norm;
    h3_gpu_tensor *self_k_norm;
    h3_gpu_tensor *cross_q;
    h3_gpu_tensor *cross_out;
    h3_gpu_tensor *cross_q_norm;
    h3_gpu_tensor *ff_in;
    h3_gpu_tensor *ff_in_bias;
    h3_gpu_tensor *ff_out;
    h3_gpu_tensor *ff_out_bias;

    /* Built once per prompt. */
    h3_gpu_tensor *cross_key;
    h3_gpu_tensor *cross_value;
    /* Built once per sequence length: the constant local-conditioning term
     * repeated across the latent rows. */
    h3_gpu_tensor *local_bias;

    float *scale_shift_gate;   /* [SLOTS * EMBED], host side */
    float *cross_kv_weight;    /* [2 * EMBED, EMBED], host side */
    float *cross_k_norm;       /* [HEAD_DIM], host side */
    float *local_constant;     /* [EMBED], host side */
    /* The shared modulation plus this block's learned offset, with the
     * gate slots already passed through their sigmoid. */
    h3_gpu_tensor *modulation;
} sa3_gpu_block;

struct sa3_dit_gpu {
    h3_gpu *gpu;  /* borrowed; the caller owns the context */

    /* Conditioning stays on the host: it is small, and only the modulation
     * changes between steps. */
    float *cond_in, *cond_out;
    float *global_in, *global_out;
    float *timestep_in, *timestep_in_bias;
    float *timestep_out, *timestep_out_bias;
    float *modulation_in, *modulation_in_bias;
    float *modulation_out, *modulation_out_bias;
    float *preprocess, *postprocess;
    float fourier[TIMESTEP_DIM / 2];

    /* Used only by the host-side entry and exit projections. */
    float *project_in, *project_out, *memory;
    h3_gpu_tensor *rope_cos;
    h3_gpu_tensor *rope_sin;
    /* AdaLN and the gates index modulation per row; one shared row here. */
    h3_gpu_tensor *row_map;
    sa3_gpu_block blocks[DEPTH];

    int frames, tokens, context_tokens;
    h3_gpu_tensor *sequence, *normed, *qkv;
    h3_gpu_tensor *query, *key, *value, *attn, *proj;
    h3_gpu_tensor *ff, *gated, *latent_in, *latent_out;
    float *host_scratch;   /* widest host staging buffer */
    size_t host_capacity;
    sa3_dit_gpu_probe probe;
    void *probe_opaque;
};

void sa3_dit_gpu_set_probe(sa3_dit_gpu *dit, sa3_dit_gpu_probe probe,
                           void *opaque) {
    if (!dit) return;
    dit->probe = probe;
    dit->probe_opaque = opaque;
}

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* ---- conversion -------------------------------------------------------- */

static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    /* Round to nearest even on the discarded low half. */
    uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
    return (uint16_t)(rounded >> 16);
}

static void widen_f16(const uint16_t *source, float *destination,
                      size_t count) {
    vImage_Buffer input = {(void *)source, 1, (vImagePixelCount)count,
                           count * sizeof(uint16_t)};
    vImage_Buffer output = {destination, 1, (vImagePixelCount)count,
                            count * sizeof(float)};
    vImageConvert_Planar16FtoPlanarF(&input, &output, 0);
}

/* Reads a tensor as f32 on the host, whatever it is stored as. */
static float *read_host(const h3_st_header *header, const char *name,
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
        if (h3_st_read_data(header, tensor, values,
                            (size_t)count * sizeof(*values), error, error_size))
            return values;
        free(values);
        return NULL;
    }
    if (tensor->dtype != H3_DTYPE_F16) {
        fail(error, error_size, "%s is %s, expected F16 or F32", name,
             h3_dtype_name(tensor->dtype));
        free(values);
        return NULL;
    }
    uint16_t *raw = malloc((size_t)count * sizeof(*raw));
    if (!raw || !h3_st_read_data(header, tensor, raw,
                                 (size_t)count * sizeof(*raw), error,
                                 error_size)) {
        free(raw);
        free(values);
        return NULL;
    }
    widen_f16(raw, values, (size_t)count);
    free(raw);
    return values;
}

static h3_gpu_tensor *upload(h3_gpu *gpu, const float *values, size_t count) {
    uint16_t *converted = malloc(count * sizeof(*converted));
    if (!converted) return NULL;
    for (size_t index = 0; index < count; index++)
        converted[index] = to_bf16(values[index]);
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(gpu, converted, count);
    free(converted);
    return tensor;
}

/* Overwrites an existing device tensor with host floats. */
static int upload_into(h3_gpu_tensor *tensor, const float *values,
                       size_t count) {
    uint16_t *converted = malloc(count * sizeof(*converted));
    if (!converted) return 0;
    for (size_t index = 0; index < count; index++)
        converted[index] = to_bf16(values[index]);
    int ok = h3_gpu_tensor_write_bf16(tensor, converted, count);
    free(converted);
    return ok;
}

/* Widens a device bf16 tensor back into host floats. */
static int download(const h3_gpu_tensor *tensor, float *values, size_t count) {
    uint16_t *raw = malloc(count * sizeof(*raw));
    if (!raw || !h3_gpu_tensor_read_bf16(tensor, raw, count)) {
        free(raw);
        return 0;
    }
    for (size_t index = 0; index < count; index++) {
        uint32_t bits = (uint32_t)raw[index] << 16;
        memcpy(&values[index], &bits, sizeof(float));
    }
    free(raw);
    return 1;
}

/* Reads a tensor straight onto the device as bf16. */
static h3_gpu_tensor *read_device(h3_gpu *gpu, const h3_st_header *header,
                                  const char *name, uint64_t expected,
                                  char *error, size_t error_size) {
    float *host = read_host(header, name, expected, error, error_size);
    if (!host) return NULL;
    uint64_t count = h3_st_tensor_elements(h3_st_find(header, name));
    h3_gpu_tensor *tensor = upload(gpu, host, (size_t)count);
    free(host);
    if (!tensor) fail(error, error_size, "cannot upload %s", name);
    return tensor;
}

/* ---- loading ----------------------------------------------------------- */

#define HOST(field, name, count)                                        \
    do {                                                                \
        dit->field = read_host(&header, (name), (count), error, error_size); \
        if (!dit->field) goto fail_out;                                 \
    } while (0)

#define DEVICE(field, name, count)                                      \
    do {                                                                \
        dit->field = read_device(gpu, &header, (name), (count), error,  \
                                 error_size);                           \
        if (!dit->field) goto fail_out;                                 \
    } while (0)

sa3_dit_gpu *sa3_dit_gpu_load(h3_gpu *gpu, const char *path, char *error,
                              size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!gpu) {
        fail(error, error_size, "no GPU context for the transformer");
        return NULL;
    }
    h3_st_header header;
    if (!h3_st_read_header(path, &header, error, error_size)) return NULL;

    sa3_dit_gpu *dit = calloc(1, sizeof(*dit));
    if (!dit) {
        h3_st_free_header(&header);
        fail(error, error_size, "out of memory allocating the transformer");
        return NULL;
    }
    dit->gpu = gpu;

    const uint64_t channels = SA3_DIT_CHANNELS;
    HOST(preprocess, "preprocess_conv.weight", channels * channels);
    HOST(postprocess, "postprocess_conv.weight", channels * channels);
    HOST(cond_in, "to_cond_embed.0.weight", (uint64_t)EMBED * SA3_DIT_CONTEXT);
    HOST(cond_out, "to_cond_embed.2.weight", (uint64_t)EMBED * EMBED);
    HOST(global_in, "to_global_embed.0.weight",
         (uint64_t)EMBED * SA3_DIT_CONTEXT);
    HOST(global_out, "to_global_embed.2.weight", (uint64_t)EMBED * EMBED);
    HOST(timestep_in, "to_timestep_embed.0.weight",
         (uint64_t)EMBED * TIMESTEP_DIM);
    HOST(timestep_in_bias, "to_timestep_embed.0.bias", EMBED);
    HOST(timestep_out, "to_timestep_embed.2.weight", (uint64_t)EMBED * EMBED);
    HOST(timestep_out_bias, "to_timestep_embed.2.bias", EMBED);
    HOST(modulation_in, "transformer.global_cond_embedder.0.weight",
         (uint64_t)EMBED * EMBED);
    HOST(modulation_in_bias, "transformer.global_cond_embedder.0.bias", EMBED);
    HOST(modulation_out, "transformer.global_cond_embedder.2.weight",
         (uint64_t)SLOTS * EMBED * EMBED);
    HOST(modulation_out_bias, "transformer.global_cond_embedder.2.bias",
         SLOTS * EMBED);

    HOST(project_in, "transformer.project_in.weight",
         (uint64_t)EMBED * channels);
    HOST(project_out, "transformer.project_out.weight",
         (uint64_t)channels * EMBED);
    HOST(memory, "transformer.memory_tokens",
         (uint64_t)MEMORY_TOKENS * EMBED);

    for (int index = 0; index < DEPTH; index++) {
        char prefix[64], name[192];
        snprintf(prefix, sizeof(prefix), "transformer.layers.%d.", index);
        sa3_gpu_block *block = &dit->blocks[index];

        #define BLOCK_DEVICE(field, suffix, count)                      \
            do {                                                        \
                snprintf(name, sizeof(name), "%s" suffix, prefix);      \
                block->field = read_device(gpu, &header, name, (count), \
                                           error, error_size);          \
                if (!block->field) goto fail_out;                       \
            } while (0)
        #define BLOCK_HOST(field, suffix, count)                        \
            do {                                                        \
                snprintf(name, sizeof(name), "%s" suffix, prefix);      \
                block->field = read_host(&header, name, (count), error, \
                                         error_size);                   \
                if (!block->field) goto fail_out;                       \
            } while (0)

        BLOCK_DEVICE(pre_norm, "pre_norm.weight", EMBED);
        BLOCK_DEVICE(cross_attend_norm, "cross_attend_norm.weight", EMBED);
        BLOCK_DEVICE(ff_norm, "ff_norm.weight", EMBED);
        BLOCK_DEVICE(self_qkv, "self_attn.to_qkv.weight",
                     (uint64_t)3 * EMBED * EMBED);
        BLOCK_DEVICE(self_out, "self_attn.to_out.weight",
                     (uint64_t)EMBED * EMBED);
        BLOCK_DEVICE(self_q_norm, "self_attn.q_norm.weight", HEAD_DIM);
        BLOCK_DEVICE(self_k_norm, "self_attn.k_norm.weight", HEAD_DIM);
        BLOCK_DEVICE(cross_q, "cross_attn.to_q.weight",
                     (uint64_t)EMBED * EMBED);
        BLOCK_DEVICE(cross_out, "cross_attn.to_out.weight",
                     (uint64_t)EMBED * EMBED);
        BLOCK_DEVICE(cross_q_norm, "cross_attn.q_norm.weight", HEAD_DIM);
        BLOCK_HOST(scale_shift_gate, "to_scale_shift_gate", SLOTS * EMBED);
        BLOCK_HOST(cross_kv_weight, "cross_attn.to_kv.weight",
                   (uint64_t)2 * EMBED * EMBED);
        BLOCK_HOST(cross_k_norm, "cross_attn.k_norm.weight", HEAD_DIM);

        /* The engine's SwiGLU applies the activation to the first half; this
         * model applies it to the second, so swap the halves once here. */
        snprintf(name, sizeof(name), "%sff.ff.0.proj.weight", prefix);
        float *ff_weight = read_host(&header, name,
                                     (uint64_t)2 * FF_INNER * EMBED, error,
                                     error_size);
        snprintf(name, sizeof(name), "%sff.ff.0.proj.bias", prefix);
        float *ff_bias = read_host(&header, name, 2 * FF_INNER, error,
                                   error_size);
        if (!ff_weight || !ff_bias) {
            free(ff_weight);
            free(ff_bias);
            goto fail_out;
        }
        float *swapped = malloc((size_t)2 * FF_INNER * EMBED * sizeof(float));
        float *swapped_bias = malloc((size_t)2 * FF_INNER * sizeof(float));
        if (!swapped || !swapped_bias) {
            free(ff_weight); free(ff_bias); free(swapped); free(swapped_bias);
            fail(error, error_size, "out of memory reordering block %d", index);
            goto fail_out;
        }
        memcpy(swapped, ff_weight + (size_t)FF_INNER * EMBED,
               (size_t)FF_INNER * EMBED * sizeof(float));
        memcpy(swapped + (size_t)FF_INNER * EMBED, ff_weight,
               (size_t)FF_INNER * EMBED * sizeof(float));
        memcpy(swapped_bias, ff_bias + FF_INNER, FF_INNER * sizeof(float));
        memcpy(swapped_bias + FF_INNER, ff_bias, FF_INNER * sizeof(float));
        free(ff_weight);
        free(ff_bias);
        block->ff_in = upload(gpu, swapped, (size_t)2 * FF_INNER * EMBED);
        block->ff_in_bias = upload(gpu, swapped_bias, 2 * FF_INNER);
        free(swapped);
        free(swapped_bias);
        if (!block->ff_in || !block->ff_in_bias) {
            fail(error, error_size, "cannot upload block %d feed-forward",
                 index);
            goto fail_out;
        }

        BLOCK_DEVICE(ff_out, "ff.ff.2.weight", (uint64_t)EMBED * FF_INNER);
        BLOCK_DEVICE(ff_out_bias, "ff.ff.2.bias", EMBED);

        /* Local conditioning sees zeros, so its two layers reduce to a
         * constant: silu(bias0) through the second layer, plus bias1. */
        float *local_in = NULL, *local_in_bias = NULL;
        float *local_out = NULL, *local_out_bias = NULL;
        snprintf(name, sizeof(name), "%sto_local_embed.seq.0.bias", prefix);
        local_in_bias = read_host(&header, name, EMBED, error, error_size);
        snprintf(name, sizeof(name), "%sto_local_embed.seq.2.weight", prefix);
        local_out = read_host(&header, name, (uint64_t)EMBED * EMBED, error,
                              error_size);
        snprintf(name, sizeof(name), "%sto_local_embed.seq.2.bias", prefix);
        local_out_bias = read_host(&header, name, EMBED, error, error_size);
        if (!local_in_bias || !local_out || !local_out_bias) {
            free(local_in); free(local_in_bias);
            free(local_out); free(local_out_bias);
            goto fail_out;
        }
        block->local_constant = malloc(EMBED * sizeof(float));
        if (!block->local_constant) {
            free(local_in_bias); free(local_out); free(local_out_bias);
            fail(error, error_size, "out of memory for block %d", index);
            goto fail_out;
        }
        float activated[EMBED];
        for (int channel = 0; channel < EMBED; channel++)
            activated[channel] = local_in_bias[channel] /
                (1.0f + expf(-local_in_bias[channel]));
        memcpy(block->local_constant, local_out_bias, EMBED * sizeof(float));
        cblas_sgemv(CblasRowMajor, CblasNoTrans, EMBED, EMBED, 1.0f, local_out,
                    EMBED, activated, 1, 1.0f, block->local_constant, 1);
        free(local_in_bias); free(local_out); free(local_out_bias);

        #undef BLOCK_DEVICE
        #undef BLOCK_HOST
    }

    int half = TIMESTEP_DIM / 2;
    for (int index = 0; index < half; index++) {
        float ramp = half > 1 ? (float)index / (float)(half - 1) : 0.0f;
        dit->fourier[index] =
            expf(ramp * (logf(FOURIER_MAX) - logf(FOURIER_MIN)) +
                 logf(FOURIER_MIN)) * 2.0f * (float)M_PI;
    }

    for (int index = 0; index < DEPTH; index++) {
        dit->blocks[index].modulation =
            h3_gpu_tensor_new_bf16(gpu, (size_t)SLOTS * EMBED);
        if (!dit->blocks[index].modulation) {
            fail(error, error_size, "cannot allocate block %d modulation",
                 index);
            goto fail_out;
        }
    }

    h3_st_free_header(&header);
    return dit;

fail_out:
    h3_st_free_header(&header);
    sa3_dit_gpu_free(dit);
    return NULL;
}

#undef HOST
#undef DEVICE

void sa3_dit_gpu_free(sa3_dit_gpu *dit) {
    if (!dit) return;
    free(dit->preprocess); free(dit->postprocess);
    free(dit->cond_in); free(dit->cond_out);
    free(dit->global_in); free(dit->global_out);
    free(dit->timestep_in); free(dit->timestep_in_bias);
    free(dit->timestep_out); free(dit->timestep_out_bias);
    free(dit->modulation_in); free(dit->modulation_in_bias);
    free(dit->modulation_out); free(dit->modulation_out_bias);
    free(dit->project_in);
    free(dit->project_out);
    free(dit->memory);
    h3_gpu_tensor_free(dit->rope_cos);
    h3_gpu_tensor_free(dit->rope_sin);
    h3_gpu_tensor_free(dit->row_map);
    for (int index = 0; index < DEPTH; index++) {
        sa3_gpu_block *block = &dit->blocks[index];
        h3_gpu_tensor_free(block->pre_norm);
        h3_gpu_tensor_free(block->cross_attend_norm);
        h3_gpu_tensor_free(block->ff_norm);
        h3_gpu_tensor_free(block->self_qkv);
        h3_gpu_tensor_free(block->self_out);
        h3_gpu_tensor_free(block->self_q_norm);
        h3_gpu_tensor_free(block->self_k_norm);
        h3_gpu_tensor_free(block->cross_q);
        h3_gpu_tensor_free(block->cross_out);
        h3_gpu_tensor_free(block->cross_q_norm);
        h3_gpu_tensor_free(block->ff_in);
        h3_gpu_tensor_free(block->ff_in_bias);
        h3_gpu_tensor_free(block->ff_out);
        h3_gpu_tensor_free(block->ff_out_bias);
        h3_gpu_tensor_free(block->cross_key);
        h3_gpu_tensor_free(block->cross_value);
        h3_gpu_tensor_free(block->local_bias);
        h3_gpu_tensor_free(block->modulation);
        free(block->scale_shift_gate);
        free(block->cross_kv_weight);
        free(block->cross_k_norm);
        free(block->local_constant);
    }
    h3_gpu_tensor_free(dit->sequence); h3_gpu_tensor_free(dit->normed);
    h3_gpu_tensor_free(dit->qkv); h3_gpu_tensor_free(dit->query);
    h3_gpu_tensor_free(dit->key); h3_gpu_tensor_free(dit->value);
    h3_gpu_tensor_free(dit->attn); h3_gpu_tensor_free(dit->proj);
    h3_gpu_tensor_free(dit->ff); h3_gpu_tensor_free(dit->gated);
    h3_gpu_tensor_free(dit->latent_in); h3_gpu_tensor_free(dit->latent_out);
    free(dit->host_scratch);
    free(dit);
}

/* ---- host helpers ------------------------------------------------------ */

static void linear_host(const float *input, const float *weight,
                        const float *bias, int rows, int in_features,
                        int out_features, float *output) {
    if (bias)
        for (int row = 0; row < rows; row++)
            memcpy(output + (size_t)row * out_features, bias,
                   (size_t)out_features * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, out_features,
                in_features, 1.0f, input, in_features, weight, in_features,
                bias ? 1.0f : 0.0f, output, out_features);
}

static void silu_host(float *values, size_t count) {
    for (size_t index = 0; index < count; index++)
        values[index] = values[index] / (1.0f + expf(-values[index]));
}

static void head_rms_norm_host(float *values, int rows, const float *weight) {
    for (int row = 0; row < rows; row++)
        for (int head = 0; head < HEADS; head++) {
            float *line = values + ((size_t)row * HEADS + head) * HEAD_DIM;
            float sum = 0.0f;
            for (int index = 0; index < HEAD_DIM; index++)
                sum += line[index] * line[index];
            float inverse = 1.0f / sqrtf(sum / (float)HEAD_DIM + QK_NORM_EPS);
            for (int index = 0; index < HEAD_DIM; index++)
                line[index] = line[index] * inverse * weight[index];
        }
}

static int ensure_host(sa3_dit_gpu *dit, size_t count) {
    if (count <= dit->host_capacity) return 1;
    free(dit->host_scratch);
    dit->host_scratch = malloc(count * sizeof(float));
    dit->host_capacity = dit->host_scratch ? count : 0;
    return dit->host_scratch != NULL;
}

/* ---- prompt-scoped setup ----------------------------------------------- */

int sa3_dit_gpu_set_context(sa3_dit_gpu *dit, const float *context,
                            int context_tokens, char *error,
                            size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !context || context_tokens < 1) {
        fail(error, error_size, "invalid context for the transformer");
        return 0;
    }

    size_t projected_count = (size_t)context_tokens * EMBED;
    float *hidden = malloc(projected_count * sizeof(float));
    float *projected = malloc(projected_count * sizeof(float));
    float *kv = malloc(projected_count * 2 * sizeof(float));
    if (!hidden || !projected || !kv) {
        free(hidden); free(projected); free(kv);
        fail(error, error_size, "out of memory projecting the context");
        return 0;
    }

    linear_host(context, dit->cond_in, NULL, context_tokens, SA3_DIT_CONTEXT,
                EMBED, hidden);
    silu_host(hidden, projected_count);
    linear_host(hidden, dit->cond_out, NULL, context_tokens, EMBED, EMBED,
                projected);

    int ok = 1;
    for (int index = 0; index < DEPTH && ok; index++) {
        sa3_gpu_block *block = &dit->blocks[index];
        linear_host(projected, block->cross_kv_weight, NULL, context_tokens,
                    EMBED, 2 * EMBED, kv);
        /* Split the packed projection into keys and values, normalising the
         * keys the way the attention expects. */
        for (int token = 0; token < context_tokens; token++)
            memcpy(hidden + (size_t)token * EMBED,
                   kv + (size_t)token * 2 * EMBED, EMBED * sizeof(float));
        head_rms_norm_host(hidden, context_tokens, block->cross_k_norm);
        h3_gpu_tensor_free(block->cross_key);
        block->cross_key = upload(dit->gpu, hidden, projected_count);

        for (int token = 0; token < context_tokens; token++)
            memcpy(hidden + (size_t)token * EMBED,
                   kv + (size_t)token * 2 * EMBED + EMBED,
                   EMBED * sizeof(float));
        h3_gpu_tensor_free(block->cross_value);
        block->cross_value = upload(dit->gpu, hidden, projected_count);

        if (!block->cross_key || !block->cross_value) {
            fail(error, error_size, "cannot upload block %d context", index);
            ok = 0;
        }
    }

    free(hidden);
    free(projected);
    free(kv);
    dit->context_tokens = ok ? context_tokens : 0;
    return ok;
}

static int reserve(sa3_dit_gpu *dit, int frames, char *error,
                   size_t error_size) {
    int tokens = MEMORY_TOKENS + frames;
    if (dit->frames == frames) return 1;

    h3_gpu_tensor_free(dit->sequence); h3_gpu_tensor_free(dit->normed);
    h3_gpu_tensor_free(dit->qkv); h3_gpu_tensor_free(dit->query);
    h3_gpu_tensor_free(dit->key); h3_gpu_tensor_free(dit->value);
    h3_gpu_tensor_free(dit->attn); h3_gpu_tensor_free(dit->proj);
    h3_gpu_tensor_free(dit->ff); h3_gpu_tensor_free(dit->gated);
    h3_gpu_tensor_free(dit->latent_in); h3_gpu_tensor_free(dit->latent_out);
    h3_gpu_tensor_free(dit->rope_cos); h3_gpu_tensor_free(dit->rope_sin);

    h3_gpu *gpu = dit->gpu;
    size_t embed = (size_t)tokens * EMBED;
    dit->sequence = h3_gpu_tensor_new_bf16(gpu, embed);
    dit->normed = h3_gpu_tensor_new_bf16(gpu, embed);
    dit->qkv = h3_gpu_tensor_new_bf16(gpu, embed * 3);
    dit->query = h3_gpu_tensor_new_bf16(gpu, embed);
    dit->key = h3_gpu_tensor_new_bf16(gpu, embed);
    dit->value = h3_gpu_tensor_new_bf16(gpu, embed);
    dit->attn = h3_gpu_tensor_new_bf16(gpu, embed);
    dit->proj = h3_gpu_tensor_new_bf16(gpu, embed);
    dit->ff = h3_gpu_tensor_new_bf16(gpu, (size_t)tokens * 2 * FF_INNER);
    dit->gated = h3_gpu_tensor_new_bf16(gpu, (size_t)tokens * FF_INNER);
    dit->latent_in = h3_gpu_tensor_new_bf16(gpu,
                                            (size_t)frames * SA3_DIT_CHANNELS);
    dit->latent_out = h3_gpu_tensor_new_bf16(gpu,
                                             (size_t)frames * SA3_DIT_CHANNELS);
    if (!dit->sequence || !dit->normed || !dit->qkv || !dit->query ||
        !dit->key || !dit->value || !dit->attn || !dit->proj || !dit->ff ||
        !dit->gated || !dit->latent_in || !dit->latent_out) {
        fail(error, error_size, "cannot allocate transformer buffers");
        dit->frames = 0;
        return 0;
    }

    /* Rotary tables cover the whole sequence, memory tokens included. */
    float *cos_values = malloc((size_t)tokens * ROPE_HALF * sizeof(float));
    float *sin_values = malloc((size_t)tokens * ROPE_HALF * sizeof(float));
    if (!cos_values || !sin_values) {
        free(cos_values); free(sin_values);
        fail(error, error_size, "out of memory building rotary tables");
        dit->frames = 0;
        return 0;
    }
    for (int token = 0; token < tokens; token++)
        for (int index = 0; index < ROPE_HALF; index++) {
            float inverse = powf(ROPE_BASE,
                                 -2.0f * (float)index / (float)(2 * ROPE_HALF));
            float angle = (float)token * inverse;
            cos_values[token * ROPE_HALF + index] = cosf(angle);
            sin_values[token * ROPE_HALF + index] = sinf(angle);
        }
    dit->rope_cos = upload(gpu, cos_values, (size_t)tokens * ROPE_HALF);
    dit->rope_sin = upload(gpu, sin_values, (size_t)tokens * ROPE_HALF);
    free(cos_values);
    free(sin_values);

    /* The constant local term, repeated across the latent rows. */
    float *rows = calloc((size_t)tokens * EMBED, sizeof(float));
    if (!rows || !dit->rope_cos || !dit->rope_sin) {
        free(rows);
        fail(error, error_size, "cannot allocate transformer tables");
        dit->frames = 0;
        return 0;
    }
    for (int index = 0; index < DEPTH; index++) {
        sa3_gpu_block *block = &dit->blocks[index];
        for (int frame = 0; frame < frames; frame++)
            memcpy(rows + (size_t)(MEMORY_TOKENS + frame) * EMBED,
                   block->local_constant, EMBED * sizeof(float));
        h3_gpu_tensor_free(block->local_bias);
        block->local_bias = upload(gpu, rows, (size_t)tokens * EMBED);
        if (!block->local_bias) {
            free(rows);
            fail(error, error_size, "cannot upload block %d local term", index);
            dit->frames = 0;
            return 0;
        }
    }
    free(rows);

    h3_gpu_tensor_free(dit->row_map);
    uint32_t *map = calloc((size_t)tokens, sizeof(*map));
    if (map) {
        dit->row_map = h3_gpu_tensor_from_u32(gpu, map, (size_t)tokens);
        free(map);
    }
    if (!dit->row_map) {
        fail(error, error_size, "cannot allocate the row map");
        dit->frames = 0;
        return 0;
    }

    if (!ensure_host(dit, (size_t)tokens * 3 * EMBED)) {
        fail(error, error_size, "out of memory staging transformer rows");
        dit->frames = 0;
        return 0;
    }

    dit->frames = frames;
    dit->tokens = tokens;
    return 1;
}

int sa3_dit_gpu_forward(sa3_dit_gpu *dit, const float *latents, int frames,
                        const float *global, float sigma, float *velocity,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !latents || !global || !velocity || frames < 1) {
        fail(error, error_size, "invalid arguments for the transformer");
        return 0;
    }
    if (dit->context_tokens < 1) {
        fail(error, error_size, "the transformer has no context set");
        return 0;
    }
    if (!reserve(dit, frames, error, error_size)) return 0;
    int tokens = dit->tokens;

    /* --- conditioning, on the host --- */
    float global_hidden[EMBED], global_embed[EMBED];
    linear_host(global, dit->global_in, NULL, 1, SA3_DIT_CONTEXT, EMBED,
                global_hidden);
    silu_host(global_hidden, EMBED);
    linear_host(global_hidden, dit->global_out, NULL, 1, EMBED, EMBED,
                global_embed);

    float features[TIMESTEP_DIM];
    int half = TIMESTEP_DIM / 2;
    for (int index = 0; index < half; index++) {
        float angle = sigma * dit->fourier[index];
        features[index] = cosf(angle);
        features[half + index] = sinf(angle);
    }
    float timestep_hidden[EMBED], timestep_embed[EMBED];
    linear_host(features, dit->timestep_in, dit->timestep_in_bias, 1,
                TIMESTEP_DIM, EMBED, timestep_hidden);
    silu_host(timestep_hidden, EMBED);
    linear_host(timestep_hidden, dit->timestep_out, dit->timestep_out_bias, 1,
                EMBED, EMBED, timestep_embed);
    for (int index = 0; index < EMBED; index++)
        global_embed[index] += timestep_embed[index];

    float modulation_hidden[EMBED];
    float *modulation = dit->host_scratch;
    linear_host(global_embed, dit->modulation_in, dit->modulation_in_bias, 1,
                EMBED, EMBED, modulation_hidden);
    silu_host(modulation_hidden, EMBED);
    linear_host(modulation_hidden, dit->modulation_out,
                dit->modulation_out_bias, 1, EMBED, SLOTS * EMBED, modulation);

    /* --- input projection, on the host --- */
    float *rows = modulation + SLOTS * EMBED;
    for (int frame = 0; frame < frames; frame++)
        for (int channel = 0; channel < SA3_DIT_CHANNELS; channel++)
            rows[(size_t)frame * SA3_DIT_CHANNELS + channel] =
                latents[(size_t)channel * frames + frame];
    float *preprocessed = rows + (size_t)frames * SA3_DIT_CHANNELS;
    linear_host(rows, dit->preprocess, NULL, frames, SA3_DIT_CHANNELS,
                SA3_DIT_CHANNELS, preprocessed);
    for (int index = 0; index < frames * SA3_DIT_CHANNELS; index++)
        preprocessed[index] += rows[index];

    /* The sequence is assembled on the host: the input projection is a few
     * megaflops, and doing it here avoids needing an offset-aware linear so
     * the memory tokens and the latents can share one buffer. */
    float *sequence = preprocessed + (size_t)frames * SA3_DIT_CHANNELS;
    memcpy(sequence, dit->memory,
           (size_t)MEMORY_TOKENS * EMBED * sizeof(float));
    linear_host(preprocessed, dit->project_in, NULL, frames,
                SA3_DIT_CHANNELS, EMBED,
                sequence + (size_t)MEMORY_TOKENS * EMBED);

    uint16_t *staged = malloc((size_t)tokens * EMBED * sizeof(*staged));
    if (!staged) {
        fail(error, error_size, "out of memory staging the sequence");
        return 0;
    }
    for (size_t index = 0; index < (size_t)tokens * EMBED; index++)
        staged[index] = to_bf16(sequence[index]);
    int ok = h3_gpu_tensor_write_bf16(dit->sequence, staged,
                                      (size_t)tokens * EMBED);
    free(staged);
    if (!ok) {
        fail(error, error_size, "cannot upload the sequence");
        return 0;
    }

    /* Every block adds its own learned offset, and the engine's gate kernel
     * multiplies by the modulation directly, so this model's sigmoid(1 - g)
     * is folded in before upload. */
    float *block_modulation = modulation + (size_t)SLOTS * EMBED;
    for (int index = 0; index < DEPTH; index++) {
        sa3_gpu_block *block = &dit->blocks[index];
        for (int slot = 0; slot < SLOTS; slot++)
            for (int channel = 0; channel < EMBED; channel++) {
                size_t at = (size_t)slot * EMBED + channel;
                float value = block->scale_shift_gate[at] + modulation[at];
                block_modulation[at] = (slot % 3 == 2)
                    ? 1.0f / (1.0f + expf(value - 1.0f)) : value;
            }
        if (!upload_into(block->modulation, block_modulation,
                         (size_t)SLOTS * EMBED)) {
            fail(error, error_size, "cannot upload block %d modulation",
                 index);
            return 0;
        }
    }

    /* --- blocks --- */
    if (!h3_gpu_begin(dit->gpu)) {
        fail(error, error_size, "cannot begin a GPU command buffer");
        return 0;
    }
    h3_gpu *gpu = dit->gpu;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    uint32_t row_count = (uint32_t)tokens;
    uint32_t context = (uint32_t)dit->context_tokens;

    for (int index = 0; index < DEPTH && ok; index++) {
        sa3_gpu_block *block = &dit->blocks[index];

        /* self-attention: normalise and modulate, project, rotate, attend */
        ok = h3_gpu_adaln_bf16(gpu, dit->normed, dit->sequence,
                               block->pre_norm, block->modulation,
                               dit->row_map, row_count, EMBED, SLOTS, 1, 0,
                               NORM_EPS) &&
             h3_gpu_linear_bf16(gpu, dit->qkv, dit->normed, block->self_qkv,
                                NULL, row_count, EMBED, 3 * EMBED) &&
             h3_gpu_qkv_rope_bf16(gpu, dit->query, dit->key, dit->value,
                                  dit->qkv, block->self_q_norm,
                                  block->self_k_norm, dit->rope_cos,
                                  dit->rope_sin, row_count, HEADS, HEAD_DIM,
                                  ROPE_HALF, QK_NORM_EPS) &&
             h3_gpu_sdpa_bf16(gpu, dit->attn, dit->query, dit->key,
                              dit->value, row_count, HEADS, HEAD_DIM, scale) &&
             h3_gpu_linear_bf16(gpu, dit->proj, dit->attn, block->self_out,
                                NULL, row_count, EMBED, EMBED) &&
             h3_gpu_gate_bf16(gpu, dit->sequence, dit->sequence, dit->proj,
                              block->modulation, dit->row_map, row_count, EMBED,
                              SLOTS, 2);

        /* cross-attention onto the prompt; keys and values were built once */
        ok = ok &&
             h3_gpu_rms_norm_bf16(gpu, dit->normed, dit->sequence,
                                  block->cross_attend_norm, row_count, EMBED,
                                  NORM_EPS) &&
             h3_gpu_linear_bf16(gpu, dit->query, dit->normed, block->cross_q,
                                NULL, row_count, EMBED, EMBED) &&
             h3_gpu_head_rms_norm_bf16(gpu, dit->query, block->cross_q_norm,
                                       row_count, HEADS, HEAD_DIM, QK_NORM_EPS) &&
             h3_gpu_sdpa_cross_bf16(gpu, dit->attn, dit->query,
                                    block->cross_key, block->cross_value,
                                    row_count, context, HEADS, HEAD_DIM, scale) &&
             h3_gpu_linear_bf16(gpu, dit->proj, dit->attn, block->cross_out,
                                NULL, row_count, EMBED, EMBED) &&
             h3_gpu_add_bf16(gpu, dit->sequence, dit->sequence, dit->proj,
                             (uint32_t)((size_t)tokens * EMBED));

        /* the constant local term, zero across the memory tokens */
        ok = ok &&
             h3_gpu_add_bf16(gpu, dit->sequence, dit->sequence,
                             block->local_bias,
                             (uint32_t)((size_t)tokens * EMBED));

        /* gated feed-forward */
        ok = ok &&
             h3_gpu_adaln_bf16(gpu, dit->normed, dit->sequence, block->ff_norm,
                               block->modulation, dit->row_map, row_count, EMBED,
                               SLOTS, 4, 3, NORM_EPS) &&
             h3_gpu_linear_bf16(gpu, dit->ff, dit->normed, block->ff_in,
                                block->ff_in_bias, row_count, EMBED,
                                2 * FF_INNER) &&
             h3_gpu_swiglu_bf16(gpu, dit->gated, dit->ff, row_count, FF_INNER) &&
             h3_gpu_linear_bf16(gpu, dit->proj, dit->gated, block->ff_out,
                                block->ff_out_bias, row_count, FF_INNER, EMBED) &&
             h3_gpu_gate_bf16(gpu, dit->sequence, dit->sequence, dit->proj,
                              block->modulation, dit->row_map, row_count, EMBED,
                              SLOTS, 5);

        if (!ok) fail(error, error_size, "block %d failed on the GPU", index);
        if (ok && dit->probe) {
            /* Draining the queue here is the point: the caller wants the
             * state this block actually produced. */
            if (!h3_gpu_submit(gpu) ||
                !download(dit->sequence, sequence, (size_t)tokens * EMBED)) {
                fail(error, error_size, "cannot probe block %d", index);
                return 0;
            }
            dit->probe(index, sequence, tokens, dit->probe_opaque);
            if (!h3_gpu_begin(gpu)) {
                fail(error, error_size, "cannot resume after block %d",
                     index);
                return 0;
            }
        }
    }

    if (!h3_gpu_submit(gpu)) {
        fail(error, error_size, "the GPU rejected the transformer pass");
        return 0;
    }
    if (!ok) return 0;

    /* --- output, back on the host --- */
    if (!download(dit->sequence, sequence, (size_t)tokens * EMBED)) {
        fail(error, error_size, "cannot read the transformer output");
        return 0;
    }
    linear_host(sequence + (size_t)MEMORY_TOKENS * EMBED,
                dit->project_out, NULL, frames, EMBED, SA3_DIT_CHANNELS,
                preprocessed);

    linear_host(preprocessed, dit->postprocess, NULL, frames,
                SA3_DIT_CHANNELS, SA3_DIT_CHANNELS, rows);
    for (int frame = 0; frame < frames; frame++)
        for (int channel = 0; channel < SA3_DIT_CHANNELS; channel++)
            velocity[(size_t)channel * frames + frame] =
                rows[(size_t)frame * SA3_DIT_CHANNELS + channel] +
                preprocessed[(size_t)frame * SA3_DIT_CHANNELS + channel];
    return 1;
}
