#include "sa3_text.h"

#include "h3_safetensors.h"

#include <Accelerate/Accelerate.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CPU T5Gemma encoder, checked against embeddings dumped from Stability's MLX
 * implementation.
 *
 * The embedding table is 256000 x 768 and dominates the file, so it is kept as
 * f16 and widened one row at a time during lookup rather than held twice. */

#define WIDTH SA3_TEXT_WIDTH
#define LAYERS 12
#define HEADS 12
#define HEAD_DIM 64
#define INNER 2048
#define VOCAB 256000
#define NORM_EPS 1e-6f
#define ROPE_THETA 10000.0f
#define SOFTCAP 50.0f
#define QUERY_SCALAR 64.0f

typedef struct {
    float *pre_attn_norm;
    float *post_attn_norm;
    float *pre_ff_norm;
    float *post_ff_norm;
    float *q, *k, *v, *o;
    float *gate, *up, *down;
} sa3_text_layer;

struct sa3_text {
    uint16_t *embed;       /* [VOCAB, WIDTH] f16, widened per row on lookup */
    float *final_norm;
    sa3_text_layer layers[LAYERS];

    float *hidden;         /* [tokens, WIDTH] */
    float *branch;
    float *query, *key, *value, *attn;
    float *scores;         /* [HEADS, tokens] one row at a time */
    float *gate_buffer, *up_buffer;
    float rope_cos[SA3_TEXT_MAX_TOKENS * HEAD_DIM / 2];
    float rope_sin[SA3_TEXT_MAX_TOKENS * HEAD_DIM / 2];
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static float *load_f32(const h3_st_header *header, const char *name,
                       uint64_t expected, char *error, size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor) {
        fail(error, error_size, "%s is missing from the text encoder", name);
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
    vImage_Buffer source = {raw, 1, (vImagePixelCount)count,
                            (size_t)count * sizeof(*raw)};
    vImage_Buffer destination = {values, 1, (vImagePixelCount)count,
                                 (size_t)count * sizeof(*values)};
    vImageConvert_Planar16FtoPlanarF(&source, &destination, 0);
    free(raw);
    return values;
}

sa3_text *sa3_text_load(const char *path, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    h3_st_header header;
    if (!h3_st_read_header(path, &header, error, error_size)) return NULL;

    sa3_text *text = calloc(1, sizeof(*text));
    if (!text) {
        h3_st_free_header(&header);
        fail(error, error_size, "out of memory allocating the text encoder");
        return NULL;
    }

    /* The embedding table stays f16: widening it would cost 750 MB to serve
     * lookups of at most 256 rows. */
    const h3_st_tensor *embed = h3_st_find(&header, "embed_tokens.weight");
    if (!embed || embed->dtype != H3_DTYPE_F16 ||
        h3_st_tensor_elements(embed) != (uint64_t)VOCAB * WIDTH) {
        fail(error, error_size, "the embedding table is missing or not F16");
        goto fail_out;
    }
    text->embed = malloc((size_t)VOCAB * WIDTH * sizeof(*text->embed));
    if (!text->embed ||
        !h3_st_read_data(&header, embed, text->embed,
                         (size_t)VOCAB * WIDTH * sizeof(*text->embed), error,
                         error_size))
        goto fail_out;

    text->final_norm = load_f32(&header, "norm.weight", WIDTH, error,
                                error_size);
    if (!text->final_norm) goto fail_out;

    for (int index = 0; index < LAYERS; index++) {
        char name[128];
        sa3_text_layer *layer = &text->layers[index];
        #define LAYER(field, suffix, count)                                 \
            do {                                                            \
                snprintf(name, sizeof(name), "layers.%d." suffix, index);   \
                layer->field = load_f32(&header, name, (count), error,      \
                                        error_size);                        \
                if (!layer->field) goto fail_out;                           \
            } while (0)
        LAYER(pre_attn_norm, "pre_self_attn_layernorm.weight", WIDTH);
        LAYER(post_attn_norm, "post_self_attn_layernorm.weight", WIDTH);
        LAYER(pre_ff_norm, "pre_feedforward_layernorm.weight", WIDTH);
        LAYER(post_ff_norm, "post_feedforward_layernorm.weight", WIDTH);
        LAYER(q, "self_attn.q_proj.weight", (uint64_t)WIDTH * WIDTH);
        LAYER(k, "self_attn.k_proj.weight", (uint64_t)WIDTH * WIDTH);
        LAYER(v, "self_attn.v_proj.weight", (uint64_t)WIDTH * WIDTH);
        LAYER(o, "self_attn.o_proj.weight", (uint64_t)WIDTH * WIDTH);
        LAYER(gate, "mlp.gate_proj.weight", (uint64_t)INNER * WIDTH);
        LAYER(up, "mlp.up_proj.weight", (uint64_t)INNER * WIDTH);
        LAYER(down, "mlp.down_proj.weight", (uint64_t)WIDTH * INNER);
        #undef LAYER
    }

    size_t tokens = SA3_TEXT_MAX_TOKENS;
    text->hidden = malloc(tokens * WIDTH * sizeof(float));
    text->branch = malloc(tokens * WIDTH * sizeof(float));
    text->query = malloc(tokens * WIDTH * sizeof(float));
    text->key = malloc(tokens * WIDTH * sizeof(float));
    text->value = malloc(tokens * WIDTH * sizeof(float));
    text->attn = malloc(tokens * WIDTH * sizeof(float));
    text->scores = malloc(tokens * sizeof(float));
    text->gate_buffer = malloc(tokens * INNER * sizeof(float));
    text->up_buffer = malloc(tokens * INNER * sizeof(float));
    if (!text->hidden || !text->branch || !text->query || !text->key ||
        !text->value || !text->attn || !text->scores || !text->gate_buffer ||
        !text->up_buffer) {
        fail(error, error_size, "out of memory allocating encoder scratch");
        goto fail_out;
    }

    /* Rotary here pairs dimension i with i + head_dim/2 and repeats the
     * frequency table across both halves. */
    int pairs = HEAD_DIM / 2;
    for (size_t token = 0; token < tokens; token++)
        for (int index = 0; index < pairs; index++) {
            float inverse = 1.0f / powf(ROPE_THETA,
                                        (float)(2 * index) / (float)HEAD_DIM);
            float angle = (float)token * inverse;
            text->rope_cos[token * pairs + index] = cosf(angle);
            text->rope_sin[token * pairs + index] = sinf(angle);
        }

    h3_st_free_header(&header);
    return text;

fail_out:
    h3_st_free_header(&header);
    sa3_text_free(text);
    return NULL;
}

void sa3_text_free(sa3_text *text) {
    if (!text) return;
    free(text->embed);
    free(text->final_norm);
    for (int index = 0; index < LAYERS; index++) {
        sa3_text_layer *layer = &text->layers[index];
        free(layer->pre_attn_norm); free(layer->post_attn_norm);
        free(layer->pre_ff_norm); free(layer->post_ff_norm);
        free(layer->q); free(layer->k); free(layer->v); free(layer->o);
        free(layer->gate); free(layer->up); free(layer->down);
    }
    free(text->hidden); free(text->branch); free(text->query);
    free(text->key); free(text->value); free(text->attn); free(text->scores);
    free(text->gate_buffer); free(text->up_buffer);
    free(text);
}

/* ---- primitives -------------------------------------------------------- */

static void linear(const float *input, const float *weight, int rows,
                   int in_features, int out_features, float *output) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, out_features,
                in_features, 1.0f, input, in_features, weight, in_features,
                0.0f, output, out_features);
}

/* Gemma normalisation: scale by (1 + weight), not by weight. */
static void gemma_norm(const float *input, const float *weight, int rows,
                       int width, float *output) {
    for (int row = 0; row < rows; row++) {
        const float *line = input + (size_t)row * width;
        float *out = output + (size_t)row * width;
        float sum = 0.0f;
        for (int index = 0; index < width; index++)
            sum += line[index] * line[index];
        float inverse = 1.0f / sqrtf(sum / (float)width + NORM_EPS);
        for (int index = 0; index < width; index++)
            out[index] = line[index] * inverse * (1.0f + weight[index]);
    }
}

static void apply_rope(float *values, int tokens, const float *cos_table,
                       const float *sin_table) {
    int pairs = HEAD_DIM / 2;
    for (int token = 0; token < tokens; token++)
        for (int head = 0; head < HEADS; head++) {
            float *vector = values + ((size_t)token * HEADS + head) * HEAD_DIM;
            const float *cosines = cos_table + (size_t)token * pairs;
            const float *sines = sin_table + (size_t)token * pairs;
            for (int index = 0; index < pairs; index++) {
                float low = vector[index];
                float high = vector[index + pairs];
                vector[index] = low * cosines[index] - high * sines[index];
                vector[index + pairs] = high * cosines[index] + low * sines[index];
            }
        }
}

int sa3_text_encode(sa3_text *text, const uint32_t *ids, int count,
                    float *embeddings, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!text || !ids || !embeddings || count < 1 ||
        count > SA3_TEXT_MAX_TOKENS) {
        fail(error, error_size, "invalid arguments for the text encoder");
        return 0;
    }
    int tokens = SA3_TEXT_MAX_TOKENS;

    /* Embedding lookup, scaled by sqrt(width) as Gemma does. Padded positions
     * still carry the pad row so the maths stays finite; the mask keeps them
     * out of every attention. */
    float normalizer = sqrtf((float)WIDTH);
    for (int token = 0; token < tokens; token++) {
        uint32_t id = token < count ? ids[token] : 0;
        if (id >= VOCAB) {
            fail(error, error_size, "token %d is out of range (%u)", token, id);
            return 0;
        }
        const uint16_t *row = text->embed + (size_t)id * WIDTH;
        float *destination = text->hidden + (size_t)token * WIDTH;
        vImage_Buffer source = {(void *)row, 1, WIDTH, WIDTH * sizeof(uint16_t)};
        vImage_Buffer output = {destination, 1, WIDTH, WIDTH * sizeof(float)};
        vImageConvert_Planar16FtoPlanarF(&source, &output, 0);
        for (int index = 0; index < WIDTH; index++)
            destination[index] *= normalizer;
    }

    float scale = 1.0f / sqrtf(QUERY_SCALAR);
    for (int index = 0; index < LAYERS; index++) {
        sa3_text_layer *layer = &text->layers[index];

        gemma_norm(text->hidden, layer->pre_attn_norm, tokens, WIDTH,
                   text->branch);
        linear(text->branch, layer->q, tokens, WIDTH, WIDTH, text->query);
        linear(text->branch, layer->k, tokens, WIDTH, WIDTH, text->key);
        linear(text->branch, layer->v, tokens, WIDTH, WIDTH, text->value);
        apply_rope(text->query, tokens, text->rope_cos, text->rope_sin);
        apply_rope(text->key, tokens, text->rope_cos, text->rope_sin);

        for (int token = 0; token < tokens; token++)
            for (int head = 0; head < HEADS; head++) {
                const float *q = text->query +
                    ((size_t)token * HEADS + head) * HEAD_DIM;
                float largest = -INFINITY;
                /* Only real tokens are visible, which is the mask. */
                for (int other = 0; other < count; other++) {
                    const float *k = text->key +
                        ((size_t)other * HEADS + head) * HEAD_DIM;
                    float sum = 0.0f;
                    for (int channel = 0; channel < HEAD_DIM; channel++)
                        sum += q[channel] * k[channel];
                    /* Gemma squashes the logits before the softmax. */
                    float logit = SOFTCAP * tanhf(sum * scale / SOFTCAP);
                    text->scores[other] = logit;
                    if (logit > largest) largest = logit;
                }
                float total = 0.0f;
                for (int other = 0; other < count; other++) {
                    text->scores[other] = expf(text->scores[other] - largest);
                    total += text->scores[other];
                }
                float inverse = 1.0f / total;
                float *destination = text->attn +
                    ((size_t)token * HEADS + head) * HEAD_DIM;
                for (int channel = 0; channel < HEAD_DIM; channel++) {
                    float sum = 0.0f;
                    for (int other = 0; other < count; other++)
                        sum += text->scores[other] *
                            text->value[((size_t)other * HEADS + head) *
                                        HEAD_DIM + channel];
                    destination[channel] = sum * inverse;
                }
            }

        linear(text->attn, layer->o, tokens, WIDTH, WIDTH, text->branch);
        /* Gemma normalises the branch again before adding it back. */
        gemma_norm(text->branch, layer->post_attn_norm, tokens, WIDTH,
                   text->query);
        for (int position = 0; position < tokens * WIDTH; position++)
            text->hidden[position] += text->query[position];

        gemma_norm(text->hidden, layer->pre_ff_norm, tokens, WIDTH,
                   text->branch);
        linear(text->branch, layer->gate, tokens, WIDTH, INNER,
               text->gate_buffer);
        linear(text->branch, layer->up, tokens, WIDTH, INNER, text->up_buffer);
        for (size_t position = 0; position < (size_t)tokens * INNER;
             position++) {
            /* tanh-approximated GELU, matching gelu_pytorch_tanh. */
            float value = text->gate_buffer[position];
            float inner = 0.7978845608028654f *
                (value + 0.044715f * value * value * value);
            text->gate_buffer[position] =
                0.5f * value * (1.0f + tanhf(inner)) * text->up_buffer[position];
        }
        linear(text->gate_buffer, layer->down, tokens, INNER, WIDTH,
               text->branch);
        gemma_norm(text->branch, layer->post_ff_norm, tokens, WIDTH,
                   text->query);
        for (int position = 0; position < tokens * WIDTH; position++)
            text->hidden[position] += text->query[position];
    }

    gemma_norm(text->hidden, text->final_norm, tokens, WIDTH, embeddings);
    return 1;
}
