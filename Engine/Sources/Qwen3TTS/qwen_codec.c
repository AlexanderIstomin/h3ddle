#include "qwen_codec.h"

#include "qwen_weights.h"

#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LATENT        1024
#define QUANT_WIDTH   512
#define CODEBOOK_DIM  256
#define CODEBOOK_SIZE 2048
#define HIDDEN        512
#define ATTENTION     1024        /* 16 heads x 64, wider than the residual */
#define HEADS         16
#define HEAD_DIM      64
#define FFN           1024
#define LAYERS        8
#define WINDOW        72
#define RMS_EPSILON   1e-5f
#define ROPE_THETA    10000.0f
#define DECODER_DIM   1536
#define STAGES        4

static const int UPSAMPLE_RATES[STAGES] = {8, 5, 4, 3};
static const int DILATIONS[3] = {1, 3, 9};

typedef struct {
    const float *weight, *bias;
    int out_channels, in_channels, kernel;
} conv1d;

typedef struct {
    const float *alpha, *beta;
} snake;

typedef struct {
    snake act1, act2;
    conv1d conv1, conv2;
} residual_unit;

typedef struct {
    snake act;
    conv1d up;                    /* transposed */
    residual_unit units[3];
} vocoder_stage;

typedef struct {
    const float *input_layernorm, *post_attention_layernorm;
    const float *q_proj, *k_proj, *v_proj, *o_proj;
    const float *gate_proj, *up_proj, *down_proj;
    const float *attention_scale, *mlp_scale;
} codec_layer;

typedef struct {
    conv1d dwconv;
    const float *norm_weight, *norm_bias;
    const float *pwconv1_weight, *pwconv1_bias;
    const float *pwconv2_weight, *pwconv2_bias;
    const float *gamma;
} convnext;

struct qwen_codec {
    qwen_weights *weights;

    const float *semantic_codebook;              /* [2048, 256] */
    const float *acoustic_codebooks[QWEN_CODEC_GROUPS - 1];
    const float *semantic_output_proj;           /* [512, 256, 1] */
    const float *acoustic_output_proj;

    conv1d pre_conv;
    const float *input_proj_weight, *input_proj_bias;
    const float *output_proj_weight, *output_proj_bias;
    const float *final_norm;
    codec_layer layers[LAYERS];

    conv1d upsample_conv[2];                     /* transposed, stride 2 */
    convnext upsample_next[2];

    conv1d head_conv;                            /* 1024 -> 1536 */
    vocoder_stage stages[STAGES];
    snake final_act;
    conv1d output_conv;                          /* 96 -> 1 */

    qwen_codec_probe probe;
    void *probe_opaque;
};

static void report(const qwen_codec *codec, const char *stage,
                   const float *values, int channels, int length) {
    if (codec->probe) codec->probe(stage, values, channels, length,
                                   codec->probe_opaque);
}

/* ---- convolution -------------------------------------------------------
 *
 * Causal: all the padding goes on the left, so output t sees inputs up to t
 * and no further. With stride 1 the reference's extra right-hand padding term
 * works out to zero, which is why it does not appear here. */
/* Loop order matters enormously here. The natural reading — for each output
 * position, for each tap, for each input channel — puts the channel loop
 * innermost, where consecutive iterations are a whole row apart in memory. At
 * 768 channels over a 1600-sample row that is a cache miss per multiply, and
 * it made the vocoder ~90% of generation time.
 *
 * Hoisting position to the innermost loop instead makes both the read and the
 * accumulation contiguous, with the weight a broadcast scalar. */
static void causal_conv(const conv1d *conv, const float *input, int length,
                        float *output, int dilation, int groups) {
    const int span = (conv->kernel - 1) * dilation;
    const int in_per_group = conv->in_channels / groups;
    const int out_per_group = conv->out_channels / groups;
    dispatch_apply((size_t)conv->out_channels, DISPATCH_APPLY_AUTO,
                   ^(size_t slot) {
        const int out_channel = (int)slot;
        const int group = out_channel / out_per_group;
        float *row = output + (size_t)out_channel * length;
        const float bias = conv->bias ? conv->bias[out_channel] : 0.0f;
        for (int position = 0; position < length; position++) row[position] = bias;

        for (int channel = 0; channel < in_per_group; channel++) {
            const int in_channel = group * in_per_group + channel;
            const float *source = input + (size_t)in_channel * length;
            const float *weights = conv->weight +
                ((size_t)out_channel * in_per_group + channel) * conv->kernel;
            for (int tap = 0; tap < conv->kernel; tap++) {
                const float weight = weights[tap];
                if (weight == 0.0f) continue;
                /* the sample this tap reads is position + offset, and the left
                 * padding is exactly where that would be negative */
                const int offset = tap * dilation - span;
                int position = offset < 0 ? -offset : 0;
                const float32x4_t broadcast = vdupq_n_f32(weight);
                for (; position + 4 <= length; position += 4)
                    vst1q_f32(row + position,
                              vfmaq_f32(vld1q_f32(row + position), broadcast,
                                        vld1q_f32(source + position + offset)));
                for (; position < length; position++)
                    row[position] += weight * source[position + offset];
            }
        }
    });
}

/* ConvTranspose1d then trim (kernel - stride) from the right, so the result is
 * exactly length * stride and stays causal. */
static void causal_trans_conv(const conv1d *conv, const float *input,
                              int length, float *output, int stride) {
    const int out_length = length * stride;
    const int raw_length = (length - 1) * stride + conv->kernel;
    (void)raw_length;
    dispatch_apply((size_t)conv->out_channels, DISPATCH_APPLY_AUTO,
                   ^(size_t slot) {
        const int out_channel = (int)slot;
        float *row = output + (size_t)out_channel * out_length;
        const float bias = conv->bias ? conv->bias[out_channel] : 0.0f;
        for (int position = 0; position < out_length; position++)
            row[position] = bias;
        for (int in_channel = 0; in_channel < conv->in_channels; in_channel++) {
            const float *weights = conv->weight +
                ((size_t)in_channel * conv->out_channels + out_channel) *
                conv->kernel;
            const float *source = input + (size_t)in_channel * length;
            for (int step = 0; step < length; step++) {
                const float value = source[step];
                if (value == 0.0f) continue;
                for (int tap = 0; tap < conv->kernel; tap++) {
                    const int target = step * stride + tap;
                    if (target >= out_length) break;   /* the right-hand trim */
                    row[target] += value * weights[tap];
                }
            }
        }
    });
}

/* SnakeBeta: x + 1/exp(beta) * sin^2(x * exp(alpha)).
 *
 * Both parameters are stored as logs and exponentiated here. Using them raw
 * leaves the activation near-linear, which degrades the vocoder without
 * breaking it — the kind of fault that survives a listening test. */
static void snake_beta(const snake *parameters, float *values, int channels,
                       int length) {
    dispatch_apply((size_t)channels, DISPATCH_APPLY_AUTO, ^(size_t slot) {
        const int channel = (int)slot;
        const float alpha = expf(parameters->alpha[channel]);
        const float beta = 1.0f / (expf(parameters->beta[channel]) + 1e-9f);
        float *row = values + (size_t)channel * length;
        for (int position = 0; position < length; position++) {
            const float sine = sinf(row[position] * alpha);
            row[position] += beta * sine * sine;
        }
    });
}

/* ---- the transformer ---------------------------------------------------- */
static void rms_norm(const float *x, const float *weight, float *y,
                     int width, int rows) {
    for (int row = 0; row < rows; row++) {
        const float *in = x + (size_t)row * width;
        float *out = y + (size_t)row * width;
        float sum = 0.0f;
        for (int index = 0; index < width; index++) sum += in[index] * in[index];
        const float scale = 1.0f / sqrtf(sum / (float)width + RMS_EPSILON);
        for (int index = 0; index < width; index++)
            out[index] = weight[index] * in[index] * scale;
    }
}

/* y[rows][outputs] = x[rows][inputs] . weight[outputs][inputs]^T + bias */
static void linear(const float *weight, const float *bias, const float *x,
                   float *y, int inputs, int outputs, int rows) {
    dispatch_apply((size_t)outputs, DISPATCH_APPLY_AUTO, ^(size_t slot) {
        const int output = (int)slot;
        const float *w = weight + (size_t)output * inputs;
        for (int row = 0; row < rows; row++) {
            const float *source = x + (size_t)row * inputs;
            float total = bias ? bias[output] : 0.0f;
            for (int index = 0; index < inputs; index++)
                total += w[index] * source[index];
            y[(size_t)row * outputs + output] = total;
        }
    });
}

static void rotate(float *head, int position) {
    for (int index = 0; index < HEAD_DIM / 2; index++) {
        const float angle = (float)position *
            powf(ROPE_THETA, -(float)(2 * index) / (float)HEAD_DIM);
        const float cosine = cosf(angle), sine = sinf(angle);
        const float low = head[index], high = head[index + HEAD_DIM / 2];
        head[index] = low * cosine - high * sine;
        head[index + HEAD_DIM / 2] = high * cosine + low * sine;
    }
}

static void transformer(const qwen_codec *codec, float *x, int frames,
                        float *normed, float *query, float *key, float *value,
                        float *attention, float *gate, float *up) {
    for (int index = 0; index < LAYERS; index++) {
        const codec_layer *layer = &codec->layers[index];
        rms_norm(x, layer->input_layernorm, normed, HIDDEN, frames);
        linear(layer->q_proj, NULL, normed, query, HIDDEN, ATTENTION, frames);
        linear(layer->k_proj, NULL, normed, key, HIDDEN, ATTENTION, frames);
        linear(layer->v_proj, NULL, normed, value, HIDDEN, ATTENTION, frames);

        for (int frame = 0; frame < frames; frame++)
            for (int head = 0; head < HEADS; head++) {
                rotate(query + (size_t)frame * ATTENTION + head * HEAD_DIM, frame);
                rotate(key + (size_t)frame * ATTENTION + head * HEAD_DIM, frame);
            }

        const float scale = 1.0f / sqrtf((float)HEAD_DIM);
        dispatch_apply((size_t)frames, DISPATCH_APPLY_AUTO, ^(size_t slot) {
            const int frame = (int)slot;
            /* causal and windowed: frame t attends to (t - 71 .. t) */
            int first = frame - (WINDOW - 1);
            if (first < 0) first = 0;
            float scores[WINDOW];
            for (int head = 0; head < HEADS; head++) {
                const float *q = query + (size_t)frame * ATTENTION +
                    head * HEAD_DIM;
                float highest = -INFINITY;
                for (int step = first; step <= frame; step++) {
                    const float *k = key + (size_t)step * ATTENTION +
                        head * HEAD_DIM;
                    float total = 0.0f;
                    for (int channel = 0; channel < HEAD_DIM; channel++)
                        total += q[channel] * k[channel];
                    total *= scale;
                    scores[step - first] = total;
                    if (total > highest) highest = total;
                }
                float sum = 0.0f;
                for (int step = first; step <= frame; step++) {
                    scores[step - first] = expf(scores[step - first] - highest);
                    sum += scores[step - first];
                }
                float *out = attention + (size_t)frame * ATTENTION +
                    head * HEAD_DIM;
                memset(out, 0, HEAD_DIM * sizeof(float));
                for (int step = first; step <= frame; step++) {
                    const float weight = scores[step - first] / sum;
                    const float *v = value + (size_t)step * ATTENTION +
                        head * HEAD_DIM;
                    for (int channel = 0; channel < HEAD_DIM; channel++)
                        out[channel] += weight * v[channel];
                }
            }
        });

        linear(layer->o_proj, NULL, attention, normed, ATTENTION, HIDDEN, frames);
        /* layer scale: a learnt per-channel gain on the residual branch */
        for (int frame = 0; frame < frames; frame++)
            for (int channel = 0; channel < HIDDEN; channel++)
                x[(size_t)frame * HIDDEN + channel] +=
                    layer->attention_scale[channel] *
                    normed[(size_t)frame * HIDDEN + channel];

        rms_norm(x, layer->post_attention_layernorm, normed, HIDDEN, frames);
        linear(layer->gate_proj, NULL, normed, gate, HIDDEN, FFN, frames);
        linear(layer->up_proj, NULL, normed, up, HIDDEN, FFN, frames);
        for (int index2 = 0; index2 < frames * FFN; index2++) {
            const float g = gate[index2];
            gate[index2] = g / (1.0f + expf(-g)) * up[index2];
        }
        linear(layer->down_proj, NULL, gate, normed, FFN, HIDDEN, frames);
        for (int frame = 0; frame < frames; frame++)
            for (int channel = 0; channel < HIDDEN; channel++)
                x[(size_t)frame * HIDDEN + channel] +=
                    layer->mlp_scale[channel] *
                    normed[(size_t)frame * HIDDEN + channel];
    }
    rms_norm(x, codec->final_norm, x, HIDDEN, frames);
}

static void convnext_block(const convnext *block, float *x, int channels,
                           int length, float *scratch) {
    /* depthwise, so each output channel reads only its own input channel */
    causal_conv(&block->dwconv, x, length, scratch, 1, channels);
    dispatch_apply((size_t)length, DISPATCH_APPLY_AUTO, ^(size_t slot) {
        const int position = (int)slot;
        float channel_values[LATENT];
        float projected[4 * LATENT];
        float mean = 0.0f;
        for (int channel = 0; channel < channels; channel++)
            mean += scratch[(size_t)channel * length + position];
        mean /= (float)channels;
        float variance = 0.0f;
        for (int channel = 0; channel < channels; channel++) {
            const float centred =
                scratch[(size_t)channel * length + position] - mean;
            channel_values[channel] = centred;
            variance += centred * centred;
        }
        variance /= (float)channels;
        const float inverse = 1.0f / sqrtf(variance + 1e-6f);
        for (int channel = 0; channel < channels; channel++)
            channel_values[channel] = channel_values[channel] * inverse *
                block->norm_weight[channel] + block->norm_bias[channel];

        for (int output = 0; output < 4 * channels; output++) {
            const float *w = block->pwconv1_weight + (size_t)output * channels;
            float total = block->pwconv1_bias[output];
            for (int channel = 0; channel < channels; channel++)
                total += w[channel] * channel_values[channel];
            /* exact GELU, matching nn.GELU's default */
            projected[output] = 0.5f * total *
                (1.0f + erff(total * 0.70710678118654752f));
        }
        for (int output = 0; output < channels; output++) {
            const float *w = block->pwconv2_weight + (size_t)output * 4 * channels;
            float total = block->pwconv2_bias[output];
            for (int index = 0; index < 4 * channels; index++)
                total += w[index] * projected[index];
            x[(size_t)output * length + position] +=
                block->gamma[output] * total;
        }
    });
}

/* ---- loading ------------------------------------------------------------ */
static int take(const qwen_weights *weights, const char *name, int ndim,
                const int64_t *shape, const float **target,
                char *error, size_t error_size) {
    *target = qwen_weights_f32(weights, name, ndim, shape, error, error_size);
    return *target != NULL;
}

static int take_conv(const qwen_weights *weights, const char *name,
                     conv1d *conv, int out_channels, int in_channels,
                     int kernel, int transposed, int with_bias,
                     char *error, size_t error_size) {
    char full[256];
    /* ConvTranspose1d stores [in, out, k]; Conv1d stores [out, in, k]. */
    const int64_t shape[3] = {
        transposed ? in_channels : out_channels,
        transposed ? out_channels : in_channels,
        kernel,
    };
    snprintf(full, sizeof(full), "%s.weight", name);
    if (!take(weights, full, 3, shape, &conv->weight, error, error_size))
        return 0;
    conv->bias = NULL;
    if (with_bias) {
        snprintf(full, sizeof(full), "%s.bias", name);
        const int64_t bias_shape[1] = {out_channels};
        if (!take(weights, full, 1, bias_shape, &conv->bias, error, error_size))
            return 0;
    }
    conv->out_channels = out_channels;
    conv->in_channels = in_channels;
    conv->kernel = kernel;
    return 1;
}

static int take_snake(const qwen_weights *weights, const char *name,
                      snake *parameters, int channels,
                      char *error, size_t error_size) {
    char full[256];
    const int64_t shape[1] = {channels};
    snprintf(full, sizeof(full), "%s.alpha", name);
    if (!take(weights, full, 1, shape, &parameters->alpha, error, error_size))
        return 0;
    snprintf(full, sizeof(full), "%s.beta", name);
    return take(weights, full, 1, shape, &parameters->beta, error, error_size);
}

qwen_codec *qwen_codec_load(const char *path, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    qwen_codec *codec = calloc(1, sizeof(*codec));
    if (!codec) {
        snprintf(error, error_size, "out of memory");
        return NULL;
    }
    codec->weights = qwen_weights_open(path, error, error_size);
    if (!codec->weights) {
        free(codec);
        return NULL;
    }
    const qwen_weights *w = codec->weights;
#define FAIL_OUT do { qwen_codec_free(codec); return NULL; } while (0)
#define TAKE(name, ndim, target, ...) do {                                     \
        const int64_t shape[] = __VA_ARGS__;                                   \
        if (!take(w, (name), (ndim), shape, (target), error, error_size))      \
            FAIL_OUT;                                                          \
    } while (0)

    {
        char name[256];
        const int64_t codebook[] = {CODEBOOK_SIZE, CODEBOOK_DIM};
        snprintf(name, sizeof(name),
                 "quantizer.rvq_first.vq.layers.0._codebook.embedding");
        if (!take(w, name, 2, codebook, &codec->semantic_codebook,
                  error, error_size)) FAIL_OUT;
        for (int index = 0; index < QWEN_CODEC_GROUPS - 1; index++) {
            snprintf(name, sizeof(name),
                     "quantizer.rvq_rest.vq.layers.%d._codebook.embedding", index);
            if (!take(w, name, 2, codebook, &codec->acoustic_codebooks[index],
                      error, error_size)) FAIL_OUT;
        }
    }
    TAKE("quantizer.rvq_first.output_proj.weight", 3, &codec->semantic_output_proj, {QUANT_WIDTH, CODEBOOK_DIM, 1});
    TAKE("quantizer.rvq_rest.output_proj.weight", 3, &codec->acoustic_output_proj, {QUANT_WIDTH, CODEBOOK_DIM, 1});

    if (!take_conv(w, "pre_conv.conv", &codec->pre_conv, LATENT, QUANT_WIDTH,
                   3, 0, 1, error, error_size)) FAIL_OUT;

    TAKE("pre_transformer.input_proj.weight", 2, &codec->input_proj_weight, {HIDDEN, LATENT});
    TAKE("pre_transformer.input_proj.bias", 1, &codec->input_proj_bias, {HIDDEN});
    TAKE("pre_transformer.output_proj.weight", 2, &codec->output_proj_weight, {LATENT, HIDDEN});
    TAKE("pre_transformer.output_proj.bias", 1, &codec->output_proj_bias, {LATENT});
    TAKE("pre_transformer.norm.weight", 1, &codec->final_norm, {HIDDEN});

    for (int index = 0; index < LAYERS; index++) {
        codec_layer *layer = &codec->layers[index];
        char name[256];
#define LAYER_TAKE(suffix, ndim, target, ...) do {                             \
        const int64_t shape[] = __VA_ARGS__;                                   \
        snprintf(name, sizeof(name), "pre_transformer.layers.%d." suffix,      \
                 index);                                                       \
        if (!take(w, name, (ndim), shape, (target), error, error_size))        \
            FAIL_OUT;                                                          \
    } while (0)
        LAYER_TAKE("input_layernorm.weight", 1, &layer->input_layernorm, {HIDDEN});
        LAYER_TAKE("post_attention_layernorm.weight", 1, &layer->post_attention_layernorm, {HIDDEN});
        LAYER_TAKE("self_attn.q_proj.weight", 2, &layer->q_proj, {ATTENTION, HIDDEN});
        LAYER_TAKE("self_attn.k_proj.weight", 2, &layer->k_proj, {ATTENTION, HIDDEN});
        LAYER_TAKE("self_attn.v_proj.weight", 2, &layer->v_proj, {ATTENTION, HIDDEN});
        LAYER_TAKE("self_attn.o_proj.weight", 2, &layer->o_proj, {HIDDEN, ATTENTION});
        LAYER_TAKE("mlp.gate_proj.weight", 2, &layer->gate_proj, {FFN, HIDDEN});
        LAYER_TAKE("mlp.up_proj.weight", 2, &layer->up_proj, {FFN, HIDDEN});
        LAYER_TAKE("mlp.down_proj.weight", 2, &layer->down_proj, {HIDDEN, FFN});
        LAYER_TAKE("self_attn_layer_scale.scale", 1, &layer->attention_scale, {HIDDEN});
        LAYER_TAKE("mlp_layer_scale.scale", 1, &layer->mlp_scale, {HIDDEN});
#undef LAYER_TAKE
    }

    for (int index = 0; index < 2; index++) {
        char name[256];
        snprintf(name, sizeof(name), "upsample.%d.0.conv", index);
        if (!take_conv(w, name, &codec->upsample_conv[index], LATENT, LATENT,
                       2, 1, 1, error, error_size)) FAIL_OUT;
        convnext *block = &codec->upsample_next[index];
        snprintf(name, sizeof(name), "upsample.%d.1.dwconv.conv", index);
        if (!take_conv(w, name, &block->dwconv, LATENT, 1, 7, 0, 1,
                       error, error_size)) FAIL_OUT;
        block->dwconv.in_channels = LATENT;   /* depthwise: groups == channels */
#define NEXT_TAKE(suffix, ndim, target, ...) do {                              \
        const int64_t shape[] = __VA_ARGS__;                                   \
        snprintf(name, sizeof(name), "upsample.%d.1." suffix, index);          \
        if (!take(w, name, (ndim), shape, (target), error, error_size))        \
            FAIL_OUT;                                                          \
    } while (0)
        NEXT_TAKE("norm.weight", 1, &block->norm_weight, {LATENT});
        NEXT_TAKE("norm.bias", 1, &block->norm_bias, {LATENT});
        NEXT_TAKE("pwconv1.weight", 2, &block->pwconv1_weight, {4 * LATENT, LATENT});
        NEXT_TAKE("pwconv1.bias", 1, &block->pwconv1_bias, {4 * LATENT});
        NEXT_TAKE("pwconv2.weight", 2, &block->pwconv2_weight, {LATENT, 4 * LATENT});
        NEXT_TAKE("pwconv2.bias", 1, &block->pwconv2_bias, {LATENT});
        NEXT_TAKE("gamma", 1, &block->gamma, {LATENT});
#undef NEXT_TAKE
    }

    if (!take_conv(w, "decoder.0.conv", &codec->head_conv, DECODER_DIM, LATENT,
                   7, 0, 1, error, error_size)) FAIL_OUT;

    int channels = DECODER_DIM;
    for (int index = 0; index < STAGES; index++) {
        vocoder_stage *stage = &codec->stages[index];
        const int out_channels = channels / 2;
        const int rate = UPSAMPLE_RATES[index];
        char name[256];
        snprintf(name, sizeof(name), "decoder.%d.block.0", index + 1);
        if (!take_snake(w, name, &stage->act, channels, error, error_size))
            FAIL_OUT;
        snprintf(name, sizeof(name), "decoder.%d.block.1.conv", index + 1);
        if (!take_conv(w, name, &stage->up, out_channels, channels, 2 * rate,
                       1, 1, error, error_size)) FAIL_OUT;
        for (int unit = 0; unit < 3; unit++) {
            residual_unit *residual = &stage->units[unit];
            snprintf(name, sizeof(name), "decoder.%d.block.%d.act1",
                     index + 1, unit + 2);
            if (!take_snake(w, name, &residual->act1, out_channels,
                            error, error_size)) FAIL_OUT;
            snprintf(name, sizeof(name), "decoder.%d.block.%d.act2",
                     index + 1, unit + 2);
            if (!take_snake(w, name, &residual->act2, out_channels,
                            error, error_size)) FAIL_OUT;
            snprintf(name, sizeof(name), "decoder.%d.block.%d.conv1.conv",
                     index + 1, unit + 2);
            if (!take_conv(w, name, &residual->conv1, out_channels,
                           out_channels, 7, 0, 1, error, error_size)) FAIL_OUT;
            snprintf(name, sizeof(name), "decoder.%d.block.%d.conv2.conv",
                     index + 1, unit + 2);
            if (!take_conv(w, name, &residual->conv2, out_channels,
                           out_channels, 1, 0, 1, error, error_size)) FAIL_OUT;
        }
        channels = out_channels;
    }
    if (!take_snake(w, "decoder.5", &codec->final_act, channels,
                    error, error_size)) FAIL_OUT;
    if (!take_conv(w, "decoder.6.conv", &codec->output_conv, 1, channels, 7, 0,
                   1, error, error_size)) FAIL_OUT;
#undef TAKE
#undef FAIL_OUT
    return codec;
}

void qwen_codec_free(qwen_codec *codec) {
    if (!codec) return;
    qwen_weights_close(codec->weights);
    free(codec);
}

void qwen_codec_set_probe(qwen_codec *codec, qwen_codec_probe probe,
                          void *opaque) {
    if (!codec) return;
    codec->probe = probe;
    codec->probe_opaque = opaque;
}

int qwen_codec_decode(qwen_codec *codec, const uint32_t *codes, int frames,
                      float *audio, qwen_codec_progress progress, void *opaque,
                      char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!codec || !codes || !audio || frames < 1) {
        snprintf(error, error_size, "invalid arguments");
        return 0;
    }

    /* Two buffers big enough for the widest stage, used in turn: the vocoder
     * peaks at 96 channels x 1920 samples per frame. */
    const size_t peak = (size_t)frames * 96 * QWEN_CODEC_UPSAMPLE;
    float *front = calloc(peak, sizeof(float));
    float *back = calloc(peak, sizeof(float));
    float *scratch = calloc(peak, sizeof(float));
    float *rows = calloc((size_t)frames * ATTENTION * 6, sizeof(float));
    if (!front || !back || !scratch || !rows) {
        free(front); free(back); free(scratch); free(rows);
        snprintf(error, error_size, "cannot allocate %.1f MB for the decode",
                 peak * 3 * sizeof(float) / 1e6);
        return 0;
    }
#define RELEASE do { free(front); free(back); free(scratch); free(rows); } while (0)
    int completed = 0;
    const int total = 4 + STAGES;

    /* 1. dequantise: sixteen lookups, semantic and acoustic projected apart */
    float *quantised = back;
    memset(quantised, 0, (size_t)QUANT_WIDTH * frames * sizeof(float));
    for (int group = 0; group < QWEN_CODEC_GROUPS; group++) {
        const float *codebook = group == 0 ? codec->semantic_codebook
                                           : codec->acoustic_codebooks[group - 1];
        const float *projection = group == 0 ? codec->semantic_output_proj
                                             : codec->acoustic_output_proj;
        const uint32_t *row = codes + (size_t)group * frames;
        for (int frame = 0; frame < frames; frame++) {
            if (row[frame] >= CODEBOOK_SIZE) {
                RELEASE;
                snprintf(error, error_size, "code %u in group %d is out of range",
                         row[frame], group);
                return 0;
            }
        }
        /* residual quantisation sums the groups, so accumulate in place */
        dispatch_apply((size_t)QUANT_WIDTH, DISPATCH_APPLY_AUTO, ^(size_t slot) {
            const int output = (int)slot;
            const float *w = projection + (size_t)output * CODEBOOK_DIM;
            float *target = quantised + (size_t)output * frames;
            for (int frame = 0; frame < frames; frame++) {
                const float *entry = codebook + (size_t)row[frame] * CODEBOOK_DIM;
                float sum = 0.0f;
                for (int index = 0; index < CODEBOOK_DIM; index++)
                    sum += w[index] * entry[index];
                target[frame] += sum;
            }
        });
    }
    if (progress) progress(++completed, total, opaque);

    /* 2. pre_conv, 512 -> 1024 */
    causal_conv(&codec->pre_conv, quantised, frames, front, 1, 1);
    report(codec, "dec_preconv", front, LATENT, frames);
    if (progress) progress(++completed, total, opaque);

    /* 3. the transformer works frame-major, the convolutions channel-major */
    float *sequence = rows;
    float *projected = rows + (size_t)frames * HIDDEN;
    for (int frame = 0; frame < frames; frame++)
        for (int channel = 0; channel < LATENT; channel++)
            projected[(size_t)frame * LATENT + channel] =
                front[(size_t)channel * frames + frame];
    linear(codec->input_proj_weight, codec->input_proj_bias, projected,
           sequence, LATENT, HIDDEN, frames);
    {
        float *normed = calloc((size_t)frames * HIDDEN, sizeof(float));
        float *query = calloc((size_t)frames * ATTENTION, sizeof(float));
        float *key = calloc((size_t)frames * ATTENTION, sizeof(float));
        float *value = calloc((size_t)frames * ATTENTION, sizeof(float));
        float *attention = calloc((size_t)frames * ATTENTION, sizeof(float));
        float *gate = calloc((size_t)frames * FFN, sizeof(float));
        float *up = calloc((size_t)frames * FFN, sizeof(float));
        if (!normed || !query || !key || !value || !attention || !gate || !up) {
            free(normed); free(query); free(key); free(value);
            free(attention); free(gate); free(up);
            RELEASE;
            snprintf(error, error_size, "out of memory in the transformer");
            return 0;
        }
        transformer(codec, sequence, frames, normed, query, key, value,
                    attention, gate, up);
        free(normed); free(query); free(key); free(value);
        free(attention); free(gate); free(up);
    }
    linear(codec->output_proj_weight, codec->output_proj_bias, sequence,
           projected, HIDDEN, LATENT, frames);
    report(codec, "dec_pretransformer", projected, frames, LATENT);
    for (int frame = 0; frame < frames; frame++)
        for (int channel = 0; channel < LATENT; channel++)
            front[(size_t)channel * frames + frame] =
                projected[(size_t)frame * LATENT + channel];
    if (progress) progress(++completed, total, opaque);

    /* 4. two ConvNeXt upsample stages, x2 each */
    int length = frames;
    for (int index = 0; index < 2; index++) {
        causal_trans_conv(&codec->upsample_conv[index], front, length, back, 2);
        length *= 2;
        memcpy(front, back, (size_t)LATENT * length * sizeof(float));
        convnext_block(&codec->upsample_next[index], front, LATENT, length, back);
    }
    report(codec, "dec_upsampled", front, LATENT, length);
    if (progress) progress(++completed, total, opaque);

    /* 5. the vocoder */
    causal_conv(&codec->head_conv, front, length, back, 1, 1);
    memcpy(front, back, (size_t)DECODER_DIM * length * sizeof(float));
    report(codec, "dec_block0", front, DECODER_DIM, length);

    int channels = DECODER_DIM;
    for (int index = 0; index < STAGES; index++) {
        const vocoder_stage *stage = &codec->stages[index];
        const int rate = UPSAMPLE_RATES[index];
        snake_beta(&stage->act, front, channels, length);
        causal_trans_conv(&stage->up, front, length, back, rate);
        length *= rate;
        channels /= 2;
        memcpy(front, back, (size_t)channels * length * sizeof(float));

        /* front carries the residual throughout; back and scratch alternate
         * as the branch, so nothing writes over the value being added to. */
        for (int unit = 0; unit < 3; unit++) {
            const residual_unit *residual = &stage->units[unit];
            const size_t count = (size_t)channels * length;
            memcpy(back, front, count * sizeof(float));
            snake_beta(&residual->act1, back, channels, length);
            causal_conv(&residual->conv1, back, length, scratch,
                        DILATIONS[unit], 1);
            snake_beta(&residual->act2, scratch, channels, length);
            causal_conv(&residual->conv2, scratch, length, back, 1, 1);
            for (size_t sample = 0; sample < count; sample++)
                front[sample] += back[sample];
        }

        char stage_name[32];
        snprintf(stage_name, sizeof(stage_name), "dec_block%d", index + 1);
        report(codec, stage_name, front, channels, length);
        if (progress) progress(++completed, total, opaque);
    }

    snake_beta(&codec->final_act, front, channels, length);
    report(codec, "dec_block5", front, channels, length);
    causal_conv(&codec->output_conv, front, length, back, 1, 1);
    report(codec, "dec_block6", back, 1, length);

    for (int index = 0; index < length; index++) {
        float sample = back[index];
        if (sample < -1.0f) sample = -1.0f;
        if (sample > 1.0f) sample = 1.0f;
        audio[index] = sample;
    }

    RELEASE;
#undef RELEASE
    return 1;
}
