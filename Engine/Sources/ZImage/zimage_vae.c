#include "zimage_vae.h"
/* The Z-Image VAE decoder, sixteen latent channels up to RGB.
 *
 * f32 throughout, and checked at f32: this is a convolution stack, not a
 * transformer carrying bf16 weights, so "within tolerance" is not the claim —
 * it should agree to rounding the way the speech codec's vocoder did.
 *
 * Channel-major [channels][height][width], which is what the CPU codec uses
 * and what the loop order below wants. The engine's own convolutions are
 * time-major and will need the transpose the vocoder needed; that is a
 * question for the GPU port, not for whether the arithmetic is right.
 *
 * The decode sequence is exposed once and driven two ways: the check
 * harness passes a tap that compares each stage against the reference,
 * and the generator passes none. One implementation, so a fix to either
 * caller cannot drift from the other.
 */
#include "qwen_weights.h"

#include <dispatch/dispatch.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GROUPS   32
#define EPSILON  1e-6f

typedef struct { const float *weight, *bias; int out, in, kernel; } conv2d;
typedef struct { const float *weight, *bias; } norm2d;

typedef struct {
    norm2d norm1, norm2;
    conv2d conv1, conv2, shortcut;   /* shortcut.weight NULL when unchanged */
} resnet;

static qwen_weights *model;     /* the decoder */
static char problem[512];

/* The release stores the whole autoencoder in bf16 even though its config asks
 * for f32 arithmetic — `force_upcast` — so widen once at load and compute in
 * f32 throughout, which is what the reference does too. */
/* The release stores the whole autoencoder in bf16 even though its config asks
 * for f32 arithmetic — `force_upcast` — so widen once at load and compute in
 * f32 throughout, which is what the reference does too.
 *
 * Shapes are stated in full rather than left as wildcards. qwen_weights_bf16
 * takes them as the expectation to check, not as an out-parameter, so a
 * wildcard gives back no size to widen — and asserting the real shape is what
 * turns a changed layout into a message instead of arithmetic on the wrong
 * bytes. */
static const float *widen(const char *name, int ndim, const int64_t *shape,
                          int required) {
    const uint16_t *raw = qwen_weights_bf16(model, name, ndim, shape,
                                            problem, sizeof(problem));
    if (!raw) {
        if (!required) return NULL;
        fprintf(stderr, "%s\n", problem);
        exit(1);
    }
    size_t count = 1;
    for (int index = 0; index < ndim; index++) count *= (size_t)shape[index];
    float *values = malloc(count * sizeof(float));
    if (!values) { fprintf(stderr, "out of memory widening %s\n", name); exit(1); }
    for (size_t index = 0; index < count; index++) {
        union { uint32_t bits; float number; } cast;
        cast.bits = (uint32_t)raw[index] << 16;
        values[index] = cast.number;
    }
    return values;
}

static void format(char *buffer, size_t size, const char *pattern,
                   int a, int b) {
    if (b >= 0)      snprintf(buffer, size, pattern, a, b);
    else if (a >= 0) snprintf(buffer, size, pattern, a);
    else             snprintf(buffer, size, "%s", pattern);
}

/* [out, in, k, k] */
static conv2d take_conv(const char *pattern, int a, int b,
                        int out, int in, int kernel, int required) {
    char name[256];
    format(name, sizeof(name), pattern, a, b);
    char full[280];
    snprintf(full, sizeof(full), "%s.weight", name);
    const int64_t weight_shape[4] = {out, in, kernel, kernel};
    const float *weight = widen(full, 4, weight_shape, required);
    if (!weight) return (conv2d){NULL, NULL, 0, 0, 0};
    snprintf(full, sizeof(full), "%s.bias", name);
    const int64_t bias_shape[1] = {out};
    return (conv2d){weight, widen(full, 1, bias_shape, 1), out, in, kernel};
}

/* weight and bias, both [channels] */
static norm2d take_norm(const char *pattern, int a, int b, int channels) {
    char name[256];
    format(name, sizeof(name), pattern, a, b);
    char full[280];
    const int64_t shape[1] = {channels};
    snprintf(full, sizeof(full), "%s.weight", name);
    const float *weight = widen(full, 1, shape, 1);
    snprintf(full, sizeof(full), "%s.bias", name);
    return (norm2d){weight, widen(full, 1, shape, 1)};
}

/* [out, in] for the attention projections */
static const float *take_linear(const char *name, int out, int in) {
    const int64_t shape[2] = {out, in};
    return widen(name, 2, shape, 1);
}

static const float *take_vector(const char *name, int count) {
    const int64_t shape[1] = {count};
    return widen(name, 1, shape, 1);
}

/* out[oc][y][x] = bias + sum over input channels and the kernel window.
 *
 * The channel loop is outermost so each thread owns one output plane and its
 * accumulation stays in cache; the input planes are read repeatedly but they
 * are the smaller side at every stage here. */
static void convolve(const conv2d *conv, const float *in, float *out,
                     int height, int width) {
    const int pad = conv->kernel / 2;
    dispatch_apply((size_t)conv->out, DISPATCH_APPLY_AUTO, ^(size_t slot) {
        const int oc = (int)slot;
        float *plane = out + (size_t)oc * height * width;
        const float bias = conv->bias ? conv->bias[oc] : 0.0f;
        for (int index = 0; index < height * width; index++) plane[index] = bias;
        for (int ic = 0; ic < conv->in; ic++) {
            const float *source = in + (size_t)ic * height * width;
            const float *weights = conv->weight +
                ((size_t)oc * conv->in + ic) * conv->kernel * conv->kernel;
            for (int ky = 0; ky < conv->kernel; ky++)
                for (int kx = 0; kx < conv->kernel; kx++) {
                    const float weight = weights[ky * conv->kernel + kx];
                    if (weight == 0.0f) continue;
                    for (int y = 0; y < height; y++) {
                        const int sy = y + ky - pad;
                        if (sy < 0 || sy >= height) continue;
                        float *row = plane + (size_t)y * width;
                        const float *from = source + (size_t)sy * width;
                        for (int x = 0; x < width; x++) {
                            const int sx = x + kx - pad;
                            if (sx < 0 || sx >= width) continue;
                            row[x] += weight * from[sx];
                        }
                    }
                }
        }
    });
}

/* GroupNorm over 32 groups of channels, then optionally SiLU. */
static void group_norm(const norm2d *norm, const float *in, float *out,
                       int channels, int height, int width, int activate) {
    const int per_group = channels / GROUPS;
    const size_t plane = (size_t)height * width;
    dispatch_apply(GROUPS, DISPATCH_APPLY_AUTO, ^(size_t slot) {
        const int group = (int)slot;
        const size_t count = (size_t)per_group * plane;
        const float *base = in + (size_t)group * per_group * plane;
        double sum = 0.0;
        for (size_t index = 0; index < count; index++) sum += base[index];
        const double mean = sum / (double)count;
        double square = 0.0;
        for (size_t index = 0; index < count; index++) {
            const double centred = base[index] - mean;
            square += centred * centred;
        }
        const float inverse = (float)(1.0 / sqrt(square / (double)count + EPSILON));
        for (int channel = 0; channel < per_group; channel++) {
            const int absolute = group * per_group + channel;
            const float scale = norm->weight[absolute] * inverse;
            const float offset = norm->bias[absolute] - (float)mean * scale;
            const float *from = base + (size_t)channel * plane;
            float *to = out + (size_t)absolute * plane;
            for (size_t index = 0; index < plane; index++) {
                float value = from[index] * scale + offset;
                if (activate) value = value / (1.0f + expf(-value));
                to[index] = value;
            }
        }
    });
}

static void resnet_forward(const resnet *block, float *x, int in_channels,
                           int out_channels, int height, int width,
                           float *scratch, float *branch) {
    const size_t plane = (size_t)height * width;
    group_norm(&block->norm1, x, scratch, in_channels, height, width, 1);
    convolve(&block->conv1, scratch, branch, height, width);
    group_norm(&block->norm2, branch, scratch, out_channels, height, width, 1);
    convolve(&block->conv2, scratch, branch, height, width);

    if (block->shortcut.weight) {
        convolve(&block->shortcut, x, scratch, height, width);
        memcpy(x, scratch, (size_t)out_channels * plane * sizeof(float));
    }
    for (size_t index = 0; index < (size_t)out_channels * plane; index++)
        x[index] += branch[index];
}

/* The mid block's single-head attention over every spatial position. */
static void attention_forward(const norm2d *norm, const float *wq,
                              const float *bq, const float *wk, const float *bk,
                              const float *wv, const float *bv,
                              const float *wo, const float *bo,
                              float *x, int channels, int height, int width,
                              float *normed) {
    const int positions = height * width;
    group_norm(norm, x, normed, channels, height, width, 0);

    float *q = malloc((size_t)positions * channels * sizeof(float));
    float *k = malloc((size_t)positions * channels * sizeof(float));
    float *v = malloc((size_t)positions * channels * sizeof(float));
    float *out = malloc((size_t)positions * channels * sizeof(float));
    if (!q || !k || !v || !out) { fprintf(stderr, "out of memory\n"); exit(1); }

    /* The projections are linear over channels, so gather each position's
     * channel vector out of the planes first. */
    dispatch_apply((size_t)positions, DISPATCH_APPLY_AUTO, ^(size_t slot) {
        const int position = (int)slot;
        float *qq = q + (size_t)position * channels;
        float *kk = k + (size_t)position * channels;
        float *vv = v + (size_t)position * channels;
        for (int out_channel = 0; out_channel < channels; out_channel++) {
            float sq = bq[out_channel], sk = bk[out_channel], sv = bv[out_channel];
            for (int in_channel = 0; in_channel < channels; in_channel++) {
                const float value =
                    normed[(size_t)in_channel * positions + position];
                sq += wq[(size_t)out_channel * channels + in_channel] * value;
                sk += wk[(size_t)out_channel * channels + in_channel] * value;
                sv += wv[(size_t)out_channel * channels + in_channel] * value;
            }
            qq[out_channel] = sq; kk[out_channel] = sk; vv[out_channel] = sv;
        }
    });

    const float scale = 1.0f / sqrtf((float)channels);
    dispatch_apply((size_t)positions, DISPATCH_APPLY_AUTO, ^(size_t slot) {
        const int position = (int)slot;
        const float *qq = q + (size_t)position * channels;
        float *scores = malloc((size_t)positions * sizeof(float));
        float highest = -INFINITY;
        for (int other = 0; other < positions; other++) {
            const float *kk = k + (size_t)other * channels;
            float total = 0.0f;
            for (int channel = 0; channel < channels; channel++)
                total += qq[channel] * kk[channel];
            total *= scale;
            scores[other] = total;
            if (total > highest) highest = total;
        }
        float sum = 0.0f;
        for (int other = 0; other < positions; other++) {
            scores[other] = expf(scores[other] - highest);
            sum += scores[other];
        }
        float *result = out + (size_t)position * channels;
        memset(result, 0, (size_t)channels * sizeof(float));
        for (int other = 0; other < positions; other++) {
            const float weight = scores[other] / sum;
            const float *vv = v + (size_t)other * channels;
            for (int channel = 0; channel < channels; channel++)
                result[channel] += weight * vv[channel];
        }
        free(scores);
    });

    /* Project out and add the residual, scattering back into planes. */
    dispatch_apply((size_t)channels, DISPATCH_APPLY_AUTO, ^(size_t slot) {
        const int out_channel = (int)slot;
        float *plane = x + (size_t)out_channel * positions;
        for (int position = 0; position < positions; position++) {
            const float *from = out + (size_t)position * channels;
            float total = bo[out_channel];
            for (int in_channel = 0; in_channel < channels; in_channel++)
                total += wo[(size_t)out_channel * channels + in_channel]
                       * from[in_channel];
            plane[position] += total;
        }
    });
    free(q); free(k); free(v); free(out);
}

/* Halving, which is what Downsample2D does: pad one column on the right and
 * one row at the bottom — never on the left or top — then a 3x3 stride-2
 * convolution with no padding of its own. The asymmetry is the whole point,
 * and getting it wrong shifts the picture half a pixel per stage. */
static void downsample_convolve(const conv2d *conv, const float *in, float *out,
                                int height, int width) {
    const int out_height = height / 2, out_width = width / 2;
    dispatch_apply((size_t)conv->out, DISPATCH_APPLY_AUTO, ^(size_t slot) {
        const int oc = (int)slot;
        float *plane = out + (size_t)oc * out_height * out_width;
        const float bias = conv->bias ? conv->bias[oc] : 0.0f;
        for (int index = 0; index < out_height * out_width; index++)
            plane[index] = bias;
        for (int ic = 0; ic < conv->in; ic++) {
            const float *source = in + (size_t)ic * height * width;
            const float *weights = conv->weight +
                ((size_t)oc * conv->in + ic) * conv->kernel * conv->kernel;
            for (int ky = 0; ky < conv->kernel; ky++)
                for (int kx = 0; kx < conv->kernel; kx++) {
                    const float weight = weights[ky * conv->kernel + kx];
                    if (weight == 0.0f) continue;
                    for (int y = 0; y < out_height; y++) {
                        const int sy = y * 2 + ky;
                        if (sy >= height) continue;
                        float *row = plane + (size_t)y * out_width;
                        const float *from = source + (size_t)sy * width;
                        for (int x = 0; x < out_width; x++) {
                            const int sx = x * 2 + kx;
                            if (sx >= width) continue;
                            row[x] += weight * from[sx];
                        }
                    }
                }
        }
    });
}

/* Nearest-neighbour doubling, which is what Upsample2D does before its conv. */
static void upsample(const float *in, float *out, int channels,
                     int height, int width) {
    dispatch_apply((size_t)channels, DISPATCH_APPLY_AUTO, ^(size_t slot) {
        const int channel = (int)slot;
        const float *from = in + (size_t)channel * height * width;
        float *to = out + (size_t)channel * height * width * 4;
        for (int y = 0; y < height * 2; y++)
            for (int x = 0; x < width * 2; x++)
                to[(size_t)y * width * 2 + x] = from[(size_t)(y / 2) * width + x / 2];
    });
}

static void emit(zimage_vae_tap tap, void *context, const char *stage,
                 const float *values, size_t count) {
    if (tap) tap(stage, values, count, context);
}

int zimage_vae_decode(qwen_weights *decoder, const float *latent,
                      int height_in, int width_in,
                      float *image, zimage_vae_tap tap, void *context) {
    model = decoder;
    const int channels[4] = {512, 512, 256, 128};   /* per up block, in order */
    const int final_height = height_in * 8, final_width = width_in * 8;

    /* The widest the stack ever gets, in floats. Walking the stages at side s:
     * conv_in and mid hold 512s^2; the up blocks hold 512*(2s)^2 and
     * 512*(4s)^2; then block 2 upsamples 256 channels to the full picture,
     * 256*(8s)^2, before block 3 halves the channels again to 128*(8s)^2. The
     * third upsample is therefore the peak — 256 channels already at full
     * size — and nothing later exceeds it. */
    const size_t peak = (size_t)256 * (size_t)final_height * (size_t)final_width;
    float *x = calloc(peak, sizeof(float));
    float *scratch = calloc(peak, sizeof(float));
    float *branch = calloc(peak, sizeof(float));
    float *spare = calloc(peak, sizeof(float));
    if (!x || !scratch || !branch || !spare) {
        free(x); free(scratch); free(branch); free(spare);
        return 0;
    }

    conv2d conv_in = take_conv("decoder.conv_in", -1, -1, 512, 16, 3, 1);
    convolve(&conv_in, latent, x, height_in, width_in);
    emit(tap, context, "conv_in", x, (size_t)512 * height_in * width_in);

    /* mid block: resnet, attention, resnet */
    for (int index = 0; index < 2; index++) {
        resnet block = {
            take_norm("decoder.mid_block.resnets.%d.norm1", index, -1, 512),
            take_norm("decoder.mid_block.resnets.%d.norm2", index, -1, 512),
            take_conv("decoder.mid_block.resnets.%d.conv1", index, -1, 512, 512, 3, 1),
            take_conv("decoder.mid_block.resnets.%d.conv2", index, -1, 512, 512, 3, 1),
            {NULL, NULL, 0, 0, 0},
        };
        resnet_forward(&block, x, 512, 512, height_in, width_in, scratch, branch);
        if (index == 0) {
            norm2d norm = take_norm("decoder.mid_block.attentions.0.group_norm",
                                    -1, -1, 512);
            attention_forward(
                &norm,
                take_linear("decoder.mid_block.attentions.0.to_q.weight", 512, 512),
                take_vector("decoder.mid_block.attentions.0.to_q.bias", 512),
                take_linear("decoder.mid_block.attentions.0.to_k.weight", 512, 512),
                take_vector("decoder.mid_block.attentions.0.to_k.bias", 512),
                take_linear("decoder.mid_block.attentions.0.to_v.weight", 512, 512),
                take_vector("decoder.mid_block.attentions.0.to_v.bias", 512),
                take_linear("decoder.mid_block.attentions.0.to_out.0.weight", 512, 512),
                take_vector("decoder.mid_block.attentions.0.to_out.0.bias", 512),
                x, 512, height_in, width_in, scratch);
        }
    }
    emit(tap, context, "mid_block", x, (size_t)512 * height_in * width_in);

    int current = 512, height = height_in, width = width_in;
    for (int block_index = 0; block_index < 4; block_index++) {
        const int out_channels = channels[block_index];
        for (int index = 0; index < 3; index++) {
            const int in_channels = index == 0 ? current : out_channels;
            resnet block = {
                take_norm("decoder.up_blocks.%d.resnets.%d.norm1",
                          block_index, index, in_channels),
                take_norm("decoder.up_blocks.%d.resnets.%d.norm2",
                          block_index, index, out_channels),
                take_conv("decoder.up_blocks.%d.resnets.%d.conv1",
                          block_index, index, out_channels, in_channels, 3, 1),
                take_conv("decoder.up_blocks.%d.resnets.%d.conv2",
                          block_index, index, out_channels, out_channels, 3, 1),
                /* only the two resnets that change channel count carry one */
                take_conv("decoder.up_blocks.%d.resnets.%d.conv_shortcut",
                          block_index, index, out_channels, in_channels, 1, 0),
            };
            resnet_forward(&block, x, in_channels, out_channels, height, width,
                           scratch, branch);
        }
        current = out_channels;

        if (block_index < 3) {
            upsample(x, spare, current, height, width);
            height *= 2; width *= 2;
            conv2d up = take_conv("decoder.up_blocks.%d.upsamplers.0.conv",
                                  block_index, -1, current, current, 3, 1);
            convolve(&up, spare, x, height, width);
        }
        char name[24];
        snprintf(name, sizeof(name), "up_block_%d", block_index);
        emit(tap, context, name, x, (size_t)current * height * width);
    }

    norm2d out_norm = take_norm("decoder.conv_norm_out", -1, -1, current);
    group_norm(&out_norm, x, scratch, current, height, width, 1);
    conv2d conv_out = take_conv("decoder.conv_out", -1, -1, 3, current, 3, 1);
    convolve(&conv_out, scratch, image, height, width);
    emit(tap, context, "image", image, (size_t)3 * height * width);

    (void)final_height; (void)final_width;
    free(x); free(scratch); free(branch); free(spare);
    return 1;
}

/* The encoder: RGB down to sixteen latent channels, an eighth of a side.
 *
 * The mirror of the decoder above, and deliberately built from the same
 * pieces — the same convolution, group norm, resnet and attention that were
 * checked stage by stage against the reference. What is new here is only the
 * order, the strided downsample, and the last step: `conv_out` produces
 * thirty-two channels, a mean and a log-variance interleaved as halves, and
 * this returns the mean.
 *
 * Sampling from that distribution is what the reference pipeline does, but
 * the variance a trained image autoencoder reports is vanishingly small — the
 * check harness prints it — so the mean is the same picture and img2img stays
 * reproducible from its seed alone.
 *
 * `image` is [3][image_side][image_side] roughly in [-1, 1]; `latent`
 * receives [16][image_side/8][image_side/8] raw, before the DiT's scale and
 * shift. */
int zimage_vae_encode(qwen_weights *encoder, const float *image,
                      int image_height, int image_width,
                      float *latent, zimage_vae_tap tap, void *context) {
    model = encoder;

    const int channels[4] = {128, 256, 512, 512};   /* per down block, in order */

    /* The stack is widest at the start: 128 channels at the full picture.
     * Every later stage doubles the channels only after quartering the area,
     * so 128*s^2 is the peak and nothing after it comes close. */
    const size_t peak = (size_t)128 * (size_t)image_height * (size_t)image_width;
    float *x = calloc(peak, sizeof(float));
    float *scratch = calloc(peak, sizeof(float));
    float *branch = calloc(peak, sizeof(float));
    float *spare = calloc(peak, sizeof(float));
    if (!x || !scratch || !branch || !spare) {
        free(x); free(scratch); free(branch); free(spare);
        return 0;
    }

    conv2d conv_in = take_conv("encoder.conv_in", -1, -1, 128, 3, 3, 1);
    convolve(&conv_in, image, x, image_height, image_width);
    emit(tap, context, "conv_in", x, (size_t)128 * image_height * image_width);

    int current = 128, height = image_height, width = image_width;
    for (int block_index = 0; block_index < 4; block_index++) {
        const int out_channels = channels[block_index];
        for (int index = 0; index < 2; index++) {
            const int in_channels = index == 0 ? current : out_channels;
            resnet block = {
                take_norm("encoder.down_blocks.%d.resnets.%d.norm1",
                          block_index, index, in_channels),
                take_norm("encoder.down_blocks.%d.resnets.%d.norm2",
                          block_index, index, out_channels),
                take_conv("encoder.down_blocks.%d.resnets.%d.conv1",
                          block_index, index, out_channels, in_channels, 3, 1),
                take_conv("encoder.down_blocks.%d.resnets.%d.conv2",
                          block_index, index, out_channels, out_channels, 3, 1),
                /* only the two resnets that change channel count carry one */
                take_conv("encoder.down_blocks.%d.resnets.%d.conv_shortcut",
                          block_index, index, out_channels, in_channels, 1, 0),
            };
            resnet_forward(&block, x, in_channels, out_channels, height, width,
                           scratch, branch);
        }
        current = out_channels;

        if (block_index < 3) {
            conv2d down = take_conv("encoder.down_blocks.%d.downsamplers.0.conv",
                                    block_index, -1, current, current, 3, 1);
            downsample_convolve(&down, x, spare, height, width);
            height /= 2; width /= 2;
            memcpy(x, spare, (size_t)current * height * width * sizeof(float));
        }
        char name[24];
        snprintf(name, sizeof(name), "down_block_%d", block_index);
        emit(tap, context, name, x, (size_t)current * height * width);
    }

    /* mid block: resnet, attention, resnet — the decoder's, in the same order */
    for (int index = 0; index < 2; index++) {
        resnet block = {
            take_norm("encoder.mid_block.resnets.%d.norm1", index, -1, 512),
            take_norm("encoder.mid_block.resnets.%d.norm2", index, -1, 512),
            take_conv("encoder.mid_block.resnets.%d.conv1", index, -1, 512, 512, 3, 1),
            take_conv("encoder.mid_block.resnets.%d.conv2", index, -1, 512, 512, 3, 1),
            {NULL, NULL, 0, 0, 0},
        };
        resnet_forward(&block, x, 512, 512, height, width, scratch, branch);
        if (index == 0) {
            norm2d norm = take_norm("encoder.mid_block.attentions.0.group_norm",
                                    -1, -1, 512);
            attention_forward(
                &norm,
                take_linear("encoder.mid_block.attentions.0.to_q.weight", 512, 512),
                take_vector("encoder.mid_block.attentions.0.to_q.bias", 512),
                take_linear("encoder.mid_block.attentions.0.to_k.weight", 512, 512),
                take_vector("encoder.mid_block.attentions.0.to_k.bias", 512),
                take_linear("encoder.mid_block.attentions.0.to_v.weight", 512, 512),
                take_vector("encoder.mid_block.attentions.0.to_v.bias", 512),
                take_linear("encoder.mid_block.attentions.0.to_out.0.weight", 512, 512),
                take_vector("encoder.mid_block.attentions.0.to_out.0.bias", 512),
                x, 512, height, width, scratch);
        }
    }
    emit(tap, context, "mid_block", x, (size_t)512 * height * width);

    norm2d out_norm = take_norm("encoder.conv_norm_out", -1, -1, 512);
    group_norm(&out_norm, x, scratch, 512, height, width, 1);
    conv2d conv_out = take_conv("encoder.conv_out", -1, -1, 32, 512, 3, 1);
    convolve(&conv_out, scratch, spare, height, width);
    emit(tap, context, "moments", spare, (size_t)32 * height * width);

    /* The first sixteen channels are the mean; the rest are the log-variance,
     * which the tap above still carries for anyone who wants to look. */
    const size_t plane = (size_t)height * width;
    memcpy(latent, spare, (size_t)ZIMAGE_VAE_LATENT_CHANNELS * plane * sizeof(float));
    emit(tap, context, "latent", latent,
         (size_t)ZIMAGE_VAE_LATENT_CHANNELS * plane);

    free(x); free(scratch); free(branch); free(spare);
    return 1;
}
