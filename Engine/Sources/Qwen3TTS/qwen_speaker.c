#include "qwen_speaker.h"

#include "qwen_weights.h"

#include <dispatch/dispatch.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MEL_BANDS   128
#define N_FFT       1024
#define HOP         256
#define BINS        (N_FFT / 2 + 1)
#define FMIN        0.0f
#define FMAX        12000.0f

#define CHANNELS    512
#define BLOCKS      4        /* one plain TDNN then three SE-Res2Net */
#define AGGREGATE   1536     /* three blocks concatenated */
#define SCALE       8        /* res2net split */
#define SE_CHANNELS 128
#define ATTENTION   128

static const int KERNELS[BLOCKS] = {5, 3, 3, 3};
static const int DILATIONS[BLOCKS] = {1, 2, 3, 4};

typedef struct {
    const float *weight, *bias;
    int out_channels, in_channels, kernel, dilation;
} tdnn;

typedef struct {
    tdnn tdnn1, tdnn2;
    tdnn res2net[SCALE - 1];
    const float *se1_weight, *se1_bias;
    const float *se2_weight, *se2_bias;
} se_res2net;

struct qwen_speaker {
    qwen_weights *weights;
    tdnn first;
    se_res2net blocks[BLOCKS - 1];
    tdnn mfa;
    tdnn asp_tdnn;
    const float *asp_conv_weight, *asp_conv_bias;
    const float *fc_weight, *fc_bias;
};

/* ---- mel ---------------------------------------------------------------- */

/* Slaney's mel scale: linear to 1 kHz, logarithmic above. */
static float hz_to_mel(float hz) {
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;
    const float logstep = logf(6.4f) / 27.0f;
    if (hz >= min_log_hz)
        return min_log_mel + logf(hz / min_log_hz) / logstep;
    return hz / f_sp;
}

static float mel_to_hz(float mel) {
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;
    const float logstep = logf(6.4f) / 27.0f;
    if (mel >= min_log_mel)
        return min_log_hz * expf(logstep * (mel - min_log_mel));
    return f_sp * mel;
}

/* Triangular filters, then slaney normalisation: each filter is scaled by
 * 2 / (upper - lower) so that filters cover equal energy rather than equal
 * height. Getting this wrong tilts the spectrum without breaking anything. */
static void mel_filterbank(float *weights) {
    float edges[MEL_BANDS + 2];
    const float low = hz_to_mel(FMIN), high = hz_to_mel(FMAX);
    for (int index = 0; index < MEL_BANDS + 2; index++)
        edges[index] = mel_to_hz(low + (high - low) * (float)index /
                                 (float)(MEL_BANDS + 1));

    memset(weights, 0, (size_t)MEL_BANDS * BINS * sizeof(float));
    for (int band = 0; band < MEL_BANDS; band++) {
        const float lower_step = edges[band + 1] - edges[band];
        const float upper_step = edges[band + 2] - edges[band + 1];
        const float energy = 2.0f / (edges[band + 2] - edges[band]);
        for (int bin = 0; bin < BINS; bin++) {
            const float frequency = 0.5f * QWEN_SPEAKER_SAMPLE_RATE *
                (float)bin / (float)(BINS - 1);
            const float lower = (frequency - edges[band]) / lower_step;
            const float upper = (edges[band + 2] - frequency) / upper_step;
            float value = lower < upper ? lower : upper;
            if (value < 0.0f) value = 0.0f;
            weights[(size_t)band * BINS + bin] = value * energy;
        }
    }
}

/* Iterative radix-2 Cooley-Tukey, in place. N_FFT is 1024, so no general
 * factorisation is needed and pulling in a library would cost more than it
 * saves — this runs once per generation, not once per frame.
 *
 * The twiddle factors come from a table rather than from repeated complex
 * multiplication of a step factor. Stepping is tempting and wrong: in float32
 * the product drifts in both magnitude and phase over the 512 iterations of
 * the last stage, which showed up here as a 1.9% error in the mel — small
 * enough to leave the speaker embedding looking correct, large enough to be
 * worth not shipping. Each entry is computed in double and rounded once. */
static void fill_twiddles(float *cosines, float *sines) {
    for (int index = 0; index < N_FFT / 2; index++) {
        const double angle = -2.0 * M_PI * (double)index / (double)N_FFT;
        cosines[index] = (float)cos(angle);
        sines[index] = (float)sin(angle);
    }
}

static void fft(float *real, float *imaginary, const float *cosines,
                const float *sines) {
    for (int index = 1, target = 0; index < N_FFT; index++) {
        int bit = N_FFT >> 1;
        for (; target & bit; bit >>= 1) target ^= bit;
        target ^= bit;
        if (index < target) {
            float swap = real[index]; real[index] = real[target]; real[target] = swap;
            swap = imaginary[index];
            imaginary[index] = imaginary[target];
            imaginary[target] = swap;
        }
    }
    for (int span = 2; span <= N_FFT; span <<= 1) {
        const int stride = N_FFT / span;
        for (int start = 0; start < N_FFT; start += span) {
            for (int offset = 0; offset < span / 2; offset++) {
                const int here = start + offset, there = here + span / 2;
                const float factor_real = cosines[offset * stride];
                const float factor_imaginary = sines[offset * stride];
                const float product_real =
                    real[there] * factor_real - imaginary[there] * factor_imaginary;
                const float product_imaginary =
                    real[there] * factor_imaginary + imaginary[there] * factor_real;
                real[there] = real[here] - product_real;
                imaginary[there] = imaginary[here] - product_imaginary;
                real[here] += product_real;
                imaginary[here] += product_imaginary;
            }
        }
    }
}

int qwen_speaker_mel(const float *audio, int samples, float **mel,
                     char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!audio || !mel || samples < N_FFT) {
        snprintf(error, error_size, "the reference clip is too short");
        return 0;
    }
    /* centre=False with the signal reflect-padded by (n_fft - hop) / 2 */
    const int padding = (N_FFT - HOP) / 2;
    const int padded_length = samples + 2 * padding;
    float *padded = malloc((size_t)padded_length * sizeof(float));
    if (!padded) {
        snprintf(error, error_size, "out of memory");
        return 0;
    }
    for (int index = 0; index < padded_length; index++) {
        int source = index - padding;
        if (source < 0) source = -source;                       /* reflect */
        if (source >= samples) source = 2 * (samples - 1) - source;
        padded[index] = audio[source];
    }

    const int frames = (padded_length - N_FFT) / HOP + 1;
    float *bank = malloc((size_t)MEL_BANDS * BINS * sizeof(float));
    float *output = malloc((size_t)MEL_BANDS * frames * sizeof(float));
    float *window = malloc(N_FFT * sizeof(float));
    if (!bank || !output || !window) {
        free(padded); free(bank); free(output); free(window);
        snprintf(error, error_size, "out of memory");
        return 0;
    }
    mel_filterbank(bank);
    for (int index = 0; index < N_FFT; index++)   /* periodic Hann */
        window[index] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * (double)index /
                                                (double)N_FFT));
    /* heap, not stack: a block cannot capture a C array by value */
    float *cosines = malloc((N_FFT / 2) * sizeof(float));
    float *sines = malloc((N_FFT / 2) * sizeof(float));
    if (!cosines || !sines) {
        free(padded); free(bank); free(output); free(window);
        free(cosines); free(sines);
        snprintf(error, error_size, "out of memory");
        return 0;
    }
    fill_twiddles(cosines, sines);

    dispatch_apply((size_t)frames, DISPATCH_APPLY_AUTO, ^(size_t slot) {
        const int frame = (int)slot;
        float real[N_FFT], imaginary[N_FFT], magnitude[BINS];
        for (int index = 0; index < N_FFT; index++) {
            real[index] = padded[(size_t)frame * HOP + index] * window[index];
            imaginary[index] = 0.0f;
        }
        fft(real, imaginary, cosines, sines);
        for (int bin = 0; bin < BINS; bin++)
            magnitude[bin] = sqrtf(real[bin] * real[bin] +
                                   imaginary[bin] * imaginary[bin] + 1e-9f);
        for (int band = 0; band < MEL_BANDS; band++) {
            const float *filter = bank + (size_t)band * BINS;
            float total = 0.0f;
            for (int bin = 0; bin < BINS; bin++)
                total += filter[bin] * magnitude[bin];
            if (total < 1e-5f) total = 1e-5f;
            output[(size_t)band * frames + frame] = logf(total);
        }
    });

    free(padded);
    free(bank);
    free(window);
    free(cosines);
    free(sines);
    *mel = output;
    return frames;
}

/* ---- ECAPA -------------------------------------------------------------- */

/* Conv with PyTorch's "same" padding and reflect mode, then ReLU when asked.
 * The extra sample of an odd total goes on the right. */
static void tdnn_forward(const tdnn *layer, const float *input, int length,
                         float *output, int activate) {
    const int total = layer->dilation * (layer->kernel - 1);
    const int left = total / 2;
    dispatch_apply((size_t)layer->out_channels, DISPATCH_APPLY_AUTO,
                   ^(size_t slot) {
        const int out_channel = (int)slot;
        const float *weights = layer->weight +
            (size_t)out_channel * layer->in_channels * layer->kernel;
        float *row = output + (size_t)out_channel * length;
        for (int position = 0; position < length; position++) {
            float sum = layer->bias ? layer->bias[out_channel] : 0.0f;
            for (int tap = 0; tap < layer->kernel; tap++) {
                int source = position - left + tap * layer->dilation;
                if (source < 0) source = -source;                  /* reflect */
                if (source >= length) source = 2 * (length - 1) - source;
                if (source < 0) source = 0;
                for (int channel = 0; channel < layer->in_channels; channel++)
                    sum += weights[(size_t)channel * layer->kernel + tap] *
                           input[(size_t)channel * length + source];
            }
            row[position] = activate && sum < 0.0f ? 0.0f : sum;
        }
    });
}

/* Res2Net: split into eight, pass the first through, and give each later
 * chunk the sum of itself and the previous chunk's output. */
static void res2net_forward(const se_res2net *block, const float *input,
                            int length, float *output) {
    const int width = CHANNELS / SCALE;
    memcpy(output, input, (size_t)width * length * sizeof(float));
    for (int index = 1; index < SCALE; index++) {
        const float *chunk = input + (size_t)index * width * length;
        float *target = output + (size_t)index * width * length;
        if (index == 1) {
            tdnn_forward(&block->res2net[0], chunk, length, target, 1);
        } else {
            float *sum = malloc((size_t)width * length * sizeof(float));
            const float *previous = output + (size_t)(index - 1) * width * length;
            for (size_t entry = 0; entry < (size_t)width * length; entry++)
                sum[entry] = chunk[entry] + previous[entry];
            tdnn_forward(&block->res2net[index - 1], sum, length, target, 1);
            free(sum);
        }
    }
}

static void se_forward(const se_res2net *block, float *values, int length) {
    float average[CHANNELS], hidden[SE_CHANNELS];
    for (int channel = 0; channel < CHANNELS; channel++) {
        float total = 0.0f;
        for (int position = 0; position < length; position++)
            total += values[(size_t)channel * length + position];
        average[channel] = total / (float)length;
    }
    for (int index = 0; index < SE_CHANNELS; index++) {
        float total = block->se1_bias[index];
        for (int channel = 0; channel < CHANNELS; channel++)
            total += block->se1_weight[(size_t)index * CHANNELS + channel] *
                     average[channel];
        hidden[index] = total > 0.0f ? total : 0.0f;
    }
    for (int channel = 0; channel < CHANNELS; channel++) {
        float total = block->se2_bias[channel];
        for (int index = 0; index < SE_CHANNELS; index++)
            total += block->se2_weight[(size_t)channel * SE_CHANNELS + index] *
                     hidden[index];
        const float gate = 1.0f / (1.0f + expf(-total));
        float *row = values + (size_t)channel * length;
        for (int position = 0; position < length; position++)
            row[position] *= gate;
    }
}

/* ---- loading ------------------------------------------------------------ */
static int take(const qwen_weights *weights, const char *name, int ndim,
                const int64_t *shape, const float **target,
                char *error, size_t error_size) {
    *target = qwen_weights_f32(weights, name, ndim, shape, error, error_size);
    return *target != NULL;
}

static int take_tdnn(const qwen_weights *weights, const char *name, tdnn *layer,
                     int out_channels, int in_channels, int kernel, int dilation,
                     char *error, size_t error_size) {
    char full[256];
    const int64_t shape[3] = {out_channels, in_channels, kernel};
    const int64_t bias_shape[1] = {out_channels};
    snprintf(full, sizeof(full), "%s.weight", name);
    if (!take(weights, full, 3, shape, &layer->weight, error, error_size))
        return 0;
    snprintf(full, sizeof(full), "%s.bias", name);
    if (!take(weights, full, 1, bias_shape, &layer->bias, error, error_size))
        return 0;
    layer->out_channels = out_channels;
    layer->in_channels = in_channels;
    layer->kernel = kernel;
    layer->dilation = dilation;
    return 1;
}

qwen_speaker *qwen_speaker_load(const char *path, char *error,
                                size_t error_size) {
    if (error && error_size) error[0] = '\0';
    qwen_speaker *speaker = calloc(1, sizeof(*speaker));
    if (!speaker) {
        snprintf(error, error_size, "out of memory");
        return NULL;
    }
    speaker->weights = qwen_weights_open(path, error, error_size);
    if (!speaker->weights) {
        free(speaker);
        return NULL;
    }
    const qwen_weights *w = speaker->weights;
#define FAIL_OUT do { qwen_speaker_free(speaker); return NULL; } while (0)

    if (!take_tdnn(w, "blocks.0.conv", &speaker->first, CHANNELS, MEL_BANDS,
                   KERNELS[0], DILATIONS[0], error, error_size)) FAIL_OUT;

    for (int index = 1; index < BLOCKS; index++) {
        se_res2net *block = &speaker->blocks[index - 1];
        char name[256];
        snprintf(name, sizeof(name), "blocks.%d.tdnn1.conv", index);
        if (!take_tdnn(w, name, &block->tdnn1, CHANNELS, CHANNELS, 1, 1,
                       error, error_size)) FAIL_OUT;
        snprintf(name, sizeof(name), "blocks.%d.tdnn2.conv", index);
        if (!take_tdnn(w, name, &block->tdnn2, CHANNELS, CHANNELS, 1, 1,
                       error, error_size)) FAIL_OUT;
        for (int part = 0; part < SCALE - 1; part++) {
            snprintf(name, sizeof(name), "blocks.%d.res2net_block.blocks.%d.conv",
                     index, part);
            if (!take_tdnn(w, name, &block->res2net[part], CHANNELS / SCALE,
                           CHANNELS / SCALE, KERNELS[index], DILATIONS[index],
                           error, error_size)) FAIL_OUT;
        }
        const int64_t se1[] = {SE_CHANNELS, CHANNELS, 1};
        const int64_t se1_bias[] = {SE_CHANNELS};
        const int64_t se2[] = {CHANNELS, SE_CHANNELS, 1};
        const int64_t se2_bias[] = {CHANNELS};
        snprintf(name, sizeof(name), "blocks.%d.se_block.conv1.weight", index);
        if (!take(w, name, 3, se1, &block->se1_weight, error, error_size)) FAIL_OUT;
        snprintf(name, sizeof(name), "blocks.%d.se_block.conv1.bias", index);
        if (!take(w, name, 1, se1_bias, &block->se1_bias, error, error_size)) FAIL_OUT;
        snprintf(name, sizeof(name), "blocks.%d.se_block.conv2.weight", index);
        if (!take(w, name, 3, se2, &block->se2_weight, error, error_size)) FAIL_OUT;
        snprintf(name, sizeof(name), "blocks.%d.se_block.conv2.bias", index);
        if (!take(w, name, 1, se2_bias, &block->se2_bias, error, error_size)) FAIL_OUT;
    }

    if (!take_tdnn(w, "mfa.conv", &speaker->mfa, AGGREGATE, AGGREGATE, 1, 1,
                   error, error_size)) FAIL_OUT;
    if (!take_tdnn(w, "asp.tdnn.conv", &speaker->asp_tdnn, ATTENTION,
                   AGGREGATE * 3, 1, 1, error, error_size)) FAIL_OUT;
    {
        const int64_t conv[] = {AGGREGATE, ATTENTION, 1};
        const int64_t conv_bias[] = {AGGREGATE};
        const int64_t fc[] = {QWEN_SPEAKER_DIM, AGGREGATE * 2, 1};
        const int64_t fc_bias[] = {QWEN_SPEAKER_DIM};
        if (!take(w, "asp.conv.weight", 3, conv, &speaker->asp_conv_weight,
                  error, error_size)) FAIL_OUT;
        if (!take(w, "asp.conv.bias", 1, conv_bias, &speaker->asp_conv_bias,
                  error, error_size)) FAIL_OUT;
        if (!take(w, "fc.weight", 3, fc, &speaker->fc_weight, error, error_size))
            FAIL_OUT;
        if (!take(w, "fc.bias", 1, fc_bias, &speaker->fc_bias, error, error_size))
            FAIL_OUT;
    }
#undef FAIL_OUT
    return speaker;
}

void qwen_speaker_free(qwen_speaker *speaker) {
    if (!speaker) return;
    qwen_weights_close(speaker->weights);
    free(speaker);
}

int qwen_speaker_embed(qwen_speaker *speaker, const float *audio, int samples,
                       float *embedding, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!speaker || !audio || !embedding) {
        snprintf(error, error_size, "invalid arguments");
        return 0;
    }
    float *mel = NULL;
    const int frames = qwen_speaker_mel(audio, samples, &mel, error, error_size);
    if (!frames) return 0;

    const size_t plane = (size_t)CHANNELS * frames;
    float *current = malloc(plane * sizeof(float));
    float *scratch = malloc(plane * sizeof(float));
    float *residual = malloc(plane * sizeof(float));
    float *aggregate = malloc((size_t)AGGREGATE * frames * sizeof(float));
    float *aggregated = malloc((size_t)AGGREGATE * frames * sizeof(float));
    if (!current || !scratch || !residual || !aggregate || !aggregated) {
        free(mel); free(current); free(scratch); free(residual);
        free(aggregate); free(aggregated);
        snprintf(error, error_size, "out of memory");
        return 0;
    }

    tdnn_forward(&speaker->first, mel, frames, current, 1);
    free(mel);

    /* Only the three SE-Res2Net blocks are aggregated; the plain first block
     * feeds the second and is not concatenated. */
    for (int index = 0; index < BLOCKS - 1; index++) {
        const se_res2net *block = &speaker->blocks[index];
        memcpy(residual, current, plane * sizeof(float));
        tdnn_forward(&block->tdnn1, current, frames, scratch, 1);
        res2net_forward(block, scratch, frames, current);
        tdnn_forward(&block->tdnn2, current, frames, scratch, 1);
        se_forward(block, scratch, frames);
        for (size_t entry = 0; entry < plane; entry++)
            scratch[entry] += residual[entry];
        memcpy(aggregate + (size_t)index * CHANNELS * frames, scratch,
               plane * sizeof(float));
        memcpy(current, scratch, plane * sizeof(float));
    }

    tdnn_forward(&speaker->mfa, aggregate, frames, aggregated, 1);

    /* attentive statistics pooling: uniform statistics first, concatenated
     * with the frames, scored, then statistics under those scores */
    float *context = malloc((size_t)AGGREGATE * 3 * frames * sizeof(float));
    float *scores = malloc((size_t)AGGREGATE * frames * sizeof(float));
    float *hidden = malloc((size_t)ATTENTION * frames * sizeof(float));
    float *pooled = malloc((size_t)AGGREGATE * 2 * sizeof(float));
    if (!context || !scores || !hidden || !pooled) {
        free(current); free(scratch); free(residual);
        free(aggregate); free(aggregated);
        free(context); free(scores); free(hidden); free(pooled);
        snprintf(error, error_size, "out of memory");
        return 0;
    }
    for (int channel = 0; channel < AGGREGATE; channel++) {
        const float *row = aggregated + (size_t)channel * frames;
        float mean = 0.0f;
        for (int frame = 0; frame < frames; frame++) mean += row[frame];
        mean /= (float)frames;
        float variance = 0.0f;
        for (int frame = 0; frame < frames; frame++)
            variance += (row[frame] - mean) * (row[frame] - mean);
        variance /= (float)frames;
        if (variance < 1e-12f) variance = 1e-12f;
        const float deviation = sqrtf(variance);
        for (int frame = 0; frame < frames; frame++) {
            context[(size_t)channel * frames + frame] = row[frame];
            context[(size_t)(AGGREGATE + channel) * frames + frame] = mean;
            context[(size_t)(2 * AGGREGATE + channel) * frames + frame] = deviation;
        }
    }
    tdnn_forward(&speaker->asp_tdnn, context, frames, hidden, 1);
    for (size_t entry = 0; entry < (size_t)ATTENTION * frames; entry++)
        hidden[entry] = tanhf(hidden[entry]);
    dispatch_apply((size_t)AGGREGATE, DISPATCH_APPLY_AUTO, ^(size_t slot) {
        const int channel = (int)slot;
        const float *w = speaker->asp_conv_weight + (size_t)channel * ATTENTION;
        float *row = scores + (size_t)channel * frames;
        for (int frame = 0; frame < frames; frame++) {
            float total = speaker->asp_conv_bias[channel];
            for (int index = 0; index < ATTENTION; index++)
                total += w[index] * hidden[(size_t)index * frames + frame];
            row[frame] = total;
        }
        float highest = row[0];
        for (int frame = 1; frame < frames; frame++)
            if (row[frame] > highest) highest = row[frame];
        float sum = 0.0f;
        for (int frame = 0; frame < frames; frame++) {
            row[frame] = expf(row[frame] - highest);
            sum += row[frame];
        }
        for (int frame = 0; frame < frames; frame++) row[frame] /= sum;
    });
    for (int channel = 0; channel < AGGREGATE; channel++) {
        const float *row = aggregated + (size_t)channel * frames;
        const float *weight = scores + (size_t)channel * frames;
        float mean = 0.0f;
        for (int frame = 0; frame < frames; frame++) mean += weight[frame] * row[frame];
        float variance = 0.0f;
        for (int frame = 0; frame < frames; frame++)
            variance += weight[frame] * (row[frame] - mean) * (row[frame] - mean);
        if (variance < 1e-12f) variance = 1e-12f;
        pooled[channel] = mean;
        pooled[AGGREGATE + channel] = sqrtf(variance);
    }

    for (int output = 0; output < QWEN_SPEAKER_DIM; output++) {
        const float *w = speaker->fc_weight + (size_t)output * AGGREGATE * 2;
        float total = speaker->fc_bias[output];
        for (int index = 0; index < AGGREGATE * 2; index++)
            total += w[index] * pooled[index];
        embedding[output] = total;
    }

    free(current); free(scratch); free(residual);
    free(aggregate); free(aggregated);
    free(context); free(scores); free(hidden); free(pooled);
    return 1;
}
