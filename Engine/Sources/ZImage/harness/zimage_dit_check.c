/* Stage 4: does one S3-DiT layer come out right?
 *
 * The text encoder was a config and the VAE was pieces that already existed.
 * This is the first Z-Image code that could genuinely be wrong, and it has
 * four independent ways to be wrong at once: sandwich norm, per-head QK-norm
 * inside a bidirectional attention, low-rank adaLN off a 256-wide vector, and
 * a 3D rope split 32/48/48 at theta 256.
 *
 * Each is therefore checked on its own before the layer is checked as a whole.
 * The rope table is compared against the reference's own freqs_cis, so a wrong
 * table is a wrong table and not "the layer is off by 3e-02"; the conditioning
 * and the embedders likewise. Only then does the block run.
 *
 * Two things here are deliberately not the same as the Qwen block next door,
 * and both would be silently absorbed if this reused it:
 *
 *  - the rotation pairs adjacent channels (2i, 2i+1), where Qwen pairs i with
 *    i + head_dim/2. Both are "rope"; they are not the same permutation;
 *  - attention is bidirectional. There is no causal mask and no kv cache.
 *
 *   ./zimage_dit_check <weights.safetensors> <golden.safetensors>
 */
#include "qwen_block.h"
#include "qwen_weights.h"

#include <dispatch/dispatch.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM          3840
#define HEADS        30
#define HEAD_DIM     128
#define FFN          10240
#define CAP_DIM      2560
#define ADALN        256
#define PATCH_DIM    64

#define LATENT_CHANNELS 16
#define LATENT_SIDE  32
#define PATCH        2
#define TOKENS_SIDE  (LATENT_SIDE / PATCH)
#define IMAGE_TOKENS (TOKENS_SIDE * TOKENS_SIDE)
#define CAP_TOKENS   40
#define CAP_PADDED   64
#define SEQUENCE     (IMAGE_TOKENS + CAP_PADDED)

#define NORM_EPS     1e-5f
#define FINAL_EPS    1e-6f
#define ROPE_THETA   256.0
#define HALF         (HEAD_DIM / 2)

static const int AXES[3] = {32, 48, 48};

/* f32 arithmetic against an f32 reference, so this measures the port and not
 * the format. bf16 weights widen exactly; nothing here rounds. */
static double compare(const char *label, const float *ours, const float *theirs,
                      size_t count) {
    double square = 0.0, reference = 0.0, worst = 0.0;
    for (size_t index = 0; index < count; index++) {
        const double difference = (double)ours[index] - (double)theirs[index];
        square += difference * difference;
        reference += (double)theirs[index] * (double)theirs[index];
        if (fabs(difference) > worst) worst = fabs(difference);
    }
    const double rms = sqrt(square / (double)count);
    const double scale = sqrt(reference / (double)count);
    const double relative = scale > 0.0 ? rms / scale : 0.0;
    printf("  %-22s RMS rel %8.2e   worst %8.2e\n", label, relative, worst);
    return relative;
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

/* y[rows][outputs] = x[rows][inputs] . weight^T + bias */
static void linear_bias(const uint16_t *weight, const uint16_t *bias,
                        const float *x, float *y,
                        int inputs, int outputs, int rows) {
    qwen_matmul(weight, x, y, inputs, outputs, rows);
    if (!bias) return;
    for (int row = 0; row < rows; row++)
        for (int output = 0; output < outputs; output++)
            y[(size_t)row * outputs + output] += qwen_widen(bias[output]);
}

/* ---- rope ------------------------------------------------------------- */
/* The three axes each own a slice of the head: 32 + 48 + 48 channels, so 16 +
 * 24 + 24 complex pairs. A token's three coordinates index one table each and
 * the results are concatenated — which is why a text token, whose two spatial
 * coordinates are both zero, still occupies 96 channels of rotation at angle
 * zero rather than only the 32 it varies in. */
static void rope_table(const float *pos_ids, float *cosines, float *sines,
                       int tokens) {
    for (int token = 0; token < tokens; token++) {
        int channel = 0;
        for (int axis = 0; axis < 3; axis++) {
            const int dim = AXES[axis];
            const double position = (double)pos_ids[(size_t)token * 3 + axis];
            for (int index = 0; index < dim / 2; index++) {
                const double frequency =
                    1.0 / pow(ROPE_THETA, (double)(2 * index) / (double)dim);
                const double angle = position * frequency;
                cosines[(size_t)token * HALF + channel] = (float)cos(angle);
                sines[(size_t)token * HALF + channel] = (float)sin(angle);
                channel++;
            }
        }
    }
}

/* Adjacent-pair rotation, in place over [tokens][heads][head_dim]. */
static void apply_rope(float *x, const float *cosines, const float *sines,
                       int tokens) {
    for (int token = 0; token < tokens; token++) {
        const float *cosine = cosines + (size_t)token * HALF;
        const float *sine = sines + (size_t)token * HALF;
        for (int head = 0; head < HEADS; head++) {
            float *row = x + ((size_t)token * HEADS + head) * HEAD_DIM;
            for (int pair = 0; pair < HALF; pair++) {
                const float real = row[2 * pair], imaginary = row[2 * pair + 1];
                row[2 * pair] = real * cosine[pair] - imaginary * sine[pair];
                row[2 * pair + 1] = real * sine[pair] + imaginary * cosine[pair];
            }
        }
    }
}

/* Per-head RMS norm over head_dim, sharing one [head_dim] weight across every
 * head, applied after projection and before rotation. */
static void head_norm(float *x, const uint16_t *weight, int tokens) {
    for (size_t row = 0; row < (size_t)tokens * HEADS; row++) {
        float *values = x + row * HEAD_DIM;
        float sum = 0.0f;
        for (int index = 0; index < HEAD_DIM; index++)
            sum += values[index] * values[index];
        const float scale = 1.0f / sqrtf(sum / (float)HEAD_DIM + NORM_EPS);
        for (int index = 0; index < HEAD_DIM; index++)
            values[index] *= qwen_widen(weight[index]) * scale;
    }
}

typedef struct {
    const uint16_t *qkv, *out, *q_norm, *k_norm;
    const uint16_t *attention_norm1, *attention_norm2;
    const uint16_t *ffn_norm1, *ffn_norm2;
    const uint16_t *w1, *w2, *w3;
    const uint16_t *adaln_weight, *adaln_bias;   /* NULL for the refiners */
} block_weights;

typedef struct {
    float *normed, *qkv, *query, *key, *value, *attention, *gate, *up, *modulation;
} block_scratch;

/* Bidirectional attention: every token sees every token, so there is no mask
 * and no cache. Parallel over heads because at 320 tokens this is most of the
 * layer's cost. */
static void attention(const block_weights *weights, block_scratch *scratch,
                      const float *cosines, const float *sines, int tokens) {
    qwen_matmul(weights->qkv, scratch->normed, scratch->qkv, DIM, 3 * DIM, tokens);
    for (int token = 0; token < tokens; token++) {
        const float *row = scratch->qkv + (size_t)token * 3 * DIM;
        memcpy(scratch->query + (size_t)token * DIM, row, DIM * sizeof(float));
        memcpy(scratch->key + (size_t)token * DIM, row + DIM, DIM * sizeof(float));
        memcpy(scratch->value + (size_t)token * DIM, row + 2 * DIM, DIM * sizeof(float));
    }
    head_norm(scratch->query, weights->q_norm, tokens);
    head_norm(scratch->key, weights->k_norm, tokens);
    apply_rope(scratch->query, cosines, sines, tokens);
    apply_rope(scratch->key, cosines, sines, tokens);

    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    float *attention_out = scratch->attention;
    const float *query = scratch->query, *key = scratch->key, *value = scratch->value;
    dispatch_apply(HEADS, DISPATCH_APPLY_AUTO, ^(size_t head) {
        float *scores = malloc((size_t)tokens * sizeof(float));
        for (int i = 0; i < tokens; i++) {
            const float *q = query + ((size_t)i * HEADS + head) * HEAD_DIM;
            float largest = -INFINITY;
            for (int j = 0; j < tokens; j++) {
                const float *k = key + ((size_t)j * HEADS + head) * HEAD_DIM;
                float dot = 0.0f;
                for (int index = 0; index < HEAD_DIM; index++)
                    dot += q[index] * k[index];
                scores[j] = dot * scale;
                if (scores[j] > largest) largest = scores[j];
            }
            float total = 0.0f;
            for (int j = 0; j < tokens; j++) {
                scores[j] = expf(scores[j] - largest);
                total += scores[j];
            }
            const float inverse = 1.0f / total;
            float *out = attention_out + ((size_t)i * HEADS + head) * HEAD_DIM;
            memset(out, 0, HEAD_DIM * sizeof(float));
            for (int j = 0; j < tokens; j++) {
                const float weight = scores[j] * inverse;
                const float *v = value + ((size_t)j * HEADS + head) * HEAD_DIM;
                for (int index = 0; index < HEAD_DIM; index++)
                    out[index] += weight * v[index];
            }
        }
        free(scores);
    });
    /* to_out reads the heads back as one 3840-wide row, which is what the
     * [tokens][heads][head_dim] layout already is. */
    qwen_matmul(weights->out, scratch->attention, scratch->normed, DIM, DIM, tokens);
}

static void feed_forward(const block_weights *weights, block_scratch *scratch,
                         const float *x, float *y, int tokens) {
    qwen_matmul(weights->w1, x, scratch->gate, DIM, FFN, tokens);
    qwen_matmul(weights->w3, x, scratch->up, DIM, FFN, tokens);
    for (size_t index = 0; index < (size_t)tokens * FFN; index++)
        scratch->gate[index] = silu(scratch->gate[index]) * scratch->up[index];
    qwen_matmul(weights->w2, scratch->gate, y, FFN, DIM, tokens);
}

/* Sandwich norm: a norm before the sublayer *and* one after it, with the
 * residual added outside both. The modulated form scales the input norm and
 * gates the output norm; the unmodulated form keeps both norms and drops the
 * scale and the gate rather than setting them to one. */
static void block_forward(const block_weights *weights, block_scratch *scratch,
                          float *x, const float *adaln_input,
                          const float *cosines, const float *sines, int tokens) {
    const float *scale_msa = NULL, *gate_msa = NULL;
    const float *scale_mlp = NULL, *gate_mlp = NULL;
    if (weights->adaln_weight) {
        linear_bias(weights->adaln_weight, weights->adaln_bias, adaln_input,
                    scratch->modulation, ADALN, 4 * DIM, 1);
        float *modulation = scratch->modulation;
        for (int index = 0; index < DIM; index++) {
            modulation[index] += 1.0f;                                  /* scale_msa */
            modulation[DIM + index] = tanhf(modulation[DIM + index]);   /* gate_msa  */
            modulation[2 * DIM + index] += 1.0f;                        /* scale_mlp */
            modulation[3 * DIM + index] = tanhf(modulation[3 * DIM + index]);
        }
        scale_msa = modulation;
        gate_msa = modulation + DIM;
        scale_mlp = modulation + 2 * DIM;
        gate_mlp = modulation + 3 * DIM;
    }

    float *residual = malloc((size_t)tokens * DIM * sizeof(float));

    qwen_rms_norm(x, weights->attention_norm1, scratch->normed, DIM, tokens, NORM_EPS);
    if (scale_msa)
        for (int token = 0; token < tokens; token++)
            for (int index = 0; index < DIM; index++)
                scratch->normed[(size_t)token * DIM + index] *= scale_msa[index];
    attention(weights, scratch, cosines, sines, tokens);
    qwen_rms_norm(scratch->normed, weights->attention_norm2, residual, DIM, tokens, NORM_EPS);
    for (int token = 0; token < tokens; token++)
        for (int index = 0; index < DIM; index++)
            x[(size_t)token * DIM + index] +=
                (gate_msa ? gate_msa[index] : 1.0f) * residual[(size_t)token * DIM + index];

    qwen_rms_norm(x, weights->ffn_norm1, scratch->normed, DIM, tokens, NORM_EPS);
    if (scale_mlp)
        for (int token = 0; token < tokens; token++)
            for (int index = 0; index < DIM; index++)
                scratch->normed[(size_t)token * DIM + index] *= scale_mlp[index];
    feed_forward(weights, scratch, scratch->normed, residual, tokens);
    qwen_rms_norm(residual, weights->ffn_norm2, residual, DIM, tokens, NORM_EPS);
    for (int token = 0; token < tokens; token++)
        for (int index = 0; index < DIM; index++)
            x[(size_t)token * DIM + index] +=
                (gate_mlp ? gate_mlp[index] : 1.0f) * residual[(size_t)token * DIM + index];

    free(residual);
}

static char error[512];

static const uint16_t *take(qwen_weights *store, const char *name,
                            int ndim, const int64_t *shape) {
    const uint16_t *value = qwen_weights_bf16(store, name, ndim, shape,
                                              error, sizeof(error));
    if (!value) { fprintf(stderr, "%s\n", error); exit(1); }
    return value;
}

static const float *golden_f32(qwen_weights *store, const char *name,
                               int ndim, const int64_t *shape) {
    const float *value = qwen_weights_f32(store, name, ndim, shape,
                                          error, sizeof(error));
    if (!value) { fprintf(stderr, "golden: %s\n", error); exit(1); }
    return value;
}

static void load_block(qwen_weights *store, const char *name,
                       block_weights *weights, int modulated) {
    char path[256];
#define AT(field, suffix, ndim, ...) do {                                     \
        const int64_t shape[] = __VA_ARGS__;                                  \
        snprintf(path, sizeof(path), "%s." suffix, name);                     \
        weights->field = take(store, path, ndim, shape);                      \
    } while (0)
    AT(qkv, "attention.qkv.weight", 2, {3 * DIM, DIM});
    AT(out, "attention.out.weight", 2, {DIM, DIM});
    AT(q_norm, "attention.q_norm.weight", 1, {HEAD_DIM});
    AT(k_norm, "attention.k_norm.weight", 1, {HEAD_DIM});
    AT(attention_norm1, "attention_norm1.weight", 1, {DIM});
    AT(attention_norm2, "attention_norm2.weight", 1, {DIM});
    AT(ffn_norm1, "ffn_norm1.weight", 1, {DIM});
    AT(ffn_norm2, "ffn_norm2.weight", 1, {DIM});
    AT(w1, "feed_forward.w1.weight", 2, {FFN, DIM});
    AT(w2, "feed_forward.w2.weight", 2, {DIM, FFN});
    AT(w3, "feed_forward.w3.weight", 2, {FFN, DIM});
    if (modulated) {
        AT(adaln_weight, "adaLN_modulation.0.weight", 2, {4 * DIM, ADALN});
        AT(adaln_bias, "adaLN_modulation.0.bias", 1, {4 * DIM});
    } else {
        weights->adaln_weight = weights->adaln_bias = NULL;
    }
#undef AT
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <weights.safetensors> <golden.safetensors>\n",
                argv[0]);
        return 2;
    }
    qwen_weights *store = qwen_weights_open(argv[1], error, sizeof(error));
    if (!store) { fprintf(stderr, "weights: %s\n", error); return 1; }
    qwen_weights *golden = qwen_weights_open(argv[2], error, sizeof(error));
    if (!golden) { fprintf(stderr, "golden: %s\n", error); return 1; }

    printf("%d tokens = %d image + %d caption, %d wide, %d heads x %d\n\n",
           SEQUENCE, IMAGE_TOKENS, CAP_PADDED, DIM, HEADS, HEAD_DIM);

    /* ---- the rope table, on its own ---------------------------------- */
    const int64_t ids_shape[2] = {SEQUENCE, 3};
    const float *pos_ids = golden_f32(golden, "pos_ids", 2, ids_shape);
    float *cosines = malloc((size_t)SEQUENCE * HALF * sizeof(float));
    float *sines = malloc((size_t)SEQUENCE * HALF * sizeof(float));
    rope_table(pos_ids, cosines, sines, SEQUENCE);

    const int64_t freq_shape[2] = {SEQUENCE, HALF};
    printf("rope:\n");
    compare("freqs real", cosines, golden_f32(golden, "freqs_real", 2, freq_shape),
            (size_t)SEQUENCE * HALF);
    compare("freqs imag", sines, golden_f32(golden, "freqs_imag", 2, freq_shape),
            (size_t)SEQUENCE * HALF);

    /* ---- conditioning ------------------------------------------------ */
    /* cos before sin, and the whole thing at t * 1000. Both are conventions
     * the config does not state. */
    const int64_t one[1] = {1};
    const float timestep = golden_f32(golden, "timestep", 1, one)[0] * 1000.0f;
    float t_freq[ADALN];
    for (int index = 0; index < ADALN / 2; index++) {
        const double frequency =
            exp(-log(10000.0) * (double)index / (double)(ADALN / 2));
        const double angle = (double)timestep * frequency;
        t_freq[index] = (float)cos(angle);
        t_freq[ADALN / 2 + index] = (float)sin(angle);
    }
    const int64_t t_freq_shape[2] = {1, ADALN};
    printf("conditioning:\n");
    compare("timestep embedding", t_freq,
            golden_f32(golden, "t_freq", 2, t_freq_shape), ADALN);

    const int64_t mlp0_shape[2] = {1024, ADALN};
    const int64_t mlp0_bias[1] = {1024};
    const int64_t mlp2_shape[2] = {ADALN, 1024};
    const int64_t mlp2_bias[1] = {ADALN};
    float hidden[1024], adaln_input[ADALN];
    linear_bias(take(store, "t_embedder.mlp.0.weight", 2, mlp0_shape),
                take(store, "t_embedder.mlp.0.bias", 1, mlp0_bias),
                t_freq, hidden, ADALN, 1024, 1);
    for (int index = 0; index < 1024; index++) hidden[index] = silu(hidden[index]);
    linear_bias(take(store, "t_embedder.mlp.2.weight", 2, mlp2_shape),
                take(store, "t_embedder.mlp.2.bias", 1, mlp2_bias),
                hidden, adaln_input, 1024, ADALN, 1);
    compare("adaln_input", adaln_input,
            golden_f32(golden, "adaln_input", 2, t_freq_shape), ADALN);

    /* ---- embedders --------------------------------------------------- */
    float *unified = malloc((size_t)SEQUENCE * DIM * sizeof(float));

    /* Patchify here rather than reading the reference's patches, because the
     * layout is the seam: the channel varies *fastest* within a patch, so a
     * channel-major port produces a well-formed image of the wrong thing and
     * every downstream comparison still passes. */
    const int64_t latent_shape[4] = {LATENT_CHANNELS, 1, LATENT_SIDE, LATENT_SIDE};
    const float *latent = golden_f32(golden, "latent", 4, latent_shape);
    float *patches = malloc((size_t)IMAGE_TOKENS * PATCH_DIM * sizeof(float));
    for (int row = 0; row < TOKENS_SIDE; row++)
        for (int column = 0; column < TOKENS_SIDE; column++) {
            float *patch = patches + ((size_t)row * TOKENS_SIDE + column) * PATCH_DIM;
            for (int y = 0; y < PATCH; y++)
                for (int x = 0; x < PATCH; x++)
                    for (int channel = 0; channel < LATENT_CHANNELS; channel++)
                        patch[(y * PATCH + x) * LATENT_CHANNELS + channel] =
                            latent[((size_t)channel * LATENT_SIDE +
                                    (row * PATCH + y)) * LATENT_SIDE +
                                   (column * PATCH + x)];
        }
    const int64_t patch_shape[2] = {IMAGE_TOKENS, PATCH_DIM};
    printf("embedders:\n");
    compare("patchify", patches, golden_f32(golden, "patches", 2, patch_shape),
            (size_t)IMAGE_TOKENS * PATCH_DIM);

    const int64_t x_weight[2] = {DIM, PATCH_DIM};
    const int64_t x_bias[1] = {DIM};
    linear_bias(take(store, "x_embedder.weight", 2, x_weight),
                take(store, "x_embedder.bias", 1, x_bias),
                patches, unified, PATCH_DIM, DIM, IMAGE_TOKENS);
    const int64_t image_shape[2] = {IMAGE_TOKENS, DIM};
    compare("x_embedder", unified,
            golden_f32(golden, "image_embedded", 2, image_shape),
            (size_t)IMAGE_TOKENS * DIM);

    const int64_t caption_shape[2] = {CAP_TOKENS, CAP_DIM};
    const int64_t cap_norm[1] = {CAP_DIM};
    const int64_t cap_weight[2] = {DIM, CAP_DIM};
    const int64_t cap_bias[1] = {DIM};
    float *caption = malloc((size_t)CAP_TOKENS * CAP_DIM * sizeof(float));
    qwen_rms_norm(golden_f32(golden, "caption", 2, caption_shape),
                  take(store, "cap_embedder.0.weight", 1, cap_norm),
                  caption, CAP_DIM, CAP_TOKENS, NORM_EPS);
    float *cap_out = unified + (size_t)IMAGE_TOKENS * DIM;
    linear_bias(take(store, "cap_embedder.1.weight", 2, cap_weight),
                take(store, "cap_embedder.1.bias", 1, cap_bias),
                caption, cap_out, CAP_DIM, DIM, CAP_TOKENS);
    /* The padded tail is the pad token, not a repeat of the last caption
     * row — the repeat happens first and is then overwritten. */
    const int64_t pad_shape[2] = {1, DIM};
    const uint16_t *pad_token = take(store, "cap_pad_token", 2, pad_shape);
    for (int token = CAP_TOKENS; token < CAP_PADDED; token++)
        for (int index = 0; index < DIM; index++)
            cap_out[(size_t)token * DIM + index] = qwen_widen(pad_token[index]);
    const int64_t cap_shape[2] = {CAP_PADDED, DIM};
    compare("cap_embedder", cap_out,
            golden_f32(golden, "cap_embedded", 2, cap_shape),
            (size_t)CAP_PADDED * DIM);

    const int64_t unified_shape[2] = {SEQUENCE, DIM};
    compare("unified sequence", unified,
            golden_f32(golden, "unified", 2, unified_shape),
            (size_t)SEQUENCE * DIM);

    /* ---- the blocks -------------------------------------------------- */
    block_scratch scratch = {
        .normed = malloc((size_t)SEQUENCE * DIM * sizeof(float)),
        .qkv = malloc((size_t)SEQUENCE * 3 * DIM * sizeof(float)),
        .query = malloc((size_t)SEQUENCE * DIM * sizeof(float)),
        .key = malloc((size_t)SEQUENCE * DIM * sizeof(float)),
        .value = malloc((size_t)SEQUENCE * DIM * sizeof(float)),
        .attention = malloc((size_t)SEQUENCE * DIM * sizeof(float)),
        .gate = malloc((size_t)SEQUENCE * FFN * sizeof(float)),
        .up = malloc((size_t)SEQUENCE * FFN * sizeof(float)),
        .modulation = malloc(4 * DIM * sizeof(float)),
    };
    float *x = malloc((size_t)SEQUENCE * DIM * sizeof(float));
    float *layer0_out = malloc((size_t)SEQUENCE * DIM * sizeof(float));

    printf("blocks:\n");
    const struct { const char *name; int modulated; } blocks[] = {
        {"layer0", 1}, {"context_refiner0", 0}, {"noise_refiner0", 1},
    };
    for (size_t index = 0; index < sizeof(blocks) / sizeof(blocks[0]); index++) {
        block_weights weights;
        load_block(store, blocks[index].name, &weights, blocks[index].modulated);
        memcpy(x, unified, (size_t)SEQUENCE * DIM * sizeof(float));
        block_forward(&weights, &scratch, x, adaln_input, cosines, sines, SEQUENCE);
        char label[64];
        snprintf(label, sizeof(label), "%s%s", blocks[index].name,
                 blocks[index].modulated ? "" : " (no adaLN)");
        char name[64];
        snprintf(name, sizeof(name), "%s_out", blocks[index].name);
        compare(label, x, golden_f32(golden, name, 2, unified_shape),
                (size_t)SEQUENCE * DIM);
        if (index == 0) memcpy(layer0_out, x, (size_t)SEQUENCE * DIM * sizeof(float));
    }

    /* ---- the head ---------------------------------------------------- */
    /* SiLU first here, where the block's adaLN has none. */
    float scale[DIM];
    float silued[ADALN];
    for (int index = 0; index < ADALN; index++) silued[index] = silu(adaln_input[index]);
    const int64_t head_weight[2] = {DIM, ADALN};
    const int64_t head_bias[1] = {DIM};
    linear_bias(take(store, "final_layer.adaLN_modulation.1.weight", 2, head_weight),
                take(store, "final_layer.adaLN_modulation.1.bias", 1, head_bias),
                silued, scale, ADALN, DIM, 1);
    for (int index = 0; index < DIM; index++) scale[index] += 1.0f;

    /* LayerNorm, not RMSNorm, and without affine — the only mean-subtracting
     * normaliser in the model. */
    float *normed = scratch.normed;
    for (int token = 0; token < SEQUENCE; token++) {
        const float *row = layer0_out + (size_t)token * DIM;
        float *out = normed + (size_t)token * DIM;
        float mean = 0.0f;
        for (int index = 0; index < DIM; index++) mean += row[index];
        mean /= (float)DIM;
        float variance = 0.0f;
        for (int index = 0; index < DIM; index++)
            variance += (row[index] - mean) * (row[index] - mean);
        variance /= (float)DIM;
        const float inverse = 1.0f / sqrtf(variance + FINAL_EPS);
        for (int index = 0; index < DIM; index++)
            out[index] = (row[index] - mean) * inverse * scale[index];
    }
    float *head = malloc((size_t)SEQUENCE * PATCH_DIM * sizeof(float));
    const int64_t linear_weight[2] = {PATCH_DIM, DIM};
    const int64_t linear_bias_shape[1] = {PATCH_DIM};
    linear_bias(take(store, "final_layer.linear.weight", 2, linear_weight),
                take(store, "final_layer.linear.bias", 1, linear_bias_shape),
                normed, head, DIM, PATCH_DIM, SEQUENCE);
    const int64_t head_shape[2] = {SEQUENCE, PATCH_DIM};
    printf("head:\n");
    compare("final_layer", head, golden_f32(golden, "final_out", 2, head_shape),
            (size_t)SEQUENCE * PATCH_DIM);

    qwen_weights_close(store);
    qwen_weights_close(golden);
    return 0;
}
