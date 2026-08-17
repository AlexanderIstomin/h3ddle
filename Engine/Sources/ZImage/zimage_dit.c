#include "zimage_dit.h"
#include "qwen_block.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *scratch_error;
static size_t scratch_error_size;

static const uint16_t *take(qwen_weights *store, const char *name,
                            int ndim, const int64_t *shape) {
    const uint16_t *value = qwen_weights_bf16(store, name, ndim, shape,
                                              scratch_error, scratch_error_size);
    if (!value) return NULL;
    return value;
}

#define NEED(target, name, ndim, ...) do {                                    \
        const int64_t shape[] = __VA_ARGS__;                                  \
        (target) = take(store, name, ndim, shape);                            \
        if (!(target)) return 0;                                              \
    } while (0)

static int load_block(qwen_weights *store, const char *prefix,
                      zimage_block_weights *weights, int modulated) {
    char path[256];
#define AT(field, suffix, ndim, ...) do {                                     \
        const int64_t shape[] = __VA_ARGS__;                                  \
        snprintf(path, sizeof(path), "%s" suffix, prefix);                    \
        weights->field = take(store, path, ndim, shape);                      \
        if (!weights->field) return 0;                                        \
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
    return 1;
}

/* The three axes own 32 + 48 + 48 channels. The caption sits at 1..P on the
 * temporal axis with both spatial axes zero; the image starts at P+1 and
 * spreads across the two spatial ones. Image tokens come first in the
 * sequence, which is the opposite of most joint-attention models. */
static void build_positions(const zimage_dit *dit, float *pos_ids) {
    for (int row = 0; row < dit->tokens_side; row++)
        for (int column = 0; column < dit->tokens_side; column++) {
            float *ids = pos_ids + ((size_t)row * dit->tokens_side + column) * 3;
            ids[0] = (float)(dit->caption_padded + 1);
            ids[1] = (float)row;
            ids[2] = (float)column;
        }
    for (int token = 0; token < dit->caption_padded; token++) {
        float *ids = pos_ids + ((size_t)dit->image_tokens + token) * 3;
        ids[0] = (float)(1 + token);
        ids[1] = ids[2] = 0.0f;
    }
}

int zimage_dit_init(zimage_dit *dit, qwen_weights *store, int latent_side,
                    const float *caption, int caption_tokens,
                    zimage_gpu *device, char *error, size_t error_size) {
    memset(dit, 0, sizeof(*dit));
    dit->device = device;
    scratch_error = error;
    scratch_error_size = error_size;

    dit->weights = store;
    dit->latent_side = latent_side;
    dit->tokens_side = latent_side / ZIMAGE_PATCH;
    dit->image_tokens = dit->tokens_side * dit->tokens_side;
    dit->caption_tokens = caption_tokens;
    dit->caption_padded = caption_tokens +
        (ZIMAGE_SEQ_MULTIPLE - caption_tokens % ZIMAGE_SEQ_MULTIPLE) % ZIMAGE_SEQ_MULTIPLE;
    dit->sequence = dit->image_tokens + dit->caption_padded;
    if (dit->image_tokens % ZIMAGE_SEQ_MULTIPLE) {
        snprintf(error, error_size,
                 "latent side %d gives %d image tokens, which is not a multiple "
                 "of %d and would need padding this path does not implement",
                 latent_side, dit->image_tokens, ZIMAGE_SEQ_MULTIPLE);
        return 0;
    }

    dit->cosines = malloc((size_t)dit->sequence * HALF * sizeof(float));
    dit->sines = malloc((size_t)dit->sequence * HALF * sizeof(float));
    dit->unified = malloc((size_t)dit->sequence * DIM * sizeof(float));
    dit->caption_refined = malloc((size_t)dit->caption_padded * DIM * sizeof(float));
    dit->patches = malloc((size_t)dit->image_tokens * PATCH_DIM * sizeof(float));
    dit->head = malloc((size_t)dit->sequence * PATCH_DIM * sizeof(float));
    float *pos_ids = malloc((size_t)dit->sequence * 3 * sizeof(float));
    if (!dit->cosines || !dit->sines || !dit->unified || !dit->caption_refined ||
        !dit->patches || !dit->head || !pos_ids ||
        !zimage_scratch_init(&dit->scratch, dit->sequence)) {
        snprintf(error, error_size, "out of memory sizing the DiT");
        free(pos_ids);
        return 0;
    }
    build_positions(dit, pos_ids);
    zimage_rope_table(pos_ids, dit->cosines, dit->sines, dit->sequence);
    free(pos_ids);

    /* ---- the caption half, once ------------------------------------- */
    const uint16_t *cap_norm, *cap_weight, *cap_bias, *pad_token;
    NEED(cap_norm, "cap_embedder.0.weight", 1, {CAP_DIM});
    NEED(cap_weight, "cap_embedder.1.weight", 2, {DIM, CAP_DIM});
    NEED(cap_bias, "cap_embedder.1.bias", 1, {DIM});
    NEED(pad_token, "cap_pad_token", 2, {1, DIM});

    float *normed = malloc((size_t)caption_tokens * CAP_DIM * sizeof(float));
    if (!normed) { snprintf(error, error_size, "out of memory"); return 0; }
    qwen_rms_norm(caption, cap_norm, normed, CAP_DIM, caption_tokens, NORM_EPS);
    zimage_linear(cap_weight, cap_bias, normed, dit->caption_refined,
                  CAP_DIM, DIM, caption_tokens);
    free(normed);
    /* The tail is the pad token, not a repeat of the last caption row: the
     * repeat happens first and is then overwritten, and only the second is
     * observable. */
    for (int token = caption_tokens; token < dit->caption_padded; token++)
        for (int index = 0; index < DIM; index++)
            dit->caption_refined[(size_t)token * DIM + index] =
                qwen_widen(pad_token[index]);

    const float *cap_cosines = dit->cosines + (size_t)dit->image_tokens * HALF;
    const float *cap_sines = dit->sines + (size_t)dit->image_tokens * HALF;
    if (dit->device)
        return zimage_gpu_refine_context(dit->device, dit->caption_refined,
                                         dit->caption_padded, cap_cosines,
                                         cap_sines, error, error_size);
    for (int layer = 0; layer < ZIMAGE_REFINERS; layer++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "context_refiner.%d.", layer);
        zimage_block_weights weights;
        if (!load_block(store, prefix, &weights, 0)) return 0;
        zimage_block_forward(&weights, &dit->scratch, dit->caption_refined,
                             NULL, cap_cosines, cap_sines, dit->caption_padded);
    }
    return 1;
}

void zimage_dit_release(zimage_dit *dit) {
    free(dit->cosines); free(dit->sines); free(dit->unified);
    free(dit->caption_refined); free(dit->patches); free(dit->head);
    zimage_scratch_release(&dit->scratch);
    memset(dit, 0, sizeof(*dit));
}

static void emit(zimage_dit_tap tap, void *context, const char *stage,
                 const float *values, size_t count) {
    if (tap) tap(stage, values, count, context);
}

int zimage_dit_step(zimage_dit *dit, const float *latent, float timestep,
                    float *velocity, zimage_dit_tap tap, void *context) {
    qwen_weights *store = dit->weights;
    char error[512];
    scratch_error = error;
    scratch_error_size = sizeof(error);

    /* ---- conditioning ------------------------------------------------ */
    /* cos before sin, at t * 1000. Neither is stated in any config. */
    float t_freq[ADALN], hidden[1024], adaln_input[ADALN];
    const float scaled = timestep * 1000.0f;
    for (int index = 0; index < ADALN / 2; index++) {
        const double frequency =
            exp(-log(10000.0) * (double)index / (double)(ADALN / 2));
        const double angle = (double)scaled * frequency;
        t_freq[index] = (float)cos(angle);
        t_freq[ADALN / 2 + index] = (float)sin(angle);
    }
    const uint16_t *mlp0, *mlp0_bias, *mlp2, *mlp2_bias;
    NEED(mlp0, "t_embedder.mlp.0.weight", 2, {1024, ADALN});
    NEED(mlp0_bias, "t_embedder.mlp.0.bias", 1, {1024});
    NEED(mlp2, "t_embedder.mlp.2.weight", 2, {ADALN, 1024});
    NEED(mlp2_bias, "t_embedder.mlp.2.bias", 1, {ADALN});
    zimage_linear(mlp0, mlp0_bias, t_freq, hidden, ADALN, 1024, 1);
    for (int index = 0; index < 1024; index++) hidden[index] = zimage_silu(hidden[index]);
    zimage_linear(mlp2, mlp2_bias, hidden, adaln_input, 1024, ADALN, 1);

    /* ---- patchify and embed ------------------------------------------ */
    /* Channel varies fastest within a patch. */
    const int side = dit->latent_side, span = dit->tokens_side;
    for (int row = 0; row < span; row++)
        for (int column = 0; column < span; column++) {
            float *patch = dit->patches + ((size_t)row * span + column) * PATCH_DIM;
            for (int y = 0; y < ZIMAGE_PATCH; y++)
                for (int x = 0; x < ZIMAGE_PATCH; x++)
                    for (int channel = 0; channel < ZIMAGE_LATENT_CHANNELS; channel++)
                        patch[(y * ZIMAGE_PATCH + x) * ZIMAGE_LATENT_CHANNELS + channel] =
                            latent[((size_t)channel * side +
                                    (row * ZIMAGE_PATCH + y)) * side +
                                   (column * ZIMAGE_PATCH + x)];
        }
    const uint16_t *x_weight, *x_bias;
    NEED(x_weight, "x_embedder.weight", 2, {DIM, PATCH_DIM});
    NEED(x_bias, "x_embedder.bias", 1, {DIM});
    zimage_linear(x_weight, x_bias, dit->patches, dit->unified,
                  PATCH_DIM, DIM, dit->image_tokens);

    /* The trunk writes through the caption rows, so restore them. */
    memcpy(dit->unified + (size_t)dit->image_tokens * DIM, dit->caption_refined,
           (size_t)dit->caption_padded * DIM * sizeof(float));

    /* ---- noise refiners and trunk ------------------------------------ */
    if (dit->device) {
        char problem[512];
        if (!zimage_gpu_forward(dit->device, dit->unified, dit->image_tokens,
                                dit->sequence, adaln_input, dit->cosines,
                                dit->sines, problem, sizeof(problem))) {
            fprintf(stderr, "gpu forward: %s\n", problem);
            return 0;
        }
        goto head;
    }
    zimage_block_weights weights;
    char prefix[64];
    for (int layer = 0; layer < ZIMAGE_REFINERS; layer++) {
        snprintf(prefix, sizeof(prefix), "noise_refiner.%d.", layer);
        if (!load_block(store, prefix, &weights, 1)) return 0;
        zimage_block_forward(&weights, &dit->scratch, dit->unified, adaln_input,
                             dit->cosines, dit->sines, dit->image_tokens);
    }
    emit(tap, context, "unified", dit->unified, (size_t)dit->sequence * DIM);

    /* ---- trunk -------------------------------------------------------- */
    for (int layer = 0; layer < ZIMAGE_LAYERS; layer++) {
        snprintf(prefix, sizeof(prefix), "layers.%d.", layer);
        if (!load_block(store, prefix, &weights, 1)) return 0;
        zimage_block_forward(&weights, &dit->scratch, dit->unified, adaln_input,
                             dit->cosines, dit->sines, dit->sequence);
        char name[24];
        snprintf(name, sizeof(name), "trunk_%02d", layer);
        emit(tap, context, name, dit->unified, (size_t)dit->sequence * DIM);
    }

head:
    /* ---- head --------------------------------------------------------- */
    /* SiLU first here, where the block's adaLN has none. */
    float silued[ADALN];
    float *scale = malloc(DIM * sizeof(float));
    if (!scale) return 0;
    for (int index = 0; index < ADALN; index++)
        silued[index] = zimage_silu(adaln_input[index]);
    const uint16_t *head_weight, *head_bias, *linear_weight, *linear_bias;
    NEED(head_weight, "final_layer.adaLN_modulation.1.weight", 2, {DIM, ADALN});
    NEED(head_bias, "final_layer.adaLN_modulation.1.bias", 1, {DIM});
    NEED(linear_weight, "final_layer.linear.weight", 2, {PATCH_DIM, DIM});
    NEED(linear_bias, "final_layer.linear.bias", 1, {PATCH_DIM});
    zimage_linear(head_weight, head_bias, silued, scale, ADALN, DIM, 1);
    for (int index = 0; index < DIM; index++) scale[index] += 1.0f;

    /* LayerNorm, not RMSNorm, and without affine — the only mean-subtracting
     * normaliser in the model. */
    for (int token = 0; token < dit->sequence; token++) {
        const float *row = dit->unified + (size_t)token * DIM;
        float *out = dit->scratch.normed + (size_t)token * DIM;
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
    free(scale);
    zimage_linear(linear_weight, linear_bias, dit->scratch.normed, dit->head,
                  DIM, PATCH_DIM, dit->sequence);
    emit(tap, context, "head", dit->head, (size_t)dit->sequence * PATCH_DIM);

    /* ---- unpatchify, image tokens only -------------------------------- */
    for (int row = 0; row < span; row++)
        for (int column = 0; column < span; column++) {
            const float *patch = dit->head + ((size_t)row * span + column) * PATCH_DIM;
            for (int y = 0; y < ZIMAGE_PATCH; y++)
                for (int x = 0; x < ZIMAGE_PATCH; x++)
                    for (int channel = 0; channel < ZIMAGE_LATENT_CHANNELS; channel++)
                        velocity[((size_t)channel * side +
                                  (row * ZIMAGE_PATCH + y)) * side +
                                 (column * ZIMAGE_PATCH + x)] =
                            patch[(y * ZIMAGE_PATCH + x) * ZIMAGE_LATENT_CHANNELS + channel];
        }
    emit(tap, context, "output", velocity,
         (size_t)ZIMAGE_LATENT_CHANNELS * side * side);
    return 1;
}
