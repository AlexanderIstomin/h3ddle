#include "zimage_gpu.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM       3840
#define HEADS     30
#define HEAD_DIM  128
#define ROPE_HALF (HEAD_DIM / 2)
#define FFN       10240
#define ADALN     256
#define LAYERS    30
#define REFINERS  2
#define BLOCKS    (LAYERS + REFINERS * 2)
#define NORM_EPS  1e-5f

/* Five slots, because h3_adaln_f32 reads a shift Z-Image does not have and
 * slot 0 is held at zero for it. */
#define SLOTS       5
#define UNIT_SLOTS  2
#define UNIT_GATE   1

/* Below this many rows the bf16 path wins; above it the f32 one does. */
#define ZIMAGE_F32_ATTENTION_ROWS 512
#define SHIFT_SLOT  0
#define SCALE_MSA   1
#define GATE_MSA    2
#define SCALE_MLP   3
#define GATE_MLP    4

typedef struct {
    h3_gpu_tensor *qkv, *qkv_scales;
    h3_gpu_tensor *out, *out_scales;
    h3_gpu_tensor *w13, *w13_scales;
    h3_gpu_tensor *w2, *w2_scales;
    h3_gpu_tensor *adaln, *adaln_scales, *adaln_bias;
    uint32_t qkv_group, out_group, w13_group, w2_group, adaln_group;
    h3_gpu_tensor *q_norm, *k_norm;
    h3_gpu_tensor *attention_norm1, *attention_norm2;
    h3_gpu_tensor *ffn_norm1, *ffn_norm2;
    int modulated;
} zimage_gpu_block;

struct zimage_gpu {
    h3_gpu *gpu;
    h3_weight_store *store;
    int max_tokens;
    zimage_gpu_block blocks[BLOCKS];   /* noise 0-1, context 2-3, trunk 4.. */
    h3_gpu_tensor *hidden, *normed, *branch, *heads;
    h3_gpu_tensor *qkv, *query, *key, *value;
    h3_gpu_tensor *ff, *activated;
    h3_gpu_tensor *query32, *key32, *value32, *heads32;
    h3_gpu_tensor *modulation, *mod_linear, *adaln_input;
    h3_gpu_tensor *row_map, *rope_cos, *rope_sin, *ones;
    double seconds;
};

static int fail(char *error, size_t error_size, const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
    return 0;
}

/* The bf16 path wants the norms in the dtype the checkpoint already stores
 * them in, so these load straight through — the f32 path had to widen every
 * one of them on the way in. */
static h3_gpu_tensor *load_vector(zimage_gpu *z, const char *name,
                                  uint64_t width, char *error, size_t size) {
    const uint64_t shape[1] = {width};
    return h3_weight_load_bf16(z->store, z->gpu, name, 1, shape, error, size);
}

static uint16_t narrow(float value) {
    union { float number; uint32_t bits; } cast = {.number = value};
    cast.bits += 0x7fffu + ((cast.bits >> 16) & 1u);
    return (uint16_t)(cast.bits >> 16);
}

/* Channel 2i to slot i and 2i+1 to slot i+64, within each head. Applied to the
 * q and k row blocks of the fused projection and to the two norm weights, it
 * turns the engine's rotate_half into Z-Image's adjacent-pair rotation. */
static void permuted_index(int *map) {
    for (int pair = 0; pair < ROPE_HALF; pair++) {
        map[pair] = 2 * pair;
        map[pair + ROPE_HALF] = 2 * pair + 1;
    }
}

static h3_gpu_tensor *permute_norm(zimage_gpu *z, const char *name,
                                   char *error, size_t size) {
    h3_gpu_tensor *plain = load_vector(z, name, HEAD_DIM, error, size);
    if (!plain) return NULL;
    uint16_t values[HEAD_DIM], shuffled[HEAD_DIM];
    int map[HEAD_DIM];
    permuted_index(map);
    if (!h3_gpu_tensor_read_bf16(plain, values, HEAD_DIM)) {
        h3_gpu_tensor_free(plain);
        return NULL;
    }
    for (int index = 0; index < HEAD_DIM; index++) shuffled[index] = values[map[index]];
    h3_gpu_tensor_free(plain);
    return h3_gpu_tensor_from_bf16(z->gpu, shuffled, HEAD_DIM);
}

/* Reads the int8 matrix and its per-row scales, and the ConvRot group beside
 * them. Rows may be shuffled on the way through: the scales are per output
 * row so they shuffle identically, and ConvRot rotates along the *input*
 * dimension so it is undisturbed. */
/* `rows` and `columns` are the logical [output][input] shape; the package
 * stores the matrix transposed so the GPU reads it coalesced, so that is what
 * the loader asks for. Rows of the logical matrix are columns of the stored
 * one, which is why the head permutation below works on the stored layout
 * rather than on rows. */
static int load_linear(zimage_gpu *z, const char *name, uint64_t rows,
                       uint64_t columns, h3_gpu_tensor **weight,
                       h3_gpu_tensor **scales, uint32_t *group,
                       int permute_heads, char *error, size_t size) {
    /* Loaded apart rather than through h3_weight_load_i8_linear, which infers
     * the scale count from the matrix's first dimension. That held while the
     * matrix was [output][input]; the package stores it transposed now, and
     * the scales are still one per *output* channel, so the two shapes no
     * longer share an axis. */
    const uint64_t stored[2] = {columns, rows};
    *weight = h3_weight_load_i8(z->store, z->gpu, name, 2, stored, error, size);
    if (!*weight) return 0;
    char scale_name[288];
    snprintf(scale_name, sizeof(scale_name), "%.*s.weight_scale",
             (int)(strlen(name) - strlen(".weight")), name);
    const uint64_t column_scale[2] = {rows, 1};
    const uint64_t flat_scale[1] = {rows};
    *scales = h3_weight_load_f32(z->store, z->gpu, scale_name, 2, column_scale,
                                 error, size);
    if (!*scales)
        *scales = h3_weight_load_f32(z->store, z->gpu, scale_name, 1, flat_scale,
                                     error, size);
    if (!*scales) return 0;
    if (!h3_weight_i8_linear_convrot_group(z->store, name, group, error, size))
        return 0;
    if (!permute_heads) return 1;

    const size_t count = (size_t)rows * columns;
    int8_t *values = malloc(count);
    int8_t *shuffled = malloc(count);
    float *scale_values = malloc(rows * sizeof(float));
    float *scale_shuffled = malloc(rows * sizeof(float));
    if (!values || !shuffled || !scale_values || !scale_shuffled ||
        !h3_gpu_tensor_read_i8(*weight, values, count) ||
        !h3_gpu_tensor_read_f32(*scales, scale_values, rows)) {
        free(values); free(shuffled); free(scale_values); free(scale_shuffled);
        return fail(error, size, "cannot read %s back to permute it", name);
    }
    int map[HEAD_DIM];
    permuted_index(map);
    /* Only q and k — the first two of the three DIM-row blocks. v keeps its
     * order, which is what leaves the attention output needing no fix-up. */
    /* Output channels are the minor axis now, so the permutation moves
     * elements within each stored row rather than moving whole rows. */
    memcpy(shuffled, values, count);
    for (size_t k = 0; k < columns; k++)
        for (int block = 0; block < 2; block++)
            for (int head = 0; head < HEADS; head++)
                for (int index = 0; index < HEAD_DIM; index++) {
                    const size_t to = (size_t)block * DIM + head * HEAD_DIM + index;
                    const size_t from = (size_t)block * DIM + head * HEAD_DIM + map[index];
                    shuffled[k * rows + to] = values[k * rows + from];
                }
    for (int block = 0; block < 2; block++)
        for (int head = 0; head < HEADS; head++)
            for (int index = 0; index < HEAD_DIM; index++) {
                const size_t to = (size_t)block * DIM + head * HEAD_DIM + index;
                const size_t from = (size_t)block * DIM + head * HEAD_DIM + map[index];
                scale_shuffled[to] = scale_values[from];
            }
    for (size_t row = (size_t)2 * DIM; row < rows; row++)
        scale_shuffled[row] = scale_values[row];

    h3_gpu_tensor_free(*weight);
    h3_gpu_tensor_free(*scales);
    *weight = h3_gpu_tensor_from_i8(z->gpu, shuffled, count);
    *scales = h3_gpu_tensor_from_f32(z->gpu, scale_shuffled, rows);
    free(values); free(shuffled); free(scale_values); free(scale_shuffled);
    if (!*weight || !*scales)
        return fail(error, size, "cannot upload the permuted %s", name);
    return 1;
}

/* w1 and w3 into one [2 * FFN, DIM], because h3_swiglu_f32 reads gate and up
 * from a single row. */
static int load_fused_mlp(zimage_gpu *z, const char *prefix,
                          zimage_gpu_block *block, char *error, size_t size) {
    char name[256];
    h3_gpu_tensor *w1 = NULL, *w1_scales = NULL, *w3 = NULL, *w3_scales = NULL;
    uint32_t w1_group = 0, w3_group = 0;
    snprintf(name, sizeof(name), "%sfeed_forward.w1.weight", prefix);
    if (!load_linear(z, name, FFN, DIM, &w1, &w1_scales, &w1_group, 0, error, size))
        return 0;
    snprintf(name, sizeof(name), "%sfeed_forward.w3.weight", prefix);
    if (!load_linear(z, name, FFN, DIM, &w3, &w3_scales, &w3_group, 0, error, size))
        return 0;
    if (w1_group != w3_group)
        return fail(error, size, "%s w1 and w3 disagree on ConvRot group "
                                 "(%u against %u), so they cannot be fused",
                    prefix, w1_group, w3_group);

    /* Stored [input][output], so the two matrices interleave by row rather
     * than stacking: row k of the fused matrix is w1's row k followed by w3's. */
    const size_t half = (size_t)FFN * DIM;
    int8_t *fused = malloc(half * 2);
    int8_t *first = malloc(half), *second = malloc(half);
    float *scales = malloc((size_t)FFN * 2 * sizeof(float));
    if (!fused || !first || !second || !scales ||
        !h3_gpu_tensor_read_i8(w1, first, half) ||
        !h3_gpu_tensor_read_i8(w3, second, half) ||
        !h3_gpu_tensor_read_f32(w1_scales, scales, FFN) ||
        !h3_gpu_tensor_read_f32(w3_scales, scales + FFN, FFN)) {
        free(fused); free(first); free(second); free(scales);
        return fail(error, size, "cannot fuse %s w1 and w3", prefix);
    }
    for (size_t k = 0; k < DIM; k++) {
        memcpy(fused + k * FFN * 2, first + k * FFN, FFN);
        memcpy(fused + k * FFN * 2 + FFN, second + k * FFN, FFN);
    }
    free(first); free(second);
    h3_gpu_tensor_free(w1); h3_gpu_tensor_free(w1_scales);
    h3_gpu_tensor_free(w3); h3_gpu_tensor_free(w3_scales);
    block->w13 = h3_gpu_tensor_from_i8(z->gpu, fused, half * 2);
    block->w13_scales = h3_gpu_tensor_from_f32(z->gpu, scales, FFN * 2);
    block->w13_group = w1_group;
    free(fused); free(scales);
    if (!block->w13 || !block->w13_scales)
        return fail(error, size, "cannot upload the fused %s MLP", prefix);
    return 1;
}

static int load_block(zimage_gpu *z, const char *prefix,
                      zimage_gpu_block *block, int modulated,
                      char *error, size_t size) {
    char name[256];
    block->modulated = modulated;
#define VECTOR(field, suffix, width) do {                                     \
        snprintf(name, sizeof(name), "%s" suffix, prefix);                    \
        block->field = load_vector(z, name, width, error, size);              \
        if (!block->field) return 0;                                          \
    } while (0)
    VECTOR(attention_norm1, "attention_norm1.weight", DIM);
    VECTOR(attention_norm2, "attention_norm2.weight", DIM);
    VECTOR(ffn_norm1, "ffn_norm1.weight", DIM);
    VECTOR(ffn_norm2, "ffn_norm2.weight", DIM);
#undef VECTOR
    snprintf(name, sizeof(name), "%sattention.q_norm.weight", prefix);
    block->q_norm = permute_norm(z, name, error, size);
    snprintf(name, sizeof(name), "%sattention.k_norm.weight", prefix);
    block->k_norm = permute_norm(z, name, error, size);
    if (!block->q_norm || !block->k_norm)
        return fail(error, size, "%s QK norms", prefix);

    snprintf(name, sizeof(name), "%sattention.qkv.weight", prefix);
    if (!load_linear(z, name, (uint64_t)DIM * 3, DIM, &block->qkv,
                     &block->qkv_scales, &block->qkv_group, 1, error, size))
        return 0;
    snprintf(name, sizeof(name), "%sattention.out.weight", prefix);
    if (!load_linear(z, name, DIM, DIM, &block->out, &block->out_scales,
                     &block->out_group, 0, error, size))
        return 0;
    if (!load_fused_mlp(z, prefix, block, error, size)) return 0;
    snprintf(name, sizeof(name), "%sfeed_forward.w2.weight", prefix);
    if (!load_linear(z, name, DIM, FFN, &block->w2, &block->w2_scales,
                     &block->w2_group, 0, error, size))
        return 0;

    if (modulated) {
        snprintf(name, sizeof(name), "%sadaLN_modulation.0.weight", prefix);
        if (!load_linear(z, name, (uint64_t)DIM * 4, ADALN, &block->adaln,
                         &block->adaln_scales, &block->adaln_group, 0,
                         error, size))
            return 0;
        snprintf(name, sizeof(name), "%sadaLN_modulation.0.bias", prefix);
        block->adaln_bias = load_vector(z, name, (uint64_t)DIM * 4, error, size);
        if (!block->adaln_bias) return 0;
    }
    return 1;
}

static void free_block(zimage_gpu_block *block) {
    h3_gpu_tensor *tensors[] = {
        block->qkv, block->qkv_scales, block->out, block->out_scales,
        block->w13, block->w13_scales, block->w2, block->w2_scales,
        block->adaln, block->adaln_scales, block->adaln_bias,
        block->q_norm, block->k_norm, block->attention_norm1,
        block->attention_norm2, block->ffn_norm1, block->ffn_norm2,
    };
    for (size_t index = 0; index < sizeof(tensors) / sizeof(tensors[0]); index++)
        h3_gpu_tensor_free(tensors[index]);
    memset(block, 0, sizeof(*block));
}

zimage_gpu *zimage_gpu_create(const char *shaders, const char *weights,
                              int max_tokens, char *error, size_t size) {
    if (!h3_gpu_prepare(shaders, error, size)) return NULL;
    zimage_gpu *z = calloc(1, sizeof(*z));
    if (!z) { fail(error, size, "out of memory"); return NULL; }
    z->max_tokens = max_tokens;
    z->gpu = h3_gpu_create(shaders, error, size);
    z->store = h3_weight_store_open(weights, error, size);
    if (!z->gpu || !z->store) { zimage_gpu_release(z); return NULL; }

    const size_t wide = (size_t)max_tokens * DIM;
#define NEW32(field, elements) do {                                           \
        z->field = h3_gpu_tensor_new_f32(z->gpu, (elements));                 \
        if (!z->field) { fail(error, size, "out of GPU memory sizing " #field); \
                         zimage_gpu_release(z); return NULL; }                \
    } while (0)
#define NEW(field, elements) do {                                             \
        z->field = h3_gpu_tensor_new_bf16(z->gpu, (elements));                \
        if (!z->field) { fail(error, size, "out of GPU memory sizing " #field); \
                         zimage_gpu_release(z); return NULL; }                \
    } while (0)
#define NEW32(field, elements) do {                                           \
        z->field = h3_gpu_tensor_new_f32(z->gpu, (elements));                 \
        if (!z->field) { fail(error, size, "out of GPU memory sizing " #field); \
                         zimage_gpu_release(z); return NULL; }                \
    } while (0)
    NEW(hidden, wide); NEW(normed, wide); NEW(branch, wide); NEW(heads, wide);
    NEW(qkv, wide * 3); NEW(query, wide); NEW(key, wide); NEW(value, wide);
    NEW(ff, (size_t)max_tokens * FFN * 2);
    NEW(activated, (size_t)max_tokens * FFN);
    NEW(modulation, (size_t)SLOTS * DIM);
    NEW(mod_linear, (size_t)DIM * 4);
    NEW(adaln_input, ADALN);
    NEW32(query32, wide); NEW32(key32, wide);
    NEW32(value32, wide); NEW32(heads32, wide);
    NEW(rope_cos, (size_t)max_tokens * ROPE_HALF);
    NEW(rope_sin, (size_t)max_tokens * ROPE_HALF);
#undef NEW

    /* One modulation row shared by every token, so the row map is all zeros.
     * The unit table is how the two unmodulated refiners get a plain residual
     * add without a bf16 scale_add: slot 0 is zero and slot 1 is one, so the
     * same h3_gpu_gate_bf16 the modulated blocks call adds the branch
     * unweighted. */
    uint32_t *zeros = calloc((size_t)max_tokens, sizeof(uint32_t));
    uint16_t *unit = calloc((size_t)UNIT_SLOTS * DIM, sizeof(uint16_t));
    if (!zeros || !unit) { free(zeros); free(unit); zimage_gpu_release(z); return NULL; }
    for (int index = 0; index < DIM; index++) unit[DIM + index] = narrow(1.0f);
    z->row_map = h3_gpu_tensor_from_u32(z->gpu, zeros, (size_t)max_tokens);
    z->ones = h3_gpu_tensor_from_bf16(z->gpu, unit, (size_t)UNIT_SLOTS * DIM);
    free(zeros); free(unit);
    if (!z->row_map || !z->ones) { zimage_gpu_release(z); return NULL; }

    static const char *const names[BLOCKS] = {
        "noise_refiner.0.", "noise_refiner.1.",
        "context_refiner.0.", "context_refiner.1.",
        "layers.0.", "layers.1.", "layers.2.", "layers.3.", "layers.4.",
        "layers.5.", "layers.6.", "layers.7.", "layers.8.", "layers.9.",
        "layers.10.", "layers.11.", "layers.12.", "layers.13.", "layers.14.",
        "layers.15.", "layers.16.", "layers.17.", "layers.18.", "layers.19.",
        "layers.20.", "layers.21.", "layers.22.", "layers.23.", "layers.24.",
        "layers.25.", "layers.26.", "layers.27.", "layers.28.", "layers.29.",
    };
    for (int index = 0; index < BLOCKS; index++) {
        /* Only the two context refiners are unmodulated. */
        const int modulated = !(index == 2 || index == 3);
        if (!load_block(z, names[index], &z->blocks[index], modulated,
                        error, size)) {
            zimage_gpu_release(z);
            return NULL;
        }
    }
    return z;
}

void zimage_gpu_release(zimage_gpu *z) {
    if (!z) return;
    for (int index = 0; index < BLOCKS; index++) free_block(&z->blocks[index]);
    h3_gpu_tensor *tensors[] = {
        z->hidden, z->normed, z->branch, z->heads, z->qkv, z->query, z->key,
        z->value, z->ff, z->activated, z->modulation, z->mod_linear,
        z->query32, z->key32, z->value32, z->heads32,
        z->adaln_input, z->row_map, z->rope_cos, z->rope_sin, z->ones,
    };
    for (size_t index = 0; index < sizeof(tensors) / sizeof(tensors[0]); index++)
        h3_gpu_tensor_free(tensors[index]);
    if (z->store) h3_weight_store_free(z->store);
    if (z->gpu) h3_gpu_free(z->gpu);
    free(z);
}

double zimage_gpu_seconds(const zimage_gpu *z) { return z ? z->seconds : 0.0; }

#define OP(call, label) do {                                                  \
    if (!(call)) return fail(error, size, "%s failed", label);                 \
} while (0)

/* A ConvRot linear: rotate the activation in place, then the int8 product.
 * A group of zero means the checkpoint stored the matrix unrotated. */
static int rotated_linear(zimage_gpu *z, h3_gpu_tensor *output,
                          h3_gpu_tensor *input, const zimage_gpu_block *block,
                          h3_gpu_tensor *weight, h3_gpu_tensor *scales,
                          h3_gpu_tensor *bias, uint32_t group, uint32_t rows,
                          uint32_t inputs, uint32_t outputs,
                          const char *label, char *error, size_t size) {
    (void)block;
    if (group)
        OP(h3_gpu_convrot_bf16(z->gpu, input, input, rows, inputs, group), label);
    OP(h3_gpu_linear_i8_weight_bf16_square(z->gpu, output, input, weight,
                                               scales, bias, rows, inputs,
                                               outputs), label);
    return 1;
}

static int run_block(zimage_gpu *z, const zimage_gpu_block *block,
                     uint32_t rows, char *error, size_t size) {
    if (block->modulated) {
        OP(h3_gpu_copy_bf16(z->gpu, z->normed, 0, z->adaln_input, 0, ADALN),
           "modulation input");
        if (!rotated_linear(z, z->mod_linear, z->normed, block, block->adaln,
                            block->adaln_scales, block->adaln_bias,
                            block->adaln_group, 1, ADALN, DIM * 4,
                            "adaLN modulation", error, size)) return 0;
        OP(h3_gpu_zimage_modulation_bf16(z->gpu, z->modulation, z->mod_linear, DIM),
           "modulation layout");
        /* norm, scale by 1 + scale_msa, no shift — the zero slot supplies it */
        OP(h3_gpu_adaln_bf16(z->gpu, z->normed, z->hidden, block->attention_norm1,
                            z->modulation, z->row_map, rows, DIM, SLOTS,
                            SHIFT_SLOT, SCALE_MSA, NORM_EPS), "attention adaLN");
    } else {
        OP(h3_gpu_rms_norm_bf16(z->gpu, z->normed, z->hidden,
                               block->attention_norm1, rows, DIM, NORM_EPS),
           "attention norm");
    }

    if (!rotated_linear(z, z->qkv, z->normed, block, block->qkv,
                        block->qkv_scales, NULL, block->qkv_group, rows,
                        DIM, DIM * 3, "QKV", error, size)) return 0;
    OP(h3_gpu_qkv_rope_bf16(z->gpu, z->query, z->key, z->value, z->qkv,
                                block->q_norm, block->k_norm, z->rope_cos,
                                z->rope_sin, rows, HEADS, HEAD_DIM, ROPE_HALF,
                                NORM_EPS), "QK norm and RoPE");
    /* Casting costs time linear in the rows while the saving is quadratic in
     * them, so the trade turns over with the canvas: measured, bf16 wins by
     * 2.5% at 320 tokens, f32 by 3.7% at 1056 and by 9% at 4128. */
    if (rows <= ZIMAGE_F32_ATTENTION_ROWS) {
        OP(h3_gpu_sdpa_bf16(z->gpu, z->heads, z->query, z->key, z->value, rows,
                            HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
           "attention");
    } else {   /* Attention runs f32 even though everything around it is bf16.
         *
         * MPSGraph's bf16 SDPA reaches 0.97 TFLOP/s on this shape where its
         * f32 path reaches 2.21, and casting three inputs up and the result
         * back still comes out ahead: interleaved and with the order swapped,
         * every f32 forward measured between 17.6 and 21.2 s against every
         * bf16 one between 22.5 and 23.0. Roughly a tenth off the forward at
         * the median.
         *
         * An earlier test said the opposite. It ran the two arms as sequential
         * blocks while the machine was drifting about 20%, which is the same
         * size as the effect — so it measured the drift. Interleave and swap
         * the order before believing any comparison here. */
        const uint32_t span = rows * DIM;
        OP(h3_gpu_cast_bf16_to_f32(z->gpu, z->query32, z->query, span), "cast q");
        OP(h3_gpu_cast_bf16_to_f32(z->gpu, z->key32, z->key, span), "cast k");
        OP(h3_gpu_cast_bf16_to_f32(z->gpu, z->value32, z->value, span), "cast v");
        OP(h3_gpu_sdpa_f32(z->gpu, z->heads32, z->query32, z->key32, z->value32,
                           rows, HEADS, HEAD_DIM,
                           1.0f / sqrtf((float)HEAD_DIM)), "attention");
        OP(h3_gpu_cast_f32_to_bf16(z->gpu, z->heads, z->heads32, span), "cast out");
    }
    if (!rotated_linear(z, z->branch, z->heads, block, block->out,
                        block->out_scales, NULL, block->out_group, rows,
                        DIM, DIM, "attention output", error, size)) return 0;

    /* Sandwich norm: the second norm sits on the branch, inside the residual. */
    OP(h3_gpu_rms_norm_bf16(z->gpu, z->branch, z->branch, block->attention_norm2,
                           rows, DIM, NORM_EPS), "attention post-norm");
    if (block->modulated)
        OP(h3_gpu_gate_bf16(z->gpu, z->hidden, z->hidden, z->branch, z->modulation,
                           z->row_map, rows, DIM, SLOTS, GATE_MSA),
           "attention gate");
    else
        OP(h3_gpu_gate_bf16(z->gpu, z->hidden, z->hidden, z->branch, z->ones,
                            z->row_map, rows, DIM, UNIT_SLOTS, UNIT_GATE),
           "attention residual");

    if (block->modulated)
        OP(h3_gpu_adaln_bf16(z->gpu, z->normed, z->hidden, block->ffn_norm1,
                            z->modulation, z->row_map, rows, DIM, SLOTS,
                            SHIFT_SLOT, SCALE_MLP, NORM_EPS), "MLP adaLN");
    else
        OP(h3_gpu_rms_norm_bf16(z->gpu, z->normed, z->hidden, block->ffn_norm1,
                               rows, DIM, NORM_EPS), "MLP norm");
    if (!rotated_linear(z, z->ff, z->normed, block, block->w13, block->w13_scales,
                        NULL, block->w13_group, rows, DIM, FFN * 2,
                        "MLP input", error, size)) return 0;
    OP(h3_gpu_swiglu_bf16(z->gpu, z->activated, z->ff, rows, FFN), "SwiGLU");
    if (!rotated_linear(z, z->branch, z->activated, block, block->w2,
                        block->w2_scales, NULL, block->w2_group, rows, FFN,
                        DIM, "MLP output", error, size)) return 0;
    OP(h3_gpu_rms_norm_bf16(z->gpu, z->branch, z->branch, block->ffn_norm2,
                           rows, DIM, NORM_EPS), "MLP post-norm");
    if (block->modulated)
        OP(h3_gpu_gate_bf16(z->gpu, z->hidden, z->hidden, z->branch, z->modulation,
                           z->row_map, rows, DIM, SLOTS, GATE_MLP), "MLP gate");
    else
        OP(h3_gpu_gate_bf16(z->gpu, z->hidden, z->hidden, z->branch, z->ones,
                            z->row_map, rows, DIM, UNIT_SLOTS, UNIT_GATE),
           "MLP residual");
    return 1;
}

static int narrow_into(h3_gpu_tensor *target, const float *values, size_t count) {
    uint16_t *narrowed = malloc(count * sizeof(uint16_t));
    if (!narrowed) return 0;
    for (size_t index = 0; index < count; index++) narrowed[index] = narrow(values[index]);
    const int ok = h3_gpu_tensor_write_bf16(target, narrowed, count);
    free(narrowed);
    return ok;
}

static int widen_from(const h3_gpu_tensor *source, float *values, size_t count) {
    uint16_t *narrowed = malloc(count * sizeof(uint16_t));
    if (!narrowed) return 0;
    const int ok = h3_gpu_tensor_read_bf16(source, narrowed, count);
    for (size_t index = 0; index < count && ok; index++) {
        union { uint32_t bits; float number; } cast;
        cast.bits = (uint32_t)narrowed[index] << 16;
        values[index] = cast.number;
    }
    free(narrowed);
    return ok;
}

static int upload_rope(zimage_gpu *z, const float *cosines, const float *sines,
                       int tokens, char *error, size_t size) {
    OP(narrow_into(z->rope_cos, cosines, (size_t)tokens * ROPE_HALF),
       "rope cosines");
    OP(narrow_into(z->rope_sin, sines, (size_t)tokens * ROPE_HALF),
       "rope sines");
    return 1;
}

int zimage_gpu_refine_context(zimage_gpu *z, float *caption, int tokens,
                              const float *cosines, const float *sines,
                              char *error, size_t size) {
    if (tokens > z->max_tokens)
        return fail(error, size, "%d caption tokens exceeds the %d this was "
                                 "sized for", tokens, z->max_tokens);
    if (!upload_rope(z, cosines, sines, tokens, error, size)) return 0;
    OP(narrow_into(z->hidden, caption, (size_t)tokens * DIM), "caption upload");
    if (!h3_gpu_begin(z->gpu)) return fail(error, size, "cannot begin");
    for (int index = 2; index < 4; index++)
        if (!run_block(z, &z->blocks[index], (uint32_t)tokens, error, size))
            return 0;
    if (!h3_gpu_submit(z->gpu)) return fail(error, size, "cannot submit");
    OP(widen_from(z->hidden, caption, (size_t)tokens * DIM), "caption readback");
    return 1;
}

int zimage_gpu_forward(zimage_gpu *z, float *unified, int image_tokens,
                       int sequence, const float *adaln_input,
                       const float *cosines, const float *sines,
                       char *error, size_t size) {
    if (sequence > z->max_tokens)
        return fail(error, size, "%d tokens exceeds the %d this was sized for",
                    sequence, z->max_tokens);
    if (!upload_rope(z, cosines, sines, sequence, error, size)) return 0;
    OP(narrow_into(z->adaln_input, adaln_input, ADALN), "conditioning upload");
    OP(narrow_into(z->hidden, unified, (size_t)sequence * DIM), "sequence upload");

    if (!h3_gpu_begin(z->gpu)) return fail(error, size, "cannot begin");
    /* The noise refiners see the image rows alone, which sit first, so the
     * same buffer and the same rope table serve both passes. */
    for (int index = 0; index < 2; index++)
        if (!run_block(z, &z->blocks[index], (uint32_t)image_tokens, error, size))
            return 0;
    for (int index = 4; index < BLOCKS; index++)
        if (!run_block(z, &z->blocks[index], (uint32_t)sequence, error, size))
            return 0;
    if (!h3_gpu_submit(z->gpu)) return fail(error, size, "cannot submit");

    h3_gpu_stats stats;
    if (h3_gpu_get_stats(z->gpu, &stats)) z->seconds = stats.gpu_seconds;
    OP(widen_from(z->hidden, unified, (size_t)sequence * DIM), "sequence readback");
    return 1;
}

h3_gpu *zimage_gpu_device(zimage_gpu *z) { return z ? z->gpu : NULL; }
