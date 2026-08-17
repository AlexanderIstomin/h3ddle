/* Stage 2: does Z-Image's text encoder run on the block we already have?
 *
 * Qwen3-4B is the third set of shapes through qwen_block_forward — after H3's
 * own text encoder at 5120 wide and the TTS talker at 1024. If that block is
 * genuinely shape-general, this file is a loader and a comparison, and the
 * only new code in the whole of stage 2.
 *
 * Checked at three depths rather than only at the end, because a single
 * end-of-stack comparison cannot say *where* a discrepancy began, and the
 * error a wrong rope or a transposed projection produces at layer 0 looks much
 * like the error thirty-six layers of accumulation produce on their own.
 *
 *   ./zimage_encoder_check <package-dir> <golden.safetensors>
 */
#include "qwen_block.h"
#include "qwen_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH     2560
#define LAYERS    36
#define HEADS     32
#define KV_HEADS  8
#define HEAD_DIM  128
#define FFN       9728
#define VOCAB     151936
#define THETA     1000000.0f
#define EPSILON   1e-6f

/* bf16's own floor: one rounding of an f32 answer. Anything at this level is
 * the format, not a fault. */
#define BF16_FLOOR 1.5e-3

static void compare(const char *label, const float *ours, const float *theirs,
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
    printf("  %-18s RMS rel %8.2e  (%4.1fx the bf16 floor)   worst %8.2e\n",
           label, relative, relative / BF16_FLOOR, worst);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <package-dir> <golden.safetensors>\n", argv[0]);
        return 2;
    }
    char error[512] = {0};
    char path[1200];
    snprintf(path, sizeof(path), "%s/text_encoder.safetensors", argv[1]);
    qwen_weights *weights = qwen_weights_open(path, error, sizeof(error));
    if (!weights) { fprintf(stderr, "encoder: %s\n", error); return 1; }
    qwen_weights *golden = qwen_weights_open(argv[2], error, sizeof(error));
    if (!golden) { fprintf(stderr, "golden: %s\n", error); return 1; }

    /* The golden carries the ids it was produced from, so the two sides cannot
     * silently diverge on tokenization — a question this stage is not trying
     * to answer. */
    const int64_t any[3] = {1, -1, WIDTH};
    const float *embedding_ref = qwen_weights_f32(golden, "embedding", 3, any,
                                                  error, sizeof(error));
    if (!embedding_ref) { fprintf(stderr, "golden: %s\n", error); return 1; }
    const int64_t ids_shape[2] = {1, -1};
    const float *ids_raw = qwen_weights_f32(golden, "input_ids", 2, ids_shape,
                                            error, sizeof(error));
    if (!ids_raw) { fprintf(stderr, "golden: %s\n", error); return 1; }
    /* Written as f32 because the reader is typed; the values are integers. */

    int tokens = 0;
    {   /* recover the length from the embedding tensor rather than trusting a
         * constant that would have to be edited with the prompt */
        FILE *probe = fopen(argv[2], "rb");
        if (!probe) return 1;
        uint64_t header_bytes = 0;
        if (fread(&header_bytes, 8, 1, probe) != 1) return 1;
        char *header = malloc(header_bytes + 1);
        if (fread(header, 1, header_bytes, probe) != header_bytes) return 1;
        header[header_bytes] = '\0';
        fclose(probe);
        const char *entry = strstr(header, "\"embedding\"");
        const char *shape = entry ? strstr(entry, "\"shape\":[1,") : NULL;
        if (shape) tokens = atoi(shape + strlen("\"shape\":[1,"));
        free(header);
    }
    if (tokens < 1) { fprintf(stderr, "cannot read the token count\n"); return 1; }
    printf("%d tokens, %d layers of %d wide\n\n", tokens, LAYERS, WIDTH);

    const qwen_block_config config = {
        .width = WIDTH, .heads = HEADS, .kv_heads = KV_HEADS,
        .head_dim = HEAD_DIM, .ffn = FFN,
        .rope_theta = THETA, .rms_epsilon = EPSILON,
    };

    /* ---- embed ---------------------------------------------------------- */
    const int64_t embed_shape[2] = {VOCAB, WIDTH};
    const uint16_t *table = qwen_weights_bf16(weights, "model.embed_tokens.weight",
                                              2, embed_shape, error, sizeof(error));
    if (!table) { fprintf(stderr, "%s\n", error); return 1; }
    float *x = malloc((size_t)tokens * WIDTH * sizeof(float));
    for (int token = 0; token < tokens; token++) {
        const uint16_t *row = table + (size_t)(int)ids_raw[token] * WIDTH;
        for (int channel = 0; channel < WIDTH; channel++)
            x[(size_t)token * WIDTH + channel] = qwen_widen(row[channel]);
    }
    printf("against the reference:\n");
    compare("embedding", x, embedding_ref, (size_t)tokens * WIDTH);

    /* ---- the stack ------------------------------------------------------ */
    qwen_rope rope;
    qwen_scratch scratch;
    if (!qwen_rope_init(&rope, &config, tokens) ||
        !qwen_scratch_init(&scratch, &config, tokens)) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    float *keys = calloc((size_t)tokens * KV_HEADS * HEAD_DIM, sizeof(float));
    float *values = calloc((size_t)tokens * KV_HEADS * HEAD_DIM, sizeof(float));
    if (!keys || !values) { fprintf(stderr, "out of memory\n"); return 1; }

    for (int layer = 0; layer < LAYERS; layer++) {
        qwen_block_weights block = {0};
#define TAKE(field, pattern, ndim, ...) do {                                   \
        const int64_t shape[] = __VA_ARGS__;                                   \
        block.field = qwen_weights_bf16_at(weights, pattern, layer, ndim,      \
                                           shape, error, sizeof(error));       \
        if (!block.field) { fprintf(stderr, "%s\n", error); return 1; }        \
    } while (0)
        TAKE(input_layernorm, "model.layers.%d.input_layernorm.weight", 1, {WIDTH});
        TAKE(post_attention_layernorm,
             "model.layers.%d.post_attention_layernorm.weight", 1, {WIDTH});
        TAKE(q_proj, "model.layers.%d.self_attn.q_proj.weight", 2,
             {HEADS * HEAD_DIM, WIDTH});
        TAKE(k_proj, "model.layers.%d.self_attn.k_proj.weight", 2,
             {KV_HEADS * HEAD_DIM, WIDTH});
        TAKE(v_proj, "model.layers.%d.self_attn.v_proj.weight", 2,
             {KV_HEADS * HEAD_DIM, WIDTH});
        TAKE(o_proj, "model.layers.%d.self_attn.o_proj.weight", 2,
             {WIDTH, HEADS * HEAD_DIM});
        TAKE(q_norm, "model.layers.%d.self_attn.q_norm.weight", 1, {HEAD_DIM});
        TAKE(k_norm, "model.layers.%d.self_attn.k_norm.weight", 1, {HEAD_DIM});
        TAKE(gate_proj, "model.layers.%d.mlp.gate_proj.weight", 2, {FFN, WIDTH});
        TAKE(up_proj, "model.layers.%d.mlp.up_proj.weight", 2, {FFN, WIDTH});
        TAKE(down_proj, "model.layers.%d.mlp.down_proj.weight", 2, {WIDTH, FFN});
#undef TAKE

        qwen_block_forward(&config, &block, &rope, x, tokens, 0, keys, values,
                           tokens, &scratch);

        if (layer == 0 || layer == 17) {
            char label[32];
            snprintf(label, sizeof(label), "block %02d", layer);
            const char *name = layer == 0 ? "block_00" : "block_17";
            const float *reference = qwen_weights_f32(golden, name, 3, any,
                                                      error, sizeof(error));
            if (!reference) { fprintf(stderr, "golden: %s\n", error); return 1; }
            compare(label, x, reference, (size_t)tokens * WIDTH);
        }
    }

    /* ---- the final norm, which the reference applies before handing the
     * hidden states on; measured at 0.000 RMS relative between the last
     * hidden state and the normed output, so it is not optional -------- */
    const int64_t norm_shape[1] = {WIDTH};
    const uint16_t *final_norm = qwen_weights_bf16(weights, "model.norm.weight",
                                                   1, norm_shape, error,
                                                   sizeof(error));
    if (!final_norm) { fprintf(stderr, "%s\n", error); return 1; }
    qwen_rms_norm(x, final_norm, x, WIDTH, tokens, EPSILON);

    const float *last = qwen_weights_f32(golden, "last_hidden_state", 3, any,
                                         error, sizeof(error));
    if (!last) { fprintf(stderr, "golden: %s\n", error); return 1; }
    compare("last_hidden_state", x, last, (size_t)tokens * WIDTH);

    printf("\nsqrt(%d layers) = %.1f, which is what pure accumulation would "
           "cost.\n", LAYERS, sqrt((double)LAYERS));

    free(x); free(keys); free(values);
    qwen_rope_release(&rope);
    qwen_scratch_release(&scratch);
    qwen_weights_close(weights);
    qwen_weights_close(golden);
    return 0;
}
