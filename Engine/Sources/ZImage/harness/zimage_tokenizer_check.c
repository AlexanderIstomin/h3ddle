/* Does the engine's tokenizer read Z-Image's tokenizer.json?
 *
 * It is a different file from the one the speech package ships — 11.4 MB
 * against 4.76 MB, different digest — so the fact that both are called
 * "Qwen2Tokenizer" settles nothing. The special tokens are the part worth
 * checking: the caption arrives wrapped in a chat template, so <|im_start|>
 * and <|im_end|> have to come back as single ids rather than being spelled out
 * as literal text, which is exactly what a tokenizer that loaded the vocab but
 * not the added-token table would do — silently, with a plausible id stream.
 *
 *   ./zimage_tokenizer_check <tokenizer.json> <expected-ids.txt>
 */
#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Exactly what apply_chat_template renders for the prompt below: no think
 * block is emitted despite enable_thinking=True, because nothing is being
 * generated. */
static const char *TEMPLATED =
    "<|im_start|>user\n"
    "A rain-slicked Tokyo alley at night, neon reflected in the puddles."
    "<|im_end|>\n<|im_start|>assistant\n";

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <tokenizer.json> <expected-ids.txt>\n", argv[0]);
        return 2;
    }
    char error[512] = {0};
    h3_tokenizer *tokenizer = h3_tokenizer_load(argv[1], error, sizeof(error));
    if (!tokenizer) { fprintf(stderr, "load: %s\n", error); return 1; }

    uint32_t *ids = NULL;
    size_t count = 0;
    if (!h3_tokenizer_encode(tokenizer, TEMPLATED, 0, &ids, &count,
                             error, sizeof(error))) {
        fprintf(stderr, "encode: %s\n", error);
        return 1;
    }

    FILE *expected = fopen(argv[2], "r");
    if (!expected) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    uint32_t theirs[512];
    size_t wanted = 0;
    while (wanted < 512 && fscanf(expected, "%u", &theirs[wanted]) == 1) wanted++;
    fclose(expected);

    printf("ours %zu ids, reference %zu ids\n", count, wanted);
    if (count != wanted) {
        printf("  ours: ");
        for (size_t index = 0; index < count; index++) printf("%u ", ids[index]);
        printf("\n");
        return 1;
    }
    size_t wrong = 0;
    for (size_t index = 0; index < count; index++)
        if (ids[index] != theirs[index]) {
            printf("  [%zu] ours %u, reference %u\n", index, ids[index], theirs[index]);
            wrong++;
        }
    if (wrong) printf("  %zu ids differ\n", wrong);
    else printf("  identical\n");

    char *round = h3_tokenizer_decode(tokenizer, ids, count, error, sizeof(error));
    if (round) {
        printf("round trip %s\n", strcmp(round, TEMPLATED) == 0 ? "exact" : "DIFFERS");
        if (strcmp(round, TEMPLATED) != 0) printf("  got: %s\n", round);
        free(round);
    } else {
        printf("decode: %s\n", error);
    }

    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    return wrong ? 1 : 0;
}
