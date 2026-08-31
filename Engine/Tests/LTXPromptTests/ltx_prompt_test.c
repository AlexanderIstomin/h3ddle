#include "ltx_prompt.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void require(int condition, const char *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
}

int main(void) {
    const uint32_t bare[] = {101, 202, 303};
    int32_t tokens[4] = {0};
    size_t count = ltx_prompt_tokens_with_bos(bare, 3, tokens, 4);
    require(count == 4, "BOS extends a bare LTX prompt");
    require(tokens[0] == (int32_t)LTX_GEMMA_BOS_TOKEN_ID &&
            tokens[1] == 101 && tokens[2] == 202 && tokens[3] == 303,
            "BOS precedes the unchanged prompt tokens");

    const uint32_t prefixed[] = {LTX_GEMMA_BOS_TOKEN_ID, 404, 505};
    count = ltx_prompt_tokens_with_bos(prefixed, 3, tokens, 4);
    require(count == 3, "an existing BOS is not duplicated");
    require(tokens[0] == (int32_t)LTX_GEMMA_BOS_TOKEN_ID &&
            tokens[1] == 404 && tokens[2] == 505,
            "an already-prefixed prompt remains unchanged");

    const uint32_t long_prompt[] = {11, 22, 33, 44};
    count = ltx_prompt_tokens_with_bos(long_prompt, 4, tokens, 3);
    require(count == 3, "the prepared prompt respects its conditioning span");
    require(tokens[0] == (int32_t)LTX_GEMMA_BOS_TOKEN_ID &&
            tokens[1] == 11 && tokens[2] == 22,
            "truncation keeps BOS and the front of the prompt");

    printf("ok: LTX prompt conditioning adds Gemma BOS exactly once\n");
    return 0;
}
