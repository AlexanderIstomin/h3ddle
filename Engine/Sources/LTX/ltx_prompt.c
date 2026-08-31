#include "ltx_prompt.h"

size_t ltx_prompt_tokens_with_bos(const uint32_t *bare, size_t bare_count,
                                  int32_t *output, size_t output_capacity) {
    if (!output || !output_capacity || (!bare && bare_count)) return 0;

    size_t source = 0;
    size_t written = 0;
    output[written++] = (int32_t)LTX_GEMMA_BOS_TOKEN_ID;
    if (bare_count && bare[0] == LTX_GEMMA_BOS_TOKEN_ID) source = 1;

    while (source < bare_count && written < output_capacity)
        output[written++] = (int32_t)bare[source++];
    return written;
}
