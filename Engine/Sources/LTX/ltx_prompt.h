/* LTX-2.5 prompt-token preparation after Gemma's bare tokenizer runs. */
#ifndef LTX_PROMPT_H
#define LTX_PROMPT_H

#include <stddef.h>
#include <stdint.h>

/* Gemma 4's <bos> token. ComfyUI supplies this as its `start` token even
 * though the tokenizer JSON embedded in the LTX checkpoint has a no-op
 * post-processor. */
#define LTX_GEMMA_BOS_TOKEN_ID UINT32_C(2)

/* Copy the bare tokenizer output into `output`, adding the leading Gemma BOS
 * exactly once. Truncation keeps the front of the fully prepared sequence,
 * matching the reference pipeline. Returns the number of output tokens, or
 * zero when no output capacity was supplied. */
size_t ltx_prompt_tokens_with_bos(const uint32_t *bare, size_t bare_count,
                                  int32_t *output, size_t output_capacity);

#endif
