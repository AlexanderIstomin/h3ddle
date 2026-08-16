#ifndef SA3_TEXT_H
#define SA3_TEXT_H

#include <stddef.h>
#include <stdint.h>

/* T5Gemma: the encoder half of a Gemma-based encoder-decoder, used here only
 * to turn a prompt into the 768-wide conditioning the transformer cross-
 * attends to.
 *
 * It runs once per generation rather than once per sampler step, so it stays
 * on the CPU: the whole pass costs a fraction of a second against the seconds
 * the sampler spends, and keeping it here avoids a second precision story.
 *
 * Two Gemma details differ from the usual transformer and are easy to miss:
 * the normalisation scales by (1 + weight) rather than weight, and attention
 * logits are squashed through a tanh soft cap before the softmax. */

typedef struct sa3_text sa3_text;

#define SA3_TEXT_WIDTH 768
#define SA3_TEXT_MAX_TOKENS 256

sa3_text *sa3_text_load(const char *path, char *error, size_t error_size);
void sa3_text_free(sa3_text *text);

/* Encodes `count` token ids into [SA3_TEXT_MAX_TOKENS, 768].
 *
 * Positions beyond `count` are padding: they are masked out of attention and
 * the caller is expected to overwrite them, which sa3_text_condition does. */
int sa3_text_encode(sa3_text *text, const uint32_t *ids, int count,
                    float *embeddings, char *error, size_t error_size);

#endif
