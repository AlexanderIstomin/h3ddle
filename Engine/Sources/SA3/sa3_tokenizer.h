#ifndef SA3_TOKENIZER_H
#define SA3_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

/* The SentencePiece-style BPE tokenizer T5Gemma was trained with.
 *
 * It is separate from the engine's own tokenizer because the two disagree
 * about everything except the file format: this one rewrites spaces as U+2581
 * before merging and falls back to per-byte tokens for anything outside the
 * vocabulary, where the engine's applies NFC and a GPT-style split. */

typedef struct sa3_tokenizer sa3_tokenizer;

sa3_tokenizer *sa3_tokenizer_load(const char *tokenizer_json, char *error,
                                  size_t error_size);
void sa3_tokenizer_free(sa3_tokenizer *tokenizer);

/* Writes at most `capacity` ids and reports how many the prompt produced.
 * Prompts longer than the capacity are truncated, which matches the reference
 * rather than failing on a long description. */
int sa3_tokenizer_encode(const sa3_tokenizer *tokenizer, const char *utf8,
                         uint32_t *ids, int capacity, int *count,
                         char *error, size_t error_size);

#endif
