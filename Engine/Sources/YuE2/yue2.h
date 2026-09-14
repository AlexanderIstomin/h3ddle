#ifndef H3DDLE_YUE2_H
#define H3DDLE_YUE2_H
#include <stddef.h>
#include <stdint.h>
/* Request-local native engine. All tensor inference uses Metal; callbacks
 * return zero to cancel. Output is interleaved 48 kHz stereo owned by caller.
 */
typedef int (*yue2_progress)(const char *, int, int, void *);
typedef struct yue2_model yue2_model;
yue2_model *yue2_load(const char *package, const char *shaders, char *error,
                      size_t error_size);
void yue2_free(yue2_model *model);
int yue2_generate(yue2_model *model, const uint32_t *text, int count,
                  double seconds, int steps, uint64_t seed, float **audio,
                  int *samples, yue2_progress progress, void *opaque,
                  char *error, size_t error_size);
void yue2_reset_cache(yue2_model *model);
void yue2_print_timing(yue2_model *model);
void yue2_set_linear_2d(yue2_model *model, int enabled);
/* Development parity entry points; no Python is needed by these functions. */
int yue2_forward(yue2_model *, const uint32_t *, int, int, int, float *, char *,
                 size_t);
int yue2_synthesize(yue2_model *, const uint32_t *, int, const uint32_t *, int,
                    const float *, int, float *, yue2_progress, void *, char *,
                    size_t);
int yue2_velocity(yue2_model *, const float *, int, float, float *, char *,
                  size_t);
int yue2_decode(yue2_model *, const float *, int, float **, int *,
                yue2_progress, void *, char *, size_t);
typedef struct yue2_tokenizer yue2_tokenizer;
yue2_tokenizer *yue2_tokenizer_load(const char *, char *, size_t);
void yue2_tokenizer_free(yue2_tokenizer *);
int yue2_tokenize(yue2_tokenizer *, const char *, uint32_t **, char *, size_t);
#endif
