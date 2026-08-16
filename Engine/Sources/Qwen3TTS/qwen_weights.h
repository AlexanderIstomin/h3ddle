#ifndef QWEN_WEIGHTS_H
#define QWEN_WEIGHTS_H

#include <stddef.h>
#include <stdint.h>

/* Memory-mapped access to a converted Qwen3-TTS safetensors file.
 *
 * The talker alone is 1.5 GB, most of it a 151936 x 2048 text embedding that a
 * given utterance touches a few dozen rows of. Reading tensors into buffers
 * would spend seconds and gigabytes on weights that are never used, so this
 * maps the file and hands out pointers into the mapping — the same
 * demand-paged arrangement h3.c uses for H3's weights, where 72 GiB streams
 * with about 1.5 GiB resident.
 *
 * Pointers stay valid until qwen_weights_close. They point at the file's own
 * bytes, so bf16 stays bf16: callers widen as they read. */

typedef struct qwen_weights qwen_weights;

qwen_weights *qwen_weights_open(const char *path, char *error,
                                size_t error_size);
void qwen_weights_close(qwen_weights *weights);

/* Returns a pointer to `name`'s data, or NULL with `error` set.
 *
 * The expected shape is checked rather than trusted: pass the dimensions the
 * caller intends to index by, and a release that changes a layout fails at
 * load instead of reading past the end or, worse, producing plausible noise.
 * Pass -1 for a dimension that may be anything. */
const uint16_t *qwen_weights_bf16(const qwen_weights *weights, const char *name,
                                  int ndim, const int64_t *shape,
                                  char *error, size_t error_size);
const float *qwen_weights_f32(const qwen_weights *weights, const char *name,
                              int ndim, const int64_t *shape,
                              char *error, size_t error_size);

/* Formats `pattern` with `index` before looking the tensor up, for the layer
 * arrays that make up most of every subsystem. */
const uint16_t *qwen_weights_bf16_at(const qwen_weights *weights,
                                     const char *pattern, int index,
                                     int ndim, const int64_t *shape,
                                     char *error, size_t error_size);
const float *qwen_weights_f32_at(const qwen_weights *weights,
                                 const char *pattern, int index,
                                 int ndim, const int64_t *shape,
                                 char *error, size_t error_size);

/* The JSON the converter stored under "config" in the header, or NULL. */
const char *qwen_weights_config(const qwen_weights *weights);

#endif
