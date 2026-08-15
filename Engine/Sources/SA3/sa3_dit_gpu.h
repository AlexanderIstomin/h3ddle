#ifndef SA3_DIT_GPU_H
#define SA3_DIT_GPU_H

#include "sa3_dit.h"

#include <stddef.h>

/* Metal path for the SA3 transformer, sharing h3.c's GPU context so the two
 * models draw from one device, one shader library, and one allocator.
 *
 * The conditioning derived from the prompt is fixed for a whole generation,
 * so it is installed once with sa3_dit_gpu_set_context and reused by every
 * sampler step. Only the noise level changes between steps. */

typedef struct h3_gpu h3_gpu;
typedef struct sa3_dit_gpu sa3_dit_gpu;

sa3_dit_gpu *sa3_dit_gpu_load(h3_gpu *gpu, const char *path, char *error,
                              size_t error_size);
void sa3_dit_gpu_free(sa3_dit_gpu *dit);

/* Installs the prompt conditioning, [context_tokens, 768], and builds the
 * cross-attention keys and values every block will reuse. */
int sa3_dit_gpu_set_context(sa3_dit_gpu *dit, const float *context,
                            int context_tokens, char *error,
                            size_t error_size);

/* Reads the working sequence back after each block, [tokens, 1024] on the
 * host. Only for checking the Metal path against a reference: it forces a
 * submit and a download per block, so it is far slower than a plain run. */
typedef void (*sa3_dit_gpu_probe)(int block, const float *sequence,
                                  int tokens, void *opaque);
void sa3_dit_gpu_set_probe(sa3_dit_gpu *dit, sa3_dit_gpu_probe probe,
                           void *opaque);

/* One denoising evaluation against the installed context. Latents and
 * velocity are both [256, frames] channel-major. */
int sa3_dit_gpu_forward(sa3_dit_gpu *dit, const float *latents, int frames,
                        const float *global, float sigma, float *velocity,
                        char *error, size_t error_size);

#endif
