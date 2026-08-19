/* The S3-DiT's thirty-four blocks on the GPU.
 *
 * Only the blocks. The embedders, the head, the sampler and the VAE stay on
 * the CPU path that already matches its reference: together they are well
 * under one percent of the arithmetic, and moving them too would double the
 * new code and the places a fault could hide for no measurable return. The
 * sequence crosses the bus once per step in each direction, which at 4128
 * tokens is 63 MB and a couple of milliseconds.
 *
 * The block maps onto kernels the engine already has. Two logical layouts are
 * consumed directly so loading never has to copy multi-gigabyte weights:
 *
 *  - h3_gpu_qkv_rope_bf16 pairs channel i with i + head_dim/2 where Z-Image
 *    pairs adjacent channels. Attention is invariant to permuting the head
 *    dimension of q and k together. The Z-Image QKV boundary reads adjacent
 *    pairs but writes that same half-paired order, preserving the established
 *    attention arithmetic without permuting weights;
 *  - h3_gpu_swiglu_bf16 reads one fused [gate | up] row, so w1 and w3 are
 *    presented as a virtual concatenation by a paired projection kernel.
 *
 * Both are row-wise, so both survive int8 quantisation untouched: the scales
 * are per output row and ConvRot rotates along the input dimension.
 */
#ifndef ZIMAGE_GPU_H
#define ZIMAGE_GPU_H

#include "h3_gpu.h"
#include "h3_weights.h"

#include <stddef.h>

typedef struct zimage_gpu zimage_gpu;

zimage_gpu *zimage_gpu_create(const char *shaders, const char *weights,
                              int max_tokens, char *error, size_t error_size);
void zimage_gpu_release(zimage_gpu *gpu);

/* The two unmodulated context_refiner blocks over the caption rows alone.
 * `caption` is [tokens][DIM] and is refined in place. */
int zimage_gpu_refine_context(zimage_gpu *gpu, float *caption, int tokens,
                              const float *cosines, const float *sines,
                              char *error, size_t error_size);

/* The two modulated noise_refiner blocks over the image rows, then the thirty
 * trunk blocks over the whole sequence. `unified` is [sequence][DIM] laid out
 * image rows first, and is advanced in place. */
int zimage_gpu_forward(zimage_gpu *gpu, float *unified, int image_tokens,
                       int sequence, const float *adaln_input,
                       const float *cosines, const float *sines,
                       char *error, size_t error_size);

/* Wall-clock seconds spent inside submitted command buffers, for reporting
 * what the port actually bought. */
double zimage_gpu_seconds(const zimage_gpu *gpu);
h3_gpu *zimage_gpu_device(zimage_gpu *gpu);

#endif
