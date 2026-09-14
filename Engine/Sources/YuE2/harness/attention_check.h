/* Development-only CPU oracle for both cached causal and noncausal attention.
 */
#include "h3_gpu.h"
#include <math.h>
#include <stdint.h>
static uint16_t attention_bf(float f) {
  uint32_t n;
  memcpy(&n, &f, 4);
  n += 0x7fff + ((n >> 16) & 1);
  return n >> 16;
}
static float attention_fp(uint16_t b) {
  uint32_t n = (uint32_t)b << 16;
  float f;
  memcpy(&f, &n, 4);
  return f;
}
static int attention_check(const char *shaders) {
  char error[512] = {0};
  if (!h3_gpu_prepare(shaders, error, sizeof(error))) {
    fprintf(stderr, "%s\n", error);
    return 1;
  }
  h3_gpu *gpu = h3_gpu_create(shaders, error, sizeof(error));
  if (!gpu)
    return 1;
  const int rows = 3, qh = 4, kh = 2, dim = 128;
  const int lengths[] = {257, 7168, 9001, 24576};
  for (int test = 0; test < 4; test++) {
    int keys = lengths[test], qn = rows * qh * dim, kn = keys * kh * dim;
    uint16_t *q = malloc(qn * 2), *k = malloc(kn * 2), *v = malloc(kn * 2),
             *out = malloc(qn * 2);
    float *scores = malloc(keys * sizeof(float));
    for (int i = 0; i < qn; i++)
      q[i] = attention_bf(3 * sinf(i * .019f));
    for (int i = 0; i < kn; i++) {
      k[i] = attention_bf(3 * cosf(i * .013f));
      v[i] = attention_bf(sinf(i * .007f));
    }
    h3_gpu_tensor *qt = h3_gpu_tensor_new_bf16(gpu, qn),
                  *kt = h3_gpu_tensor_new_bf16(gpu, kn),
                  *vt = h3_gpu_tensor_new_bf16(gpu, kn),
                  *ot = h3_gpu_tensor_new_bf16(gpu, qn);
    if (!qt || !kt || !vt || !ot || !h3_gpu_tensor_write_bf16(qt, q, qn) ||
        !h3_gpu_tensor_write_bf16(kt, k, kn) ||
        !h3_gpu_tensor_write_bf16(vt, v, kn))
      return 1;
    for (int causal = 0; causal < 2; causal++) {
      float scale = 1 / sqrtf(dim);
      if (!h3_gpu_begin(gpu))
        return 1;
      int ok =
          causal ? h3_gpu_gqa_causal_cache_bf16(gpu, ot, qt, kt, vt, rows,
                                                keys - rows, qh, kh, dim, scale)
                 : h3_gpu_gqa_cross_bf16(gpu, ot, qt, kt, vt, rows, keys - rows,
                                         qh, kh, dim, scale);
      if (!ok || !h3_gpu_submit(gpu) || !h3_gpu_tensor_read_bf16(ot, out, qn)) {
        fprintf(stderr, "attention failed: %s\n", h3_gpu_error(gpu));
        return 1;
      }
      float worst = 0;
      for (int r = 0; r < rows; r++)
        for (int h = 0; h < qh; h++) {
          int count = causal ? keys - rows + r + 1 : keys;
          float maximum = -INFINITY;
          for (int t = 0; t < count; t++) {
            float dot = 0;
            for (int d = 0; d < dim; d++)
              dot = fmaf(attention_fp(q[(r * qh + h) * dim + d]) * scale,
                         attention_fp(k[(t * kh + h / 2) * dim + d]), dot);
            scores[t] = dot;
            maximum = fmaxf(maximum, dot);
          }
          double denominator = 0;
          for (int t = 0; t < count; t++) {
            scores[t] = expf(scores[t] - maximum);
            denominator += scores[t];
          }
          for (int d = 0; d < dim; d++) {
            double total = 0;
            for (int t = 0; t < count; t++)
              total += scores[t] * attention_fp(v[(t * kh + h / 2) * dim + d]);
            float actual = attention_fp(out[(r * qh + h) * dim + d]);
            if (!isfinite(actual))
              return 1;
            worst = fmaxf(worst, (float)fabs(actual - total / denominator));
          }
        }
      printf("attention keys=%d causal=%d max_abs=%g\n", keys, causal, worst);
      if (worst > .002f)
        return 1;
    }
    h3_gpu_tensor_free(qt);
    h3_gpu_tensor_free(kt);
    h3_gpu_tensor_free(vt);
    h3_gpu_tensor_free(ot);
    free(q);
    free(k);
    free(v);
    free(out);
    free(scores);
  }
  h3_gpu_free(gpu);
  return 0;
}
