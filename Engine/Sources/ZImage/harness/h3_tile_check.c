/* Does the Z-Image tile work carry over to H3's video DiT?
 *
 * H3's DiT runs the same int8 GEMM through the 8x8 tile that Z-Image used
 * before the sweep, at its own widths — 5376 hidden, 56 heads, 14336 FFN. If
 * the geometry is what mattered rather than anything about Z-Image, the same
 * 64x40 tile and input-major storage should carry, and this says by how much
 * on H3's own four shapes.
 */
#include "h3_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define HIDDEN 5376
#define INNER  (56 * 128)
#define FFN    14336

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

typedef struct { const char *label; int in, out; } shape;

int main(int argc, char **argv) {
    char error[512] = {0};
    if (!h3_gpu_prepare(argv[1], error, sizeof(error))) return 1;
    h3_gpu *gpu = h3_gpu_create(argv[1], error, sizeof(error));
    if (!gpu) return 1;
    const int rows = argc > 2 ? atoi(argv[2]) : 4096;
    const shape shapes[] = {
        {"qkv", HIDDEN, INNER * 3},
        {"out", INNER, HIDDEN},
        {"fc1", HIDDEN, FFN * 2},
        {"fc2", FFN, HIDDEN},
    };
    printf("H3 DiT at %d rows\n\n%-6s %10s %10s %9s %10s\n", rows,
           "shape", "8x8 now", "64x40", "speedup", "TFLOP/s");
    double total_old = 0, total_new = 0;
    for (int s = 0; s < 4; s++) {
        const shape sh = shapes[s];
        const size_t in = (size_t)rows * sh.in, out = (size_t)rows * sh.out;
        const size_t weights = (size_t)sh.out * sh.in;
        uint16_t *host = calloc(in > out ? in : out, 2);
        int8_t *w = calloc(weights, 1), *wt = calloc(weights, 1);
        float *sc = calloc(sh.out, sizeof(float));
        for (size_t i = 0; i < weights; i++) w[i] = (int8_t)((i % 13) - 6);
        for (int r = 0; r < sh.out; r++)
            for (int c = 0; c < sh.in; c++)
                wt[(size_t)c * sh.out + r] = w[(size_t)r * sh.in + c];
        for (int i = 0; i < sh.out; i++) sc[i] = 0.01f;
        h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(gpu, host, in);
        h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, out);
        h3_gpu_tensor *rowmajor = h3_gpu_tensor_from_i8(gpu, w, weights);
        h3_gpu_tensor *kmajor = h3_gpu_tensor_from_i8(gpu, wt, weights);
        h3_gpu_tensor *scales = h3_gpu_tensor_from_f32(gpu, sc, sh.out);
        free(host); free(w); free(wt); free(sc);
        double old = 0, new = 0;
        /* Interleaved, because sequential blocks on this machine measure drift. */
        for (int pass = 0; pass < 3; pass++) {
            double began = now();
            if (!h3_gpu_begin(gpu)) return 1;
            for (int r = 0; r < 3; r++)
                h3_gpu_linear_i8_weight_bf16(gpu, output, input, rowmajor,
                                             scales, NULL, rows, sh.in, sh.out);
            if (!h3_gpu_submit(gpu)) return 1;
            const double a = (now() - began) / 3;
            began = now();
            if (!h3_gpu_begin(gpu)) return 1;
            for (int r = 0; r < 3; r++)
                h3_gpu_linear_i8_weight_bf16_square(gpu, output, input, kmajor,
                                                    scales, NULL, rows, sh.in, sh.out);
            if (!h3_gpu_submit(gpu)) return 1;
            const double b = (now() - began) / 3;
            if (pass == 0 || a < old) old = a;
            if (pass == 0 || b < new) new = b;
        }
        total_old += old; total_new += new;
        printf("%-6s %8.1f ms %8.1f ms %8.2fx %10.2f\n", sh.label, old * 1e3,
               new * 1e3, old / new,
               2.0 * rows * sh.in * sh.out / new / 1e12);
        h3_gpu_tensor_free(input); h3_gpu_tensor_free(output);
        h3_gpu_tensor_free(rowmajor); h3_gpu_tensor_free(kmajor);
        h3_gpu_tensor_free(scales);
    }
    printf("\nblock  %8.1f ms %8.1f ms %8.2fx\n", total_old * 1e3,
           total_new * 1e3, total_old / total_new);
    h3_gpu_free(gpu);
    return 0;
}
