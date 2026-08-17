/* Does the Z-Image tile work carry over to LTX-2.5's dual-stream DiT?
 *
 * The same question h3_tile_check.c asks for H3, at LTX's shapes and at both
 * of its row counts. LTX's DiT ships int8 ConvRot already, so it will run this
 * GEMM whatever else the port does; the only question is which geometry, and
 * unlike H3 nothing has been converted yet, so the answer costs a converter
 * default rather than a migration.
 *
 * Two regimes, not one. The video stream runs thousands of tokens and is the
 * regime the 64x40 tile was swept in. The audio stream runs a few hundred,
 * which nobody has measured: at 256 rows the tile is four deep, so occupancy
 * decides rather than the weight re-read the sweep was reasoning about.
 *
 * Shapes are from the released checkpoint, not from the paper. Note that the
 * feed-forward is plain GELU at 4x rather than gated — ff.net.0.proj is
 * [16384, 4096] and net.2 consumes all 16384 — so there is no w1/w3 fusion
 * here to survive the transpose, unlike Z-Image.
 *
 * This times; it does not check that the two kernels agree. A retiling is the
 * same arithmetic in a different order and the gate for that is a reference
 * capture before and after, the way the Z-Image port was gated.
 */
#include "h3_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

typedef struct { const char *label; int rows, in, out; } shape;

int main(int argc, char **argv) {
    char error[512] = {0};
    if (!h3_gpu_prepare(argv[1], error, sizeof(error))) { puts(error); return 1; }
    h3_gpu *gpu = h3_gpu_create(argv[1], error, sizeof(error));
    if (!gpu) { puts(error); return 1; }

    const int video = argc > 2 ? atoi(argv[2]) : 4096;
    const int audio = argc > 3 ? atoi(argv[3]) : 256;

    /* One of each distinct shape. A block runs several of some of them: six of
     * the 4096->4096 at video rows (attn1's q, k, v and out, then attn2's q
     * and out), and two of the v2a k/v. */
    const shape shapes[] = {
        {"v qkv",   video,  4096,  4096},
        {"v out",   video,  4096,  4096},
        {"v ff1",   video,  4096, 16384},
        {"v ff2",   video, 16384,  4096},
        /* Cross-modal: video rows, but projected through the audio width. */
        {"a2v q",   video,  4096,  2048},
        {"a2v out", video,  2048,  4096},
        {"v2a k",   video,  4096,  2048},
        /* The audio stream, at its own row count. */
        {"a qkv",   audio,  2048,  2048},
        {"a ff1",   audio,  2048,  8192},
        {"a ff2",   audio,  8192,  2048},
    };
    const int count = (int)(sizeof(shapes) / sizeof(shapes[0]));

    printf("LTX-2.5 DiT: %d video rows, %d audio rows\n\n", video, audio);
    printf("%-8s %6s %6s %6s %10s %10s %9s %9s\n", "shape", "rows", "K", "N",
           "8x8 now", "64x40", "speedup", "TFLOP/s");
    for (int s = 0; s < count; s++) {
        const shape sh = shapes[s];
        const size_t in = (size_t)sh.rows * sh.in, out = (size_t)sh.rows * sh.out;
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
        /* Interleaved, because sequential blocks on this machine measure drift.
         * Discard the whole first run rather than the first pass: shader
         * compilation and first-touch cost showed up as a 45% spread between
         * two shapes that are identical. */
        for (int pass = 0; pass < 3; pass++) {
            double began = now();
            if (!h3_gpu_begin(gpu)) return 1;
            for (int r = 0; r < 2; r++)
                h3_gpu_linear_i8_weight_bf16(gpu, output, input, rowmajor,
                                             scales, NULL, sh.rows, sh.in, sh.out);
            if (!h3_gpu_submit(gpu)) return 1;
            const double a = (now() - began) / 2;
            began = now();
            if (!h3_gpu_begin(gpu)) return 1;
            for (int r = 0; r < 2; r++)
                h3_gpu_linear_i8_weight_bf16_square(gpu, output, input, kmajor,
                                                    scales, NULL, sh.rows, sh.in, sh.out);
            if (!h3_gpu_submit(gpu)) return 1;
            const double b = (now() - began) / 2;
            if (pass == 0 || a < old) old = a;
            if (pass == 0 || b < new) new = b;
        }
        printf("%-8s %6d %6d %6d %8.2f ms %8.2f ms %8.2fx %9.2f\n", sh.label,
               sh.rows, sh.in, sh.out, old * 1e3, new * 1e3, old / new,
               2.0 * sh.rows * sh.in * sh.out / new / 1e12);
        fflush(stdout);
        h3_gpu_tensor_free(input); h3_gpu_tensor_free(output);
        h3_gpu_tensor_free(rowmajor); h3_gpu_tensor_free(kmajor);
        h3_gpu_tensor_free(scales);
    }
    h3_gpu_free(gpu);
    return 0;
}
