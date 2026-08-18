/* Does the lifted encoder reproduce LTX's own?
 *
 * Unlike the other checks this one does not compare against the driver — it
 * compares against the anchor the driver itself is held to, which is
 * `CausalVideoAutoencoder`'s output for the same picture. That is a stronger
 * bar and it is available here for free, so there is no reason to take the
 * weaker one.
 *
 * build:
 *   clang -O2 -fobjc-arc -o ltx_video_encode_check \
 *       harness/ltx_video_encode_check.c ltx_video.c \
 *       ../../Vendor/h3.c/h3_gpu.m ../../Vendor/h3.c/h3_weights.c \
 *       ../../Vendor/h3.c/h3_safetensors.c -I../../Vendor/h3.c -I. \
 *       -framework Foundation -framework Metal \
 *       -framework MetalPerformanceShaders \
 *       -framework MetalPerformanceShadersGraph -framework Accelerate -lm
 *
 * usage: ltx_video_encode_check VAE.safetensors ANCHOR.safetensors */
#include "ltx_video.h"

#include "h3_safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum { FRAMES = 9, SIDE = 128, LATENT_FRAMES = 2, LATENT_SIDE = 4 };

/* The reference computes in F32 and so does this; what remains is the order
 * the two accumulate a 3x3x3 convolution in, which the driver measures at
 * 5e-5 of peak across every stage. */
#define TOLERANCE 5e-5

static float *read_f32(const h3_weight_store *store, const char *name,
                       size_t expected) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) { fprintf(stderr, "no tensor %s\n", name); return NULL; }
    const size_t elements = (size_t)h3_st_tensor_elements(tensor);
    if (elements != expected) {
        fprintf(stderr, "%s has %zu elements, expected %zu\n", name, elements,
                expected);
        return NULL;
    }
    float *values = malloc(elements * sizeof(*values));
    char why[256];
    if (!values || !h3_st_read_data(header, tensor, values,
                                    elements * sizeof(*values), why,
                                    sizeof(why))) {
        fprintf(stderr, "cannot read %s\n", name);
        free(values);
        return NULL;
    }
    return values;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s VAE.safetensors ANCHOR.safetensors\n",
                argv[0]);
        return 2;
    }
    char error[512];
    h3_weight_store *vae = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!vae) { fprintf(stderr, "cannot open the VAE: %s\n", error); return 1; }
    h3_weight_store *anchor = h3_weight_store_open(argv[2], error, sizeof(error));
    if (!anchor) { fprintf(stderr, "cannot open the anchor: %s\n", error); return 1; }
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "cannot create Metal: %s\n", error); return 1; }

    const size_t pixels = (size_t)LTX_VIDEO_CHANNELS * FRAMES * SIDE * SIDE;
    const size_t cells = (size_t)LATENT_FRAMES * LATENT_SIDE * LATENT_SIDE;
    float *image = read_f32(anchor, "image", pixels);
    float *want = read_f32(anchor, "enc_normalized",
                           cells * LTX_VIDEO_LATENT_CHANNELS);
    float *latent = malloc(cells * LTX_VIDEO_LATENT_CHANNELS * sizeof(*latent));
    if (!image || !want || !latent) return 1;

    printf("LTX-2.5 video VAE encoder, %d frames of %dx%d -> [%d, %d, %d, %d]\n",
           FRAMES, SIDE, SIDE, LTX_VIDEO_LATENT_CHANNELS, LATENT_FRAMES,
           LATENT_SIDE, LATENT_SIDE);
    if (!ltx_video_encode(gpu, vae, image, FRAMES, SIDE, SIDE, latent, error,
                          sizeof(error))) {
        fprintf(stderr, "FAIL: %s\n", error);
        return 1;
    }

    /* The anchor is channel-major and the encoder is channels-last. */
    double worst = 0.0, peak = 0.0;
    for (size_t cell = 0; cell < cells; cell++)
        for (uint32_t channel = 0; channel < LTX_VIDEO_LATENT_CHANNELS; channel++) {
            const double got = latent[cell * LTX_VIDEO_LATENT_CHANNELS + channel];
            const double expected = want[(size_t)channel * cells + cell];
            if (!isfinite(got)) {
                fprintf(stderr, "FAIL: produced %f at cell %zu\n", got, cell);
                return 1;
            }
            const double delta = fabs(got - expected);
            if (delta > worst) worst = delta;
            if (fabs(expected) > peak) peak = fabs(expected);
        }
    const double relative = peak > 0.0 ? worst / peak : worst;
    printf("  %s enc_normalized %.3e = %.2e of peak %.4f\n",
           relative > TOLERANCE ? "FAIL" : "ok  ", worst, relative, peak);
    free(image); free(want); free(latent);
    h3_gpu_free(gpu);
    h3_weight_store_free(vae);
    h3_weight_store_free(anchor);
    return relative > TOLERANCE ? 1 : 0;
}
