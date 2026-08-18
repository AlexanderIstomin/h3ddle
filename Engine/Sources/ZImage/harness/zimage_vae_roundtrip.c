/* Does the encoder invert the decoder?
 *
 * There is no golden for this stage and no way to make one here — the
 * reference needs torch, which is not installed. What there is instead is a
 * decoder already checked stage by stage against the reference, which turns
 * the encoder into the only unknown in a round trip: run the reference's own
 * decoded picture back through it and the picture has to survive.
 *
 * That is a weaker claim than "agrees to rounding" and it is stated as such.
 * It is also the claim that actually matters for img2img, and it is not one a
 * broken encoder can pass by accident: the downsample alone, padded on the
 * wrong side, walks the picture half a pixel per stage and takes the
 * reconstruction apart.
 *
 * The latent comparison is reported but deliberately not asserted on. The
 * golden's latent is a seeded normal rather than anything the encoder ever
 * produced, so encode(decode(z)) is only approximately z off the manifold.
 *
 *   ./zimage_vae_roundtrip <package-dir> <vae-golden.safetensors> <encoder.safetensors>
 */
#include "zimage_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char problem[512];

static void moments(const char *stage, const float *values, size_t count,
                    void *context) {
    (void)context;
    /* The log-variance decides whether sampling and the mean differ enough to
     * matter; everything else here is silent. */
    if (strcmp(stage, "moments") != 0) return;
    const size_t half = count / 2;
    double low = 1e30, high = -1e30, sum = 0;
    for (size_t index = half; index < count; index++) {
        const double value = values[index];
        if (value < low) low = value;
        if (value > high) high = value;
        sum += value;
    }
    printf("log-variance  min %.3f  mean %.3f  max %.3f"
           "  -> sigma up to %.3g\n", low, sum / (double)half, high,
           exp(0.5 * high));
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <package-dir> <golden> <encoder>\n", argv[0]);
        return 2;
    }
    char path[1200];
    snprintf(path, sizeof(path), "%s/vae_decoder.safetensors", argv[1]);
    qwen_weights *decoder = qwen_weights_open(path, problem, sizeof(problem));
    if (!decoder) { fprintf(stderr, "decoder: %s\n", problem); return 1; }
    qwen_weights *golden = qwen_weights_open(argv[2], problem, sizeof(problem));
    if (!golden) { fprintf(stderr, "golden: %s\n", problem); return 1; }
    qwen_weights *encoder = qwen_weights_open(argv[3], problem, sizeof(problem));
    if (!encoder) { fprintf(stderr, "encoder: %s\n", problem); return 1; }

    const int side = 16, image_side = side * 8;
    int64_t image_shape[4] = {1, 3, image_side, image_side};
    const float *image = qwen_weights_f32(golden, "image", 4, image_shape,
                                          problem, sizeof(problem));
    int64_t latent_shape[4] = {1, 16, side, side};
    const float *reference_latent = qwen_weights_f32(
        golden, "latent", 4, latent_shape, problem, sizeof(problem));
    if (!image || !reference_latent) {
        fprintf(stderr, "golden: %s\n", problem); return 1;
    }
    printf("picture %dx%d -> latent %dx%dx16\n\n", image_side, image_side,
           side, side);

    const size_t latent_count = (size_t)16 * side * side;
    const size_t image_count = (size_t)3 * image_side * image_side;
    float *latent = malloc(latent_count * sizeof(float));
    float *again = malloc(image_count * sizeof(float));
    if (!latent || !again) { fprintf(stderr, "out of memory\n"); return 1; }

    if (!zimage_vae_encode(encoder, image, image_side, latent, moments, NULL)) {
        fprintf(stderr, "encode failed\n"); return 1;
    }

    /* Against the golden's own latent, for information only. */
    double sxx = 0, syy = 0, sxy = 0, sx = 0, sy = 0;
    for (size_t index = 0; index < latent_count; index++) {
        const double a = latent[index], b = reference_latent[index];
        sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b;
    }
    const double n = (double)latent_count;
    const double correlation = (sxy - sx * sy / n) /
        sqrt((sxx - sx * sx / n) * (syy - sy * sy / n));
    printf("latent vs the golden's seeded normal: correlation %.4f\n",
           correlation);

    if (!zimage_vae_decode(decoder, latent, side, again, NULL, NULL)) {
        fprintf(stderr, "decode failed\n"); return 1;
    }
    double error = 0, peak = 0;
    for (size_t index = 0; index < image_count; index++) {
        const double delta = again[index] - image[index];
        error += delta * delta;
        if (fabs(image[index]) > peak) peak = fabs(image[index]);
    }
    const double rmse = sqrt(error / (double)image_count);
    printf("\nround trip: rmse %.5f over +/-%.3f  ->  PSNR %.2f dB\n",
           rmse, peak, 20.0 * log10(2.0 / rmse));
    printf("%s\n", rmse < 0.1 ? "the picture survives" : "IT DOES NOT");
    return 0;
}
