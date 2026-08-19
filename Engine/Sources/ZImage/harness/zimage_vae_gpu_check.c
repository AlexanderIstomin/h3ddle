/* Does the GPU decoder agree with the reference the CPU one already matches?
 *
 * Same golden as stage 3, so the two decoders are being held to one standard
 * rather than to each other — an agreement between two of my own
 * implementations would prove only that I made the same mistake twice.
 *
 *   ./zimage_vae_gpu_check <shaders.metal> <vae_decoder.safetensors> <golden>
 *   ./zimage_vae_gpu_check <shaders.metal> <vae_decoder.safetensors> --synthetic [h] [w]
 */
#include "zimage_vae_gpu.h"
#include "zimage_vae.h"
#include "qwen_weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char problem[512];

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int synthetic_compare(const char *shaders, const char *decoder_path,
                             int height, int width) {
    if (height < 1 || width < 1) {
        fprintf(stderr, "latent dimensions must be positive\n");
        return 2;
    }
    const int image_height = height * 8, image_width = width * 8;
    const size_t latent_count = (size_t)16 * height * width;
    const size_t image_count = (size_t)3 * image_height * image_width;
    float *latent = malloc(latent_count * sizeof(float));
    float *host = malloc(image_count * sizeof(float));
    float *device_image = malloc(image_count * sizeof(float));
    if (!latent || !host || !device_image) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    for (size_t index = 0; index < latent_count; index++)
        latent[index] = 0.2f * sinf((float)index * 0.017f) +
                        0.1f * cosf((float)index * 0.031f);

    qwen_weights *decoder = qwen_weights_open(decoder_path, problem,
                                               sizeof(problem));
    if (!decoder) { fprintf(stderr, "decoder: %s\n", problem); return 1; }
    double began = now();
    if (!zimage_vae_decode(decoder, latent, height, width, host, NULL, NULL)) {
        fprintf(stderr, "CPU decode failed\n");
        return 1;
    }
    const double cpu_seconds = now() - began;

    h3_gpu *device = h3_gpu_create(shaders, problem, sizeof(problem));
    if (!device) { fprintf(stderr, "device: %s\n", problem); return 1; }
    zimage_vae_gpu *vae = zimage_vae_gpu_create(
        shaders, decoder_path, device, height, width, problem, sizeof(problem));
    if (!vae) { fprintf(stderr, "create: %s\n", problem); return 1; }
    began = now();
    if (!zimage_vae_gpu_decode(vae, latent, height, width, device_image,
                               problem, sizeof(problem))) {
        fprintf(stderr, "GPU decode: %s\n", problem);
        return 1;
    }
    const double gpu_seconds = now() - began;

    double square = 0.0, total = 0.0, worst = 0.0;
    for (size_t index = 0; index < image_count; index++) {
        const double difference = (double)device_image[index] - host[index];
        square += difference * difference;
        total += (double)host[index] * host[index];
        if (fabs(difference) > worst) worst = fabs(difference);
    }
    const double relative_rms = sqrt(square / (double)image_count) /
                                sqrt(total / (double)image_count);
    h3_gpu_stats stats;
    h3_gpu_get_stats(device, &stats);
    printf("latent %dx%d -> image %dx%d\n", height, width,
           image_height, image_width);
    printf("  cpu %7.2f s   device %6.2f s   %5.1fx faster\n",
           cpu_seconds, gpu_seconds, cpu_seconds / gpu_seconds);
    printf("  worst |difference| %.3e   relative rms %.3e\n",
           worst, relative_rms);
    printf("  device peak %.3f GiB\n",
           (double)stats.peak_live_bytes / (1024.0 * 1024.0 * 1024.0));

    free(latent); free(host); free(device_image);
    zimage_vae_gpu_release(vae);
    h3_gpu_free(device);
    qwen_weights_close(decoder);
    const int agrees = isfinite(relative_rms) && worst < 2e-2 &&
                       relative_rms < 5e-4;
    printf("  %s\n", agrees ? "they agree" : "THEY DO NOT AGREE");
    return agrees ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <shaders.metal> <vae_decoder.safetensors> "
                        "<golden.safetensors>\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[3], "--synthetic") == 0) {
        const int height = argc > 4 ? atoi(argv[4]) : 16;
        const int width = argc > 5 ? atoi(argv[5]) : height;
        return synthetic_compare(argv[1], argv[2], height, width);
    }
    qwen_weights *golden = qwen_weights_open(argv[3], problem, sizeof(problem));
    if (!golden) { fprintf(stderr, "golden: %s\n", problem); return 1; }

    int side = 0;
    {
        FILE *probe = fopen(argv[3], "rb");
        uint64_t header_bytes = 0;
        if (!probe || fread(&header_bytes, 8, 1, probe) != 1) return 1;
        char *header = malloc(header_bytes + 1);
        if (fread(header, 1, header_bytes, probe) != header_bytes) return 1;
        header[header_bytes] = '\0';
        fclose(probe);
        const char *entry = strstr(header, "\"latent\"");
        const char *shape = entry ? strstr(entry, "\"shape\":[1,16,") : NULL;
        if (shape) side = atoi(shape + strlen("\"shape\":[1,16,"));
        free(header);
    }
    if (side < 1) { fprintf(stderr, "cannot read the latent size\n"); return 1; }

    const int64_t latent_shape[4] = {1, 16, side, side};
    const float *latent = qwen_weights_f32(golden, "latent", 4, latent_shape,
                                           problem, sizeof(problem));
    if (!latent) { fprintf(stderr, "golden: %s\n", problem); return 1; }

    double began = now();
    zimage_vae_gpu *vae = zimage_vae_gpu_create(argv[1], argv[2], NULL, side,
                                                side, problem, sizeof(problem));
    if (!vae) { fprintf(stderr, "create: %s\n", problem); return 1; }
    printf("latent %dx%d -> image %dx%d, loaded in %.1f s\n\n",
           side, side, side * 8, side * 8, now() - began);

    const int pixels = side * 8;
    float *image = malloc((size_t)3 * pixels * pixels * sizeof(float));
    double elapsed = 0.0;
    for (int pass = 0; pass < 3; pass++) {   /* the first pays for pipelines */
        began = now();
        if (!zimage_vae_gpu_decode(vae, latent, side, side, image,
                                   problem, sizeof(problem))) {
            fprintf(stderr, "decode: %s\n", problem);
            return 1;
        }
        elapsed = now() - began;
        printf("  pass %d: %.3f s\n", pass, elapsed);
    }

    const int64_t image_shape[4] = {1, 3, pixels, pixels};
    const float *theirs = qwen_weights_f32(golden, "image", 4, image_shape,
                                           problem, sizeof(problem));
    if (!theirs) { fprintf(stderr, "golden: %s\n", problem); return 1; }
    double square = 0.0, total = 0.0, worst = 0.0;
    for (size_t index = 0; index < (size_t)3 * pixels * pixels; index++) {
        const double difference = (double)image[index] - (double)theirs[index];
        square += difference * difference;
        total += (double)theirs[index] * (double)theirs[index];
        if (fabs(difference) > worst) worst = fabs(difference);
    }
    const size_t count = (size_t)3 * pixels * pixels;
    printf("\n  image        RMS rel %9.2e   worst %9.2e\n",
           sqrt(square / (double)count) / sqrt(total / (double)count), worst);
    printf("  the CPU decoder takes 21 s at 256px and 80 s at 512px\n");

    free(image);
    zimage_vae_gpu_release(vae);
    qwen_weights_close(golden);
    return 0;
}
