/* Drive zimage_generate() the way the service will, and write the raw planes.
 *
 * Thin on purpose: the pipeline moved into zimage_generate.c so the service
 * has one call, and this exists to exercise that call rather than a private
 * copy of the same steps. If the two ever diverge the picture is no longer
 * evidence about what ships.
 *
 *   ./zimage_render <package> <prompt> <width> <height> <steps> <out.bin>
 *                   [shaders]
 */
#include "zimage_generate.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static double started;

static int report(const char *phase, int step, int steps, void *context) {
    (void)context;
    printf("  %-14s %d/%d  %.1f s\n", phase ? phase : "?", step, steps,
           now() - started);
    fflush(stdout);
    started = now();
    return 1;                     /* returning 0 here is how the service cancels */
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s <package> <prompt> <width> <height> <steps> "
                        "<out.bin> [shaders]\n", argv[0]);
        return 2;
    }
    const zimage_request request = {
        .package = argv[1],
        .shaders = argc > 7 ? argv[7] : NULL,
        .prompt = argv[2],
        .width = atoi(argv[3]),
        .height = atoi(argv[4]),
        .steps = atoi(argv[5]),
        .seed = 0,
    };
    const size_t count = (size_t)3 * request.width * request.height;
    float *image = malloc(count * sizeof(float));
    if (!image) { fprintf(stderr, "out of memory\n"); return 1; }

    char error[512] = {0};
    const double began = now();
    started = began;
    if (!zimage_generate(&request, image, report, NULL, error, sizeof(error))) {
        fprintf(stderr, "%s\n", *error ? error : "cancelled");
        return 1;
    }
    printf("%d x %d in %.1f s\n", request.width, request.height, now() - began);

    FILE *out = fopen(argv[6], "wb");
    if (!out) { fprintf(stderr, "cannot write %s\n", argv[6]); return 1; }
    fwrite(image, sizeof(float), count, out);
    fclose(out);
    free(image);
    return 0;
}
