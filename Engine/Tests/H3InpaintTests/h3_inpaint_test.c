#include "h3_inpaint.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(int condition, const char *message) {
    if (condition) return;
    fprintf(stderr, "inpaint mask test failed: %s\n", message);
    exit(1);
}

static void paint(float *mask, int frames, int height, int width,
                  int frame, int y, int x) {
    size_t area = (size_t)height * (size_t)width;
    for (int channel = 0; channel < 3; channel++)
        mask[((size_t)channel * (size_t)frames + (size_t)frame) * area +
             (size_t)y * (size_t)width + (size_t)x] = 1.0f;
}

int main(void) {
    enum { FRAMES = 5, HEIGHT = 32, WIDTH = 64, LATENT_T = 2 };
    size_t pixels = (size_t)3 * FRAMES * HEIGHT * WIDTH;
    float *mask = calloc(pixels, sizeof(*mask));
    uint8_t rows[4] = {0};
    char error[256];
    check(mask != NULL, "allocation");

    /* Pixel frame zero belongs only to the first latent; frame one starts the
     * four-frame group represented by the second latent. */
    paint(mask, FRAMES, HEIGHT, WIDTH, 0, 3, 2);
    paint(mask, FRAMES, HEIGHT, WIDTH, 1, 3, 40);
    check(h3_inpaint_hard_mask_rows(mask, FRAMES, FRAMES, HEIGHT, WIDTH,
                                    LATENT_T, 2, 4, rows, 4,
                                    error, sizeof(error)), error);
    check(rows[0] == 1 && rows[1] == 0,
          "the first pixel frame maps to the first latent only");
    check(rows[2] == 0 && rows[3] == 1,
          "the four-frame group maps to the second latent");

    memset(mask, 0, pixels * sizeof(*mask));
    paint(mask, FRAMES, HEIGHT, WIDTH, 4, 0, 0);
    memset(rows, 0, sizeof(rows));
    check(h3_inpaint_hard_mask_rows(mask, FRAMES, FRAMES, HEIGHT, WIDTH,
                                    LATENT_T, 2, 4, rows, 4,
                                    error, sizeof(error)), error);
    check(rows[2] == 1, "the aligned tail belongs to the final latent");

    float still[3 * HEIGHT * WIDTH];
    memset(still, 0, sizeof(still));
    paint(still, 1, HEIGHT, WIDTH, 0, 0, 33);
    memset(rows, 0, sizeof(rows));
    check(h3_inpaint_hard_mask_rows(still, 1, FRAMES, HEIGHT, WIDTH,
                                    LATENT_T, 2, 4, rows, 4,
                                    error, sizeof(error)), error);
    check(rows[1] == 1 && rows[3] == 1,
          "a still mask is held over every latent frame");

    /* The public boundary is stricter than latent preservation: black pixels
     * are copied from the source after decode so VAE error cannot leak into
     * an area the user did not select. */
    float source[6] = {0.25f, 0.75f, 0.5f, 0.5f, 1.0f, 0.0f};
    float composite_mask[6] = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    uint8_t generated[6] = {7, 8, 9, 10, 11, 12};
    check(h3_inpaint_composite_rgb24(
              generated, 1, 1, 2, source, composite_mask, 1,
              error, sizeof(error)), error);
    check(generated[0] == 64 && generated[1] == 128 && generated[2] == 255,
          "black pixels are restored from channel-major source RGB");
    check(generated[3] == 10 && generated[4] == 11 && generated[5] == 12,
          "white pixels retain generated RGB");

    free(mask);
    return 0;
}
