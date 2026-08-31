#include "ltx_tae_geometry.h"

#include <math.h>
#include <stdio.h>

static int check(float actual, float expected, const char *what) {
    if (fabsf(actual - expected) < 1e-6f) return 1;
    fprintf(stderr, "%s: got %.6f, expected %.6f\n", what, actual, expected);
    return 0;
}

int main(void) {
    /* Two packed pixels side-by-side. Every channel/tile coordinate gets a
     * unique value so a wrong channel-to-space order cannot hide. */
    float packed[2 * 48];
    float rgb[4 * 8 * 3] = {0};
    for (int pixel = 0; pixel < 2; pixel++)
        for (int channel = 0; channel < 3; channel++)
            for (int dy = 0; dy < 4; dy++)
                for (int dx = 0; dx < 4; dx++) {
                    const int at = pixel * 48 + channel * 16 + dy * 4 + dx;
                    packed[at] = (float)(pixel * 40 + channel * 10 +
                                         dy * 2 + dx) / 100.0f;
                }
    packed[0] = -1.0f;
    packed[48 + 2 * 16 + 3 * 4 + 3] = 2.0f;

    ltx_tae_pixel_shuffle4(packed, 1, 2, rgb);
    int ok = 1;
    ok &= check(rgb[(0 * 8 + 0) * 3 + 0], 0.0f, "lower clamp");
    ok &= check(rgb[(2 * 8 + 3) * 3 + 1], 0.17f, "first packed pixel");
    ok &= check(rgb[(1 * 8 + 6) * 3 + 2], 0.64f, "second packed pixel");
    ok &= check(rgb[(3 * 8 + 7) * 3 + 2], 1.0f, "upper clamp");
    return ok ? 0 : 1;
}
