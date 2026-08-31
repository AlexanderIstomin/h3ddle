#include "ltx_tae_geometry.h"

#include <stddef.h>

enum { PATCH_SIZE = 4, OUTPUT_CHANNELS = 3 * PATCH_SIZE * PATCH_SIZE };

void ltx_tae_pixel_shuffle4(const float *packed, int height, int width,
                            float *rgb) {
    if (!packed || !rgb || height < 1 || width < 1) return;
    const int output_width = width * PATCH_SIZE;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int channel = 0; channel < 3; channel++)
                for (int dy = 0; dy < PATCH_SIZE; dy++)
                    for (int dx = 0; dx < PATCH_SIZE; dx++) {
                        const size_t source =
                            ((size_t)y * (size_t)width + (size_t)x) *
                                OUTPUT_CHANNELS +
                            (size_t)channel * PATCH_SIZE * PATCH_SIZE +
                            (size_t)dy * PATCH_SIZE + (size_t)dx;
                        const size_t destination =
                            ((size_t)(y * PATCH_SIZE + dy) *
                                 (size_t)output_width +
                             (size_t)(x * PATCH_SIZE + dx)) * 3 +
                            (size_t)channel;
                        const float value = packed[source];
                        rgb[destination] = value < 0.0f ? 0.0f :
                                           (value > 1.0f ? 1.0f : value);
                    }
}
