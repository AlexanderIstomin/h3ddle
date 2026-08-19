#include "ltx_tiling.h"

#include <limits.h>

size_t ltx_tile_axis(uint32_t length, uint32_t tile_size, uint32_t overlap,
                     int temporal_causal, ltx_tile_interval *out,
                     size_t capacity) {
    if (!length || !tile_size || overlap >= tile_size) return 0;
    size_t count = 1;
    if (length > tile_size) {
        const uint64_t stride = (uint64_t)tile_size - overlap;
        /* ceil((length - overlap) / stride), written the same way as LTX's
         * `split_by_size` so its last short tile lands at the same coordinate. */
        count = (size_t)(((uint64_t)length + tile_size - 2 * (uint64_t)overlap - 1) /
                         stride);
    }
    if (!out) return count;
    if (capacity < count) return 0;

    if (count == 1) {
        out[0] = (ltx_tile_interval){0, length, 0, 0};
        return 1;
    }

    const uint32_t stride = tile_size - overlap;
    for (size_t index = 0; index < count; index++) {
        ltx_tile_interval interval;
        if (index == 0) {
            interval = (ltx_tile_interval){0, tile_size, 0, overlap};
        } else if (index + 1 == count) {
            const uint64_t start = (uint64_t)index * stride;
            if (start >= length || start > UINT32_MAX) return 0;
            interval = (ltx_tile_interval){(uint32_t)start, length, overlap, 0};
        } else {
            const uint64_t start = (uint64_t)index * stride;
            const uint64_t end = start + tile_size;
            if (end > length || end > UINT32_MAX) return 0;
            interval = (ltx_tile_interval){(uint32_t)start, (uint32_t)end,
                                           overlap, overlap};
        }
        if (temporal_causal && index > 0) {
            if (!interval.start || interval.left_ramp == UINT32_MAX) return 0;
            interval.start -= 1;
            interval.left_ramp += 1;
        }
        out[index] = interval;
    }
    return count;
}

ltx_tile_interval ltx_tile_output(ltx_tile_interval latent, uint32_t scale,
                                  int temporal) {
    ltx_tile_interval output = {0};
    if (!scale || latent.end <= latent.start) return output;
    const uint64_t start = (uint64_t)latent.start * scale;
    const uint64_t end = temporal
        ? 1 + (uint64_t)(latent.end - 1) * scale
        : (uint64_t)latent.end * scale;
    const uint64_t left = temporal
        ? (latent.left_ramp ? 1 + (uint64_t)(latent.left_ramp - 1) * scale : 0)
        : (uint64_t)latent.left_ramp * scale;
    const uint64_t right = (uint64_t)latent.right_ramp * scale;
    if (start > UINT32_MAX || end > UINT32_MAX ||
        left > UINT32_MAX || right > UINT32_MAX) return output;
    output.start = (uint32_t)start;
    output.end = (uint32_t)end;
    output.left_ramp = (uint32_t)left;
    output.right_ramp = (uint32_t)right;
    return output;
}

float ltx_tile_mask(uint32_t length, uint32_t left_ramp,
                    uint32_t right_ramp, int left_starts_at_zero,
                    uint32_t index) {
    if (!length || index >= length) return 0.0f;
    if (left_ramp > length) left_ramp = length;
    if (right_ramp > length) right_ramp = length;
    float value = 1.0f;
    if (left_ramp && index < left_ramp) {
        value *= left_starts_at_zero
            ? (float)index / (float)left_ramp
            : (float)(index + 1) / (float)(left_ramp + 1);
    }
    if (right_ramp && index >= length - right_ramp) {
        const uint32_t within = index - (length - right_ramp);
        value *= (float)(right_ramp - within) / (float)(right_ramp + 1);
    }
    return value;
}
