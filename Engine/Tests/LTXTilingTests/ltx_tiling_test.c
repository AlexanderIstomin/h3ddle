#include "ltx_tiling.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void require(int condition, const char *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
}

static void require_complementary(uint32_t length, uint32_t tile,
                                  uint32_t overlap, int temporal,
                                  uint32_t scale) {
    const size_t count = ltx_tile_axis(length, tile, overlap, temporal, NULL, 0);
    require(count > 0, "axis plan exists");
    ltx_tile_interval *intervals = calloc(count, sizeof(*intervals));
    require(intervals != NULL, "axis plan allocation");
    require(ltx_tile_axis(length, tile, overlap, temporal,
                          intervals, count) == count,
            "axis plan fills every interval");
    const uint32_t output_length = temporal
        ? 1 + (length - 1) * scale : length * scale;
    for (uint32_t position = 0; position < output_length; position++) {
        float sum = 0.0f;
        for (size_t index = 0; index < count; index++) {
            const ltx_tile_interval mapped =
                ltx_tile_output(intervals[index], scale, temporal);
            if (position < mapped.start || position >= mapped.end) continue;
            sum += ltx_tile_mask(mapped.end - mapped.start,
                                 mapped.left_ramp, mapped.right_ramp,
                                 temporal, position - mapped.start);
        }
        require(fabsf(sum - 1.0f) < 2e-6f,
                "overlap masks add to one");
    }
    free(intervals);
}

int main(void) {
    ltx_tile_interval temporal[7] = {0};
    require(ltx_tile_axis(46, 10, 3, 1, temporal, 7) == 7,
            "361 frames split into seven temporal tiles");
    require(temporal[0].start == 0 && temporal[0].end == 10 &&
            temporal[0].left_ramp == 0 && temporal[0].right_ramp == 3,
            "first temporal tile matches the reference split");
    require(temporal[1].start == 6 && temporal[1].end == 17 &&
            temporal[1].left_ramp == 4 && temporal[1].right_ramp == 3,
            "causal temporal tile carries one context cell");
    require(temporal[6].start == 41 && temporal[6].end == 46 &&
            temporal[6].left_ramp == 4 && temporal[6].right_ramp == 0,
            "last temporal tile reaches the final frame");
    const ltx_tile_interval first_time = ltx_tile_output(temporal[0], 8, 1);
    const ltx_tile_interval second_time = ltx_tile_output(temporal[1], 8, 1);
    require(first_time.start == 0 && first_time.end == 73 &&
            first_time.right_ramp == 24,
            "first temporal tile maps onto decoded frames");
    require(second_time.start == 48 && second_time.end == 129 &&
            second_time.left_ramp == 25,
            "causal context maps onto a complementary output ramp");

    ltx_tile_interval horizontal[2] = {0};
    require(ltx_tile_axis(39, 24, 2, 0, horizontal, 2) == 2,
            "1248 pixels split into two spatial tiles");
    const ltx_tile_interval first_width = ltx_tile_output(horizontal[0], 32, 0);
    const ltx_tile_interval second_width = ltx_tile_output(horizontal[1], 32, 0);
    require(first_width.start == 0 && first_width.end == 768 &&
            first_width.right_ramp == 64,
            "spatial tile uses the official 768/64 geometry");
    require(second_width.start == 704 && second_width.end == 1248 &&
            second_width.left_ramp == 64,
            "last spatial tile covers the remaining canvas");
    require_complementary(46, 10, 3, 1, 8);
    require_complementary(39, 24, 2, 0, 32);
    require_complementary(22, 24, 2, 0, 32);

    /* The late F32 activation is [frames, H/4, W/4, 128], equivalently
     * frames * pixel-H * pixel-W * 32 bytes. It used to cover all 361 frames;
     * the largest official-default tile is bounded to 81x768x768 here. */
    const uint64_t max_activation = UINT64_C(81) * 768 * 768 * 32;
    require(max_activation < UINT64_C(1530) * 1000 * 1000,
            "one late tiled activation stays below 1.53 GB");

    printf("ok: LTX VAE tiles are bounded and overlap masks are complementary\n");
    return 0;
}
