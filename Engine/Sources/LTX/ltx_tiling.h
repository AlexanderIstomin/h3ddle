/* Small, allocation-free helpers for LTX-2.5 VAE tiling.
 *
 * Tile sizes are expressed on the latent grid. Mapping an interval through the
 * decoder's 8x temporal or 32x spatial scale also maps its blend ramps, so the
 * independently decoded tiles add back to one without a full-video weights
 * buffer.
 */
#ifndef LTX_TILING_H
#define LTX_TILING_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t start;
    uint32_t end;
    uint32_t left_ramp;
    uint32_t right_ramp;
} ltx_tile_interval;

/* Returns the number of intervals, or zero for invalid arguments. When `out`
 * is NULL the call only measures the plan. A temporal causal plan shifts every
 * tile after the first one cell to the left, matching LTX's reference tiler. */
size_t ltx_tile_axis(uint32_t length, uint32_t tile_size, uint32_t overlap,
                     int temporal_causal, ltx_tile_interval *out,
                     size_t capacity);

/* Maps a latent-grid interval to its decoded output coordinates and ramps. */
ltx_tile_interval ltx_tile_output(ltx_tile_interval latent, uint32_t scale,
                                  int temporal);

/* One value from the reference tiler's trapezoidal blend mask. */
float ltx_tile_mask(uint32_t length, uint32_t left_ramp,
                    uint32_t right_ramp, int left_starts_at_zero,
                    uint32_t index);

#endif
