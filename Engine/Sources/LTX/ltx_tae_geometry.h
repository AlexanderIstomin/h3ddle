#ifndef LTX_TAE_GEOMETRY_H
#define LTX_TAE_GEOMETRY_H

/* Rearranges [height, width, 3*4*4] into [height*4, width*4, 3] and clamps
 * the final RGB values to [0, 1]. */
void ltx_tae_pixel_shuffle4(const float *packed, int height, int width,
                            float *rgb);

#endif
