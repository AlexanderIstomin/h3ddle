#!/usr/bin/env python3
"""Turn the generator's raw f32 planes into a PNG.

The decoder emits [3][H][W] in roughly [-1, 1], which is what the reference's
image_processor.postprocess takes: map to [0, 1], clamp, and quantise. Doing
the clamp *after* the shift rather than before is the difference between a
blown-out picture and a correct one.

  zimage_png.py render.bin render.png
"""

import sys

import numpy as np
from PIL import Image


def main():
    raw = np.fromfile(sys.argv[1], dtype=np.float32)
    side = int(round((raw.size / 3) ** 0.5))
    assert side * side * 3 == raw.size, f"{raw.size} floats is not 3 square planes"
    planes = raw.reshape(3, side, side)
    print(f"{side}x{side}, range [{planes.min():.3f}, {planes.max():.3f}], "
          f"mean {planes.mean():.3f}")
    pixels = np.clip(planes / 2.0 + 0.5, 0.0, 1.0)
    Image.fromarray((pixels * 255).round().astype(np.uint8).transpose(1, 2, 0)).save(sys.argv[2])
    print(f"wrote {sys.argv[2]}")


if __name__ == "__main__":
    main()
