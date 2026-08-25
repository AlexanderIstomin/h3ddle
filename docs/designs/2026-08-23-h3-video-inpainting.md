# Native H3 masked-video inpainting

## Scope

H3ddle accepts a source clip, a black-and-white still or video mask, and one
or more ordered Ref2VA images. White mask pixels are regenerated; black pixels
remain source footage. The source soundtrack is preserved by default and can
be discarded. Source clips and video masks are sampled at H3's 24 fps clock and
must resolve to the requested frame count.

The public Apache-2.0
[MiniMax H3 inpainting pipeline](https://huggingface.co/diffusers-modular/minimax-h3-inpainting)
is the behavioral reference. H3ddle's native C implementation is independent:
no Python pipeline code, model assets, schemas, or identifiers are copied into
the repository.

## Mask geometry

The public workflow recommends hard masks, which are the first native contract.
The media reader normalizes RGB mask pixels and the engine reduces them with a
maximum over every target DiT row:

- 16×16 source pixels per VideoVAE spatial cell;
- 2×2 VideoVAE cells per transformer row, or 32×32 source pixels; and
- the VideoVAE temporal sequence `1, 4, 4, 4, 4`, repeated for each 17 source
  frames.

Any source pixel at or above 0.5 makes its target row generative. This slightly
expands small white regions instead of letting a mixed row overwrite pixels the
user expected to preserve. After decoding, the source RGB is pasted back at
every black pixel, making the public output boundary exact even where the VAE
has reconstruction error.

## Sampling

Target rows use the normal timestep schedule. Preserved video rows use 0.999,
and preserved audio rows use 1.0, matching clean conditioning. The encoded
source is imposed before the first transformer evaluation and after every Euler
transition; preserved video rows are made exact again after the final step.

This requires the exact sampler path. Whole-denoiser reuse, core reuse, token
reduction, and tail-block caching are disabled for inpainting. The native API
also rejects those combinations so a non-app client cannot silently receive a
different mask behavior.

## Product boundary

- Inpainting is H3 video only and requires a reference-capable checkpoint.
- Frame anchors cannot be mixed with the source/mask pair.
- Both source and mask are required; a half-configured request cannot start.
- At least one ordered reference image is required to describe the replacement.
- Soft-mask modulation and a separate audio mask are deferred. The existing
  soundtrack toggle covers the common preserve-all or regenerate-all choice
  without shipping the large continuous timestep modulation table.

The engine protocol advertises this separately as `videoInpainting`, so older
helpers fail visibly instead of accepting inputs they cannot use.

## Tests

Pure C tests cover still-mask reuse, temporal group boundaries, partial tail
groups, and exact final compositing. DiT schedule tests assert that generated
and preserved target rows select different timestep levels. Swift protocol and
client tests keep the source, mask kind, soundtrack choice, and URLs intact
across the JSON-lines process boundary.
