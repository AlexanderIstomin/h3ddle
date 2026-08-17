# LTX-2.5's video VAE, and the reference that does not match it

Written before implementing the decoder, because the first thing checking the
released weights against the public implementation found was that they are
different models — and the difference is invisible to a loader that matches on
tensor names.

## The finding

`CausalVideoAutoencoder` in `Lightricks/LTX-Video@main` builds a decoder whose
parameter **names** match the LTX-2.5 checkpoint exactly — 84 tensors, none
missing, none unexpected — and whose **shapes** disagree on 45 of them.

The cause is two block types. `compress_all` takes a `multiplier` and uses it
to reduce its output channel count; `compress_time` and `compress_space` accept
the same `multiplier` in their config and ignore it. The released checkpoint
was trained with all three honouring it.

Passing it through for those two, and dividing the running channel width the
way `compress_all` already does, reconciles the two exactly:

```
as fetched:  missing 0, unexpected 0, wrong shape 45
patched:     missing 0, unexpected 0, wrong shape 0
```

The check is `check_vae_version.py` in the session scratchpad. It builds the
reference decoder from the checkpoint's own config and diffs the state dict, so
it settles the question on shapes rather than on reading.

**This is the dangerous shape of version drift.** Same class name, same config
schema, same tensor names, same tensor count. `load_state_dict(strict=False)`
reports nothing. Only the shapes say anything, and only if something checks
them.

## What the checkpoint actually is

`ltx-2.5-video-vae-conv-bf16.safetensors`, 1.4 GB, 170 tensors: 84 encoder, 84
decoder, and two per-channel statistics vectors. Config: `dims: 3`,
`latent_channels: 128`, `out_channels: 3`, `patch_size: 4`,
`norm_layer: pixel_norm`, `causal_decoder: false`,
`timestep_conditioning: false`, `spatial_padding_mode: zeros`.

The decoder walks `decoder_blocks` in reverse. Every convolution is 3x3x3.

| | block | convolution | pixel shuffle | out |
|---|---|---|---|---|
| | `conv_in` | 128 → 1024 | | 1024 |
| 0 | `res_x` x2 | 1024 | | 1024 |
| 1 | `compress_all` m=2 | 1024 → 4096 | 2x2x2 | 512 |
| 2 | `res_x` x2 | 512 | | 512 |
| 3 | `compress_all` m=1 | 512 → 4096 | 2x2x2 | 512 |
| 4 | `res_x` x4 | 512 | | 512 |
| 5 | `compress_time` m=2 | 512 → 512 | 2x1x1 | 256 |
| 6 | `res_x` x6 | 256 | | 256 |
| 7 | `compress_space` m=2 | 256 → 512 | 1x2x2 | 128 |
| 8 | `res_x` x4 | 128 | | 128 |
| | `conv_out` | 128 → 48 | | 48 |

Rows 5 and 7 are the ones the public reference gets wrong: it would emit 1024
and 1024 there and carry 512 and 512 forward.

The output is 48 channels because `out_channels * patch_size**2` is 3 x 16 —
the last unpatchify turns 48 channels into 3 at four times the spatial
resolution. Total upsampling is 8x in time and 4 x 4 x 2 x 2 = 32x... in the
convolutions alone it is 2 x 2 x 2 in space from the two `compress_all` and the
`compress_space`, then the patch unpack multiplies by 4 again.

`DepthToSpaceUpsample` drops the first temporal frame after any shuffle with a
stride of 2 in time, which is what keeps a causal stack aligned. There is no
learnable normalization anywhere in the decoder: `pixel_norm` is
`x * rsqrt(mean(x^2) over channels + 1e-8)`, parameter-free, and the checkpoint
carries no norm tensors at all.

## What this means for the port

The reference cannot be used as an anchor as fetched. Two options, and the
first is much better:

- **patch the fetched reference** with the two-line change above and anchor
  against that. The evidence it is right is that all 84 tensors then line up,
  which is a strong constraint — 45 shapes across five distinct widths do not
  agree by accident. This is what the scratchpad does today.
- fetch a 2.5-era implementation if one is published. None was found under
  `Lightricks/LTX-Video`; the repo's `ltx_video` package is the 2.x line, while
  the DiT work in this port used `ltx_core`, which is a different package and
  is not on that repo.

Either way, **check shapes against the checkpoint before trusting any LTX
reference again.** The DiT and the text tower were both anchored against
`ltx_core`, which came with the model and matched. This one did not, and the
only reason it was caught is that the state dict was diffed before any code was
written against it.

## Cost, roughly

The decoder is 25 residual blocks and four upsamples of 3x3x3 convolutions at
128 to 1024 channels, run once per generation rather than once per denoising
step. h3.c already has `h3_gpu_conv3d_f32` and `h3_gpu_conv3d_same_f32` from
H3's own video VAE, so the kernels exist; what does not exist is the pixel
shuffle, the parameter-free pixel norm, and the block driver.

The audio side is a larger piece and is untouched: `ltx-2.5-audio-vae-bf16`
holds 1329 tensors, of which 102 are the VAE proper (2D convolutions) and
**1227 are a vocoder**. That is a separate subsystem, not a variation on this
one.
