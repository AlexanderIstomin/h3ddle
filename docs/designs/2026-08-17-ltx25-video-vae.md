# LTX-2.5's video VAE, and the reference that does not match it

Written before implementing the decoder, because the first thing checking the
released weights against the public implementation found was that they are
different models — and the difference is invisible to a loader that matches on
tensor names.

**The decoder now runs** — `tests/test_real_ltx_video_vae.c` against
`gen_ltx_vae_anchor.py`, all fourteen stages within 1.2e-05 of peak, 14 of 15
mutations caught. What that took, and what the patch turned out to rest on, is
in "What running it settled" at the bottom.

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

## What running it settled

The patch is right, and there are now two independent reasons rather than one.
The shape argument was always a bit circular — I chose the patch that made the
shapes line up, so of course they line up. Two things fix that.

First, the patch does not invent a mechanism. `out_channels_reduction_factor`
is already a parameter of `DepthToSpaceUpsample`, already passed by
`compress_all`. The change passes an argument the class already accepts to the
two siblings that were not given it. That is a plausible upstream omission, not
a plausible coincidence.

Second, and this is the one that does not depend on my reading at all: the
patched encoder and decoder **round-trip a structured image at 30.8 dB**. A
decoder whose blocks are shaped right but ordered or wired wrong still loads
and still runs; it does not reconstruct edges and gradients. Shapes agreeing
shows the parts fit. Only a round trip shows they fit *together*.

### The conventions, all mutation-checked

| convention | the other plausible reading |
|---|---|
| temporal padding **replicates**, spatial padding **zero-fills** | both the same, on the same convolution |
| `compress_time`/`compress_all` drop their **first** output frame, unconditionally | drop the last, or drop none |
| pixel norm is parameter-free, epsilon **inside** the root | epsilon outside, or a learned weight |
| unpatchify packs channels **(c, t, w, h)** | (c, t, h, w) — transposes every 4×4 patch |
| the shuffle decomposes channels **major-to-minor** as (c, p1, p2, p3) | minor-to-major, or depth innermost |

The dropped frame is why 2 latent frames become 3, then 5, then 9 rather than
4, 8, 16 — it is what keeps the stack on the `8*(k-1)+1` frame count.

### The epsilon is inert, and that is measured rather than assumed

Setting pixel norm's `1e-8` to zero changes nothing this test can see, so no
mutation catches it. That is a true statement about the model at these
magnitudes, not a hole in the test — the smallest mean square anywhere in the
stack is **5.0e+02**, about 5e10 times the epsilon. The test now prints that
margin and fails if it ever closes, which is the useful form: "shown not to
apply here" rather than "untested".

### Two engine facts worth keeping

**Uploads must not happen inside an open command buffer.** A convolution
encoded after an upload into the same buffer reads the allocation as zeros and
emits its bias alone — which is exactly what happened, and was diagnosable only
because the bias was recognisable in the output. Load first, then
`h3_gpu_begin`. The same applies in reverse at the end: submit before freeing
anything the encoded work still reads.

**Layout is NDHWC with OIDHW weights.** OIDHW is torch's own `Conv3d` layout,
so weights load with no rearrangement and only activations need permuting.
`h3_gpu_conv3d_same_f32` zero-pads height and width and leaves depth alone,
which is precisely the split LTX needs: pad time explicitly by replication
first, then let the kernel handle the spatial axes.

### What is still missing

The four pixel shuffles run on the host, one readback each. At the anchor size
the largest moves 1.2M floats and costs little, but a real decode wants a
kernel — `b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)`, a pure gather. It
is the only piece of the decoder the GPU cannot do end to end, and it is a
performance gap rather than a correctness one. Deliberately left out of the
shared Metal files.

A 9-frame 128×128 decode takes 2.65 s, nearly all of it weight loading at this
size. The encoder is unported — the round trip above runs it in torch.
