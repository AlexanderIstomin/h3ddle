# LTX-2.5's audio path, resolved

Written before implementing it, because that ordering has now paid for itself
twice on the video side. Everything here is read off the released checkpoint
and `ltx_core`, not from library defaults.

## It is three subsystems, not one

`ltx-2.5-audio-vae-bf16.safetensors` holds 1329 tensors in three parts:

| part | tensors | what it is |
|---|---|---|
| `audio_vae.{encoder,decoder}` | 102 | a taming/LDM-style **2D** VAE over log-mel |
| `vocoder.vocoder` | 667 | **BigVGAN**, mel → 16 kHz stereo |
| `vocoder.bwe_generator` | 557 | **BigVGAN**, 16 kHz → 48 kHz |
| `vocoder.mel_stft` | 3 | STFT/mel bases, the *bandwidth extender's* front end |

Plus two `audio_vae.per_channel_statistics` vectors.

## The reference matches, unpatched

`ltx_core.model.audio_vae` — the same package the DiT, the text tower and (as
of this session) the video VAE are anchored against — builds all three from the
checkpoint's own config and matches exactly:

```
audio encoder: 44/44   missing 0, unexpected 0, wrong shape 0
audio decoder: 56/56   missing 0, unexpected 0, wrong shape 0
vocoder:     1227/1227 missing 0, unexpected 0, wrong shape 0
```

No patch, no drift. `pip install ltx-core`. This is the reference to use; see
`2026-08-17-ltx25-video-vae.md` for what happened the one time a different one
was used instead.

## The VAE: 2D over log-mel, and time is *height*

`ddconfig`: `mel_bins 64`, `z_channels 8` (`double_z`), `ch 128`,
`ch_mult [1,2,4]`, `num_res_blocks 2`, `norm_type: pixel`,
**`causality_axis: height`**, `mid_block_add_attention: false`,
`downsample_time: false`. Stereo in and out, 16 kHz, hop 160, n_fft 1024.

The tensor layout is `(batch, channels, frames, mel_bins)`, so the "height"
axis is **time**. Three consequences, each a place to be quietly wrong:

- **the 3×3 convolutions are causal along height**: two frames of padding in
  front, none behind, while the mel axis pads symmetrically. Structurally the
  same split as the video VAE's replicate-in-time / zero-in-space, on different
  axes and for a different reason;
- **`mid_block_add_attention` is false**, so `attn_1` is an Identity and the
  mid block is just two resnets. The checkpoint carries no attention tensors,
  which is the tell;
- **the frame arithmetic is causal in both directions.** Encoding, a stride-2
  downsample takes 29 frames to 15. Decoding, upsampling takes 8 → 15 → 29, one
  short of a clean doubling each time. `_denormalize_latents` states the rule:
  `target_frames = 4 * latent_frames`, less 3 when the axis is causal.

Measured shapes, from `gen_ltx_audio_vae.py`:

```
mel  [ 2, 29, 64]  ->  latent [ 8,  8, 16]  ->  mel [ 2, 29, 64]
enc: conv_in 2->128, down0 128 @29x64, /2 -> 15x32, down1 ->256,
     /2 -> 8x16, down2 ->512, mid, norm, conv_out ->16 (double_z)
dec: conv_in 8->512, mid, up2 512 @8x16, x2 -> 15x32, up1 ->256,
     x2 -> 29x64, up0 ->128, norm, conv_out ->2
```

## The latent is normalized, and it happens *through the patchifier*

Same finding as the video half, with a twist that makes it easy to miss:

```python
# AudioEncoder._normalize_latents
means = torch.chunk(latent_output, 2, dim=1)[0]
latent_patched = self.patchifier.patchify(means)          # [B,8,T,16] -> [B,T,128]
latent_normalized = self.per_channel_statistics.normalize(latent_patched)
return self.patchifier.unpatchify(latent_normalized, latent_shape)
```

The statistics are **128-wide while the latent is 8 channels**, because the
normalization is applied in the *patchified* space the DiT works in —
`8 z-channels × 16 mel-latent bins = 128`. A port that reads the shapes and
tries to normalize the 8-channel tensor directly will not even fit.

The scales are larger than the video half's:

```
std   0.7422 .. 2.0469
mean -1.9062 .. +2.7344
```

`AudioDecoder._denormalize_latents` is the exact inverse, and — as established
on the video side — **no round trip can check either of them**, because the two
cancel. Assert the boundary tensors.

## The vocoder is BigVGAN, and h3.c already has one

`vocoder.vocoder`: `conv_pre` 128→1536 (k=7), six `ConvTranspose1d` upsamples
1536→768→384→192→96→48→24 with kernels `[11,4,4,4,4,4]` and rates
`[5,2,2,2,2,2]` (product 160 = the hop), 18 AMP1 resblocks (6 stages × kernels
`[3,7,11]`, dilations `[1,3,5]`), alias-free `snakebeta` throughout,
`conv_post` 24→2 with **no bias**.

`h3_audio_vae.c` is already a BigVGAN driver for H3 — same architecture, same
resblock kernels and dilations, different constants:

| | H3 | LTX-2.5 |
|---|---|---|
| upsample rates | 5,5,2,2,2,2,2 | 5,2,2,2,2,2 |
| upsample kernels | 9,9,4,4,4,4,4 | 11,4,4,4,4,4 |
| sample rate / hop | 32000 / 800 | 16000 / 160 |

and `h3_gpu_alias_free_snake_f32`, `h3_gpu_snake1d_f32`,
`h3_gpu_weight_norm_f32` and the conv1d variants all exist. So the vocoder is
plausibly mostly a re-parameterization rather than new work — the thing to
check first is whether the driver is written against those tables or against
H3's constants.

Two details that will matter: `ups.N.weight` is `[in, out, k]`, i.e.
**ConvTranspose1d**, not a plain convolution; and weight norm is already folded
(there is no `weight_g`/`weight_v` pair anywhere), so `h3_gpu_weight_norm_f32`
is not needed on this path.

The bandwidth extender is a second BigVGAN — `conv_pre` 128→512, rates
`[6,5,2,2,2]`, taking the 16 kHz waveform back through an STFT and mel (the
stored `mel_stft`, n_fft 512, hop 80) and out at 48 kHz.

## What is done and what is next

**The decoder runs** — `tests/test_real_ltx_audio_vae.c`, twelve stages inside
the measured floor (worst 2.6e-06 of peak), 15 of 16 mutations caught, 0.5 s
for 29 frames. It needed no new kernel: `h3_gpu_conv3d_same_f32` covers the 2D
convolution by mapping frames onto the depth axis and mel onto width with
height 1, so depth is left unpadded for the explicit causal pad while mel is
same-padded by the kernel — exactly the split this VAE wants.
`h3_gpu_nearest2x_nhwc_f32` is the upsample.

Two things the implementation pinned that reading alone had not:

- **the causal pad is zeros, not replicated frames.** The video VAE replicates
  edge frames in the same position; this one zero-fills. Nothing but
  `CausalConv2d` says so, and both readings produce a plausible mel;
- **the statistics index is `c * 16 + f`, channel-major.** Reading it mel-major
  is a permutation of the same 128 values — it fits, it runs, and it is wrong.
  Caught only because the boundary tensor is compared directly, which is the
  lesson the video half taught the hard way.

The reference's own F32-vs-F64 spread is a flat 1.0e-06 to 2.9e-06 the whole
way down, unlike the video stack which spikes by two orders at one stage, so a
single bound is defensible here.

Not implemented: **the encoder** (the mirror of the above, and only needed to
*condition* on existing audio rather than to generate it) and **the vocoder**.

The vocoder is now the last thing between a DiT audio latent and a waveform,
and per the table above it is architecturally what `h3_audio_vae.c` already
does. The first question to settle is whether that driver is written against
its rate tables or against H3's constants.

The input mel is built in the generator rather than taken from the checkpoint,
because the VAE's own front end (n_fft 1024, hop 160) is preprocessing and only
the *extender's* bases ship. Pinning the exact causal-STFT convention is a
vocoder-stage question and is deliberately deferred.
