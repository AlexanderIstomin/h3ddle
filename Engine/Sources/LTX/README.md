# LTX-2.5

Text to video **with a synchronized soundtrack**, 22B distilled, eight steps.
Kept outside the vendored engine on the same footing as SA3, Qwen3TTS and
ZImage, so upstream merges of `h3.c` stay clean while still reaching its
headers and GPU layer.

Two streams denoise together — video and audio — through one DiT, and the
soundtrack is conditioned on the same prompt and on the video itself. That is
the reason to carry a second video engine at all: H3 already writes a joint
soundtrack, so LTX earns its place on quality, not on capability.

## What runs where

Everything of consequence is on the GPU through `h3_gpu`. The DiT's six
attentions and two feed-forwards per block, both VAEs, and the vocoder's
109 alias-free activations are all existing kernels — **this engine needed no
new Metal**, which is most of why it was worth porting.

| file | what |
|------|------|
| `ltx_audio.c` | audio VAE decoder and BigVGAN vocoder: latent → 16 kHz stereo |

Still to move out of `Vendor/h3.c/tests/`, where they currently work as
standalone drivers: the DiT sampler (`ltx_generate.c`), the video VAE decoder
(`ltx_decode.c`), and the Gemma tower plus embeddings connector.

## The order of operations, and why it is not one process yet

`Vendor/h3.c/tests/ltx_clip.sh` runs the whole path and is the specification
this module is being written against:

```
tokenize → Gemma tower → connector → denoise → video VAE + audio VAE/vocoder → mux
```

**The tower and the DiT do not fit in memory together** — 37 GB — so the tower
runs, writes its context, and exits before the DiT opens. That is a real
constraint on the eventual `ltx_generate()`, not scaffolding: it has to load,
run and free in sequence rather than holding both.

## Things that are true and not obvious

Each of these cost a real bug, and none was catchable by a component test —
every one was found by rendering something and looking at or listening to it.

**The rope's time axis is in seconds.** `create_initial_state` divides axis 0
by the fps right after `get_pixel_coords`. Feeding pixel frames scales the axis
by the frame rate and, worse, puts video on a different footing from audio,
whose bounds are already seconds — and the two streams attend to each other
through those positions. This axis has now been wrong three separate ways:
latent cells, then pixel frames, then seconds.

**The audio length follows from the video's.**
`AudioLatentShape.from_video_pixel_shape` is `round(frames / fps × 25)`, where
25 is `sample_rate / hop_length / downsample`. It is not a free parameter.

**The context span is the tokenizer's, not the register count.** The tokenizer
left-pads to 256; the connector's 128 learnable registers *replace padded
positions* rather than setting a length. Reading 128 as the span silently
halves the context.

**The audio latent normalizes through the patchifier.** The statistics are 128
wide while the latent is 8 channels, because normalization lives in the
patchified space the DiT works in: `[8, T, 16]` reads as `[T, 128]` with
channel index `c * 16 + f`.

**The three resblocks in a vocoder stage are averaged, not summed**, and AMP1
applies no activation before the upsample. Both produce a plausible waveform
when read the other way.

## Testing a stack that amplifies

The vocoder is ill-conditioned in one place: stage 1's resblocks multiply a
difference in their input by about thirteen. So a composed stage-by-stage
comparison measures the model's conditioning rather than this port's accuracy,
and an F32-vs-F64 floor does not bound an independent implementation — the same
stage magnifies rounding 2.9× and an equally-sized MPS-vs-torch disagreement
13.6×.

Gate each stage **from the reference's own input**. Held that way the error is
flat at 2e-07 to 2.2e-06 from the first stage to the last, at every length
tried. The composed drift is worth printing beside it, with its amplification
factor, but not worth failing on.

## Harness

`harness/` holds the checks. They are not in the build — each has its own
`main`, and they need model files that cannot live in the repository. Build one
directly:

    clang -O2 -fobjc-arc -o ltx_audio_check \
        harness/ltx_audio_check.c ltx_audio.c \
        ../../Vendor/h3.c/h3_gpu.m ../../Vendor/h3.c/h3_weights.c \
        ../../Vendor/h3.c/h3_safetensors.c \
        -I../../Vendor/h3.c -I. \
        -framework Foundation -framework Metal \
        -framework MetalPerformanceShaders \
        -framework MetalPerformanceShadersGraph -framework Accelerate -lm

`ltx_audio_check` decodes a real latent and writes a WAV to be compared **byte
for byte** with the one `h3_ltx_audio_decode` writes for the same file. The
arithmetic did not change in the move, so anything but identical bytes is the
lift's fault — which is the failure mode worth guarding, since an error latch
that short-circuits one branch early yields a plausible waveform from a
half-finished decode.

## Not done

Nothing is wired to the engine protocol or the app yet — this is the port, not
the feature. The DiT sampler, the video VAE and the Gemma tower still live as
standalone drivers under `Vendor/h3.c/tests/`, and `ltx_generate()` is a
contract in `ltx_generate.h` with no implementation behind it.
