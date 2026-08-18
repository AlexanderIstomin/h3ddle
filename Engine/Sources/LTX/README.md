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
| `ltx_video.c` | video VAE decoder: latent → frames, 8× in time and 32× in space |
| `ltx_connector.c` | the tower's features → DiT context, 8 blocks per stream |
| `ltx_text.c` | Gemma 4 12B int8, 48 layers, and the feature aggregation |
| `ltx_dit.c` | the dual-stream DiT and its sampler, 48 blocks, both latents |
| `ltx_generate.c` | prompt to clip: the five stages, in the order memory allows |

All five stages are here. `Vendor/h3.c/tests/ltx_generate.c` remains as the
driver — it keeps the reference comparisons, and it is where a disagreement is
reproduced.

The DiT was the one stage that is not a pure lift. The driver takes its
geometry from `-D` flags, because they sized the block helpers and one stack
array, and the service cannot ship a binary per clip length. So the shape
moved into a `run` struct and is threaded. That turned up a latent bug on the
way: the driver sizes its attention scratch from the *video* row count alone,
which overflows whenever a clip has fewer video tokens than the prompt has
context rows — 2x8x8 is 128 against a span of 256. No geometry it ever ran hit
it. The module sizes from the larger of the two.

The video VAE's **encoder** is deliberately not carried over. It exists to
condition on an existing clip, which nothing calls yet, and leaving it in the
driver keeps the module to what the service needs.

## The order of operations

`Vendor/h3.c/tests/ltx_clip.sh` runs the whole path as separate processes and
was the specification this module was written against. `ltx_generate()` now
does the same in one:

```
tokenize → Gemma tower → connector → denoise → video VAE + audio VAE/vocoder → mux
```

**The tower and the DiT do not fit in memory together** — 37 GB — so the tower
runs, hands back its features and is freed before the DiT store opens. That is
a real constraint rather than scaffolding, and it is the shape of
`ltx_generate.c`: load, run, free, in sequence, never holding both.

The tokenizer rides inside the text encoder checkpoint as a `tokenizer_json`
tensor rather than sitting beside it as a file, and this copy of it has a
**no-op post-processor** — no `<bos>` is prepended, unlike stock Gemma. h3.c's
SentencePiece path reproduces its ids exactly; `h3_gemma_tokenizer_test`
against the reference `tokenizers` library is what says so.

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

**The connector's registers go on the tail, and the prompt goes at the
front.** `_replace_padded_with_learnable_registers` compacts the unmasked rows
forward and then *flips the mask* before choosing between them and the
registers — that flip is the whole trick. Hand it a left-padded span instead
and the branch never runs at all: the prompt sits at rope positions `pad..255`
and the tower's output for pad tokens fills the slots the registers belong in.
It conditions plausibly either way, which is why it survived a day of being
looked at.

**The tower must not be run over its own padding.** The reference passes the
attention mask into the HF model; this engine has no masked GQA kernel, so a
padded run lets every real token attend to ~238 pads. Running the bare prompt
is *equivalent*, not approximate — Gemma's positions enter only through RoPE,
RoPE is relative, and translating every position by the pad length leaves every
attention logit unchanged. If a masked GQA kernel ever appears, this stops
being a workaround and starts being a choice.

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

Each check runs a real input through the module and writes the driver's own
output format, to be compared **byte for byte** with what the driver writes for
the same file. All five are identical today.

`ltx_dit_check` is the one that needs care: the geometry is an argument, so it
only means anything held against a driver built with the *same* `-D` flags and
run with the same seed and context. Its own header gives the pair of commands. The
arithmetic did not change in the move, so anything but identical bytes is the
lift's fault — which is the failure mode worth guarding, since an error latch
that short-circuits one branch early yields a plausible waveform from a
half-finished decode.

## Speed

512 square, 65 pixel frames, 8 steps, with nothing else on the GPU: **36 s a
denoise step**, 297 s for the sampler, and the eight steps within 1.8 s of each
other. The prefetch worker has already taken weight loading down to under a
second of that.

An earlier figure of 65 s a step was measured while another generation had the
GPU. It is not this engine's number, and it is the second time a timing taken
on a shared machine has had to be withdrawn here — see the prefetch A/B in
`ltx_dit.c`.

## Not done

Nothing is wired to the engine protocol, the manifest or the app yet — this is
the port, not the feature. `ltx_generate()` also hardcodes the published
filenames inside the snapshot's three directories, which is what an app-side
package will have to match or what will have to become a parameter.

Nothing longer than 2.7 s has ever been generated, and nothing but 512 square.
The resolution ladder, the guidance path and the 16→48 kHz bandwidth extender
are all unported.
