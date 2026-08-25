# H3ddle

H3ddle is an open-source native macOS studio for local generative media. It
creates video, still images, music, sound effects, and cloned-voice speech
entirely on Apple silicon, then assembles generated or imported media on a
deliberately small program timeline: one text lane, one visual lane, and one
audio lane.

The project is an independent SwiftUI/AppKit implementation over a vendored
Metal engine. Model weights are never stored in this repository or bundled with
the app; they are downloaded from pinned, checksummed Hugging Face revisions
into Application Support.

[Download the latest macOS build](https://github.com/AlexanderIstomin/h3ddle/releases)
from GitHub Releases. The newest development build is listed first; see its
release notes for macOS first-launch instructions.

## Current state

Generation, on the local Metal engine:

- MiniMax H3 and LTX-2.5 video with synchronized sound, including prompt,
  keyframe, reference-image, and reference-audio workflows;
- Z-Image-Turbo stills from a prompt or source image, plus H3 still generation;
- Stable Audio 3 music, sound effects, and ambience;
- Qwen3-TTS speech in a voice cloned from a short reference recording;
- model-specific aspect ratios, resolutions, durations, and sampling controls;
- managed model packages pinned by revision and SHA-256, downloaded only when
  chosen and kept outside the application bundle;
- live progress, denoising previews where supported, and remaining-time
  estimates projected from each run's measured pace; and
- reproducible generation statistics containing the settings another user
  needs to repeat a result.

Editing and output:

- a program timeline with a T1 text lane, filmstrip, and waveform previews;
- canvas objects with direct gesture editing, text items, visual effects and
  transitions, and undo/redo;
- drag-and-drop import of existing video, image, and audio files;
- H.264, H.265, or ProRes export with loudness normalization; and
- Download and Copy statistics on any finished generation.

Each model exposes only the controls it supports. H3 keeps its structured
prompt, quality ladder, and block-cache path; LTX and Z-Image use their native
distilled sampling controls; Stable Audio and Qwen3-TTS provide dedicated
audio workflows. Generation and export use system media frameworks and do not
require a cloud service.

Transformer-block thinning and core reuse remain available for engine research,
but are hidden from the normal studio because both can move a result away from
the full model. Launch with `H3DDLE_ENABLE_H3_ADVANCED_CONTROLS=1` to expose
them. H3 masked-source video inpainting is also experimental and appears only
with `H3DDLE_ENABLE_H3_MASKED_SOURCE=1`. With neither variable set, H3ddle runs
all 50 transformer blocks and leaves core reuse off, including when older saved
settings contain experimental values.

## H3 performance

The released H3 stack is optimized for the Mac it is running on; no environment
variables are needed. Its main implemented paths are:

- pre-quantized INT8 ConvRot transformers whose 200 core projections are stored
  input-major, avoiding a transpose in every projection without changing tensor
  values;
- bounded, overlapped transformer and Qwen weight streaming for 16/32 GB Macs,
  with prompt-length-aware Qwen kernels and a prefetched text-encoder ring;
- prefetched, layer-major Video VAE decode, native F16 execution when memory
  permits, and resident VAE reuse between generations;
- compact hybrid-reference overlays, replacing a second 20.97 GB transformer
  download with 43.55 MB while preserving the full FL2VA base;
- exact-attention scheduling, activation-buffer aliasing, retained immutable
  graph bindings, and fused patch/head kernels selected by GPU generation; and
- M5-only, runtime-guarded Metal 4/TensorOps projection, quantization, QKV/RoPE,
  and GPU-sampler paths. M3 and earlier Macs retain their measured faster Metal
  paths rather than being forced through M5 kernels.

The short-shape measurements below are completed A/B generations on a 32 GB
M1 Pro. Input-major is an exact layout change; the paired outputs were identical.

| H3 workload | Regular layout | Input-major layout | Measured gain |
|---|---:|---:|---:|
| Turbo FL2VA still, 512×512, 8 passes | 278.7 s | 256.5 s | 8.0% end to end; 8.9% denoise |
| Turbo Ref2VA still, 512×896, 8 passes | 631.4 s | 595.9 s | 5.6% end to end; 4.8% denoise |

Long 768p H3 video is a very different workload: the open release uses full
attention, whose cost grows approximately with the square of sequence length.
The official model describes sparse attention but does not yet publish that
inference implementation. The following therefore reports engineering
projections, not completed 10-second benchmarks. The workload is 16:9
1344×768, 243 frames (10.125 seconds at 24 fps), audio enabled, all 50 blocks,
reuse and block cache off. “Unoptimized” means the Standard 20-pass,
output-major INT8 stack. “Full stack” means the shipped Turbo 8-pass,
input-major INT8 weights plus the automatic app/engine optimizations above.
Turbo is step-distilled, so this is a product-stack comparison rather than an
exact-output A/B.

| Mac configuration | Evidence level | Unoptimized | Full optimized stack | Stack gain |
|---|---|---:|---:|---:|
| M1 Pro, 16-core GPU, 32 GB | M1 measurements extrapolated by the app's dense-attention model | ~43 h | ~16 h | ~2.7× |
| M3 Max, 40-core GPU, 128 GB | hardware-scaled projection | ~14–20 h | ~5.5–7.5 h | ~2.7× |
| M5, 10-core GPU, 32 GB | hardware- and M5-kernel-scaled projection | ~9–14 h | ~3–5.5 h | ~2.7× |
| M5 Max, 40-core GPU, 128 GB | hardware- and M5-kernel-scaled projection | ~5.5–7 h | ~2–2.7 h | ~2.7× |

The M1 projection is fitted to H3ddle's recorded 1,625- and 5,095-row runs and
then evaluated at 73,402 rows; real thermals, SSD contention, prompt length,
and references can move it substantially. The newer-Mac ranges scale that
model conservatively using Apple's published GPU and memory characteristics:
[M1 Pro has 200 GB/s](https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/),
[M3 Max has up to a 40-core GPU and 400 GB/s](https://support.apple.com/en-us/117736),
[M5 has a 10-core GPU with a Neural Accelerator per core and 153 GB/s](https://www.apple.com/newsroom/2025/10/apple-unveils-new-14-inch-macbook-pro-powered-by-the-m5-chip/),
and [M5 Max reaches 40 GPU cores, 128 GB, and 614 GB/s](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/).
Those peak comparisons are bounds, not H3 benchmarks. The exact sparse Sol
research path remains disabled by default because current quality testing did
not justify enabling approximate attention.

## Requirements

- Apple Silicon Mac
- macOS 15 or newer
- Xcode 26 or newer
- XcodeGen 2.46 or newer
- FFmpeg and FFprobe are optional. Generation, muxing, and the reference
  media a Mac usually holds all go through the system frameworks; FFmpeg is
  only consulted for containers those decline, such as Matroska

## Build

```sh
xcodegen generate
open H3ddle.xcodeproj
```

Open the model picker in the toolbar to install a managed video, image, or audio
package, or add a compatible package already on this Mac. Downloads go into
`~/Library/Application Support/H3ddle/Models`; model weights stay outside the
repository and app bundle. Every managed source uses an immutable Hub revision
and every file is verified by SHA-256 before install. Packages reuse identical
files where possible so installing a related model does not duplicate shared
weights.

The two managed MiniMax H3 packages are reference-capable: Standard + Hybrid
References and Turbo + Hybrid References. Both ship full input-major FL2VA
transformers. All 200 quantized core projections are pre-transposed for the
faster Metal path; the tensor values and prompt-only output are unchanged.
The standard artifact is published in
[PulpCut/MiniMax-H3-INT8-ConvRot](https://huggingface.co/PulpCut/MiniMax-H3-INT8-ConvRot),
and the distilled artifact remains in
[PulpCut/MiniMax-H3-Turbo-INT8-ConvRot](https://huggingface.co/PulpCut/MiniMax-H3-Turbo-INT8-ConvRot).
Both managed packages add the same 43.55 MB Ref2VA AdaLN
overlay instead of a second 20.97 GB transformer. Full standard Ref2VA weights
remain in the
[upstream repository](https://huggingface.co/Comfy-Org/MiniMax-H3), while both
full Turbo Ref2VA layouts remain in the
[H3ddle weight repository](https://huggingface.co/PulpCut/MiniMax-H3-Ref2VA-Turbo-INT8-ConvRot)
for exact-weight use and comparisons.

For another compatible optimized MiniMax H3 transformer, prepare the same
layout locally:

```sh
python3 -B Scripts/repack-h3-input-major.py /path/to/transformer.safetensors
```

The source model is not changed. H3ddle automatically uses the validated
layout marker when the resulting `_input_major.safetensors` checkpoint is
selected. FL2VA and Ref2VA transformers must be repacked separately if both
flows should benefit.

The managed Standard + Hybrid References and Turbo + Hybrid References
packages keep their respective FL2VA transformers and overlay only Ref2VA
AdaLN blocks 25–49, reducing the additional reference weight file from
20.97 GB to 43.55 MB. Build the same overlay locally without changing a source
checkpoint:

```sh
python3 -B Scripts/build-h3-hybrid-adaln.py /path/to/ref2va.safetensors
```

Place the resulting file at
`diffusion_models/minimax_h3_ref2va_pruned_int8_convrot_hybrid_adaln_25_49.safetensors`.
If a full Ref2VA checkpoint is also installed, `H3_REF2VA_HYBRID=1` opts into
the hybrid for comparison; if only the overlay is installed, the loader uses
it automatically. The hybrid intentionally changes generation and is not an
exact-output or speed optimization. Standard and Turbo Ref2VA produce the same
overlay bytes, so the managed packages share one pinned download.

The managed LTX-2.5 transformer is likewise stored input-major. Its 1,344 INT8
projections are exact transposes of Lightricks' regular layout and its other
5,885 tensors are byte-identical. On a 32 GiB M1 Pro, a matched five-second,
512-square, eight-pass run fell from 655.7 to 565.5 seconds overall: 13.8%
faster end to end and 19.4% faster in denoising, with visually identical
quality. The optimized layout and the memory-gated F32 video self-attention
path are selected automatically; the released app needs no environment
variables. To prepare a compatible checkpoint locally:

```sh
python3 -B Scripts/repack-ltx-input-major.py /path/to/transformer.safetensors
```

Denoising reports progress for every transformer layer, and the video decoder
reports its own blocks, so a long decode does not look like a hang. Cancel
terminates the job-specific helper immediately, releasing mapped weights and
Metal allocations even during a long GPU operation. Use the CLI in
`Engine/Vendor/h3.c` for experiments the app does not expose.

Run all non-UI checks and build the application with:

```sh
Scripts/ci.sh
```

## Credits

H3ddle itself is Apache-2.0. It builds on other people's work, under their own
terms — the app downloads these weights rather than bundling them, and links
each licence before the download starts.

**Engine.** The Metal engine is a fork of [h3.c](https://github.com/antirez/h3.c)
by Salvatore Sanfilippo, MIT licensed, vendored under `Engine/Vendor/h3.c`.

**MiniMax H3** — video, image, and joint audio generation. Weights by MiniMax
under the MiniMax H3 Community License Agreement, fetched from
`Comfy-Org/MiniMax-H3`. The input-major and step-distilled transformers under
`PulpCut` are conversions of those weights made for this project and carry the
same licence.

**MiniMax H3 TAE** — the 9 MB preview decoder behind live denoising previews,
by Kijai, Apache-2.0.

**LTX-2.5** — video with synchronized audio by Lightricks, used under the
LTX-2.x Community License Agreement. H3ddle downloads an INT8 ConvRot package
whose tensor values match the published distilled release.

**Z-Image-Turbo** — text-to-image and image-to-image generation by Tongyi-MAI,
Apache-2.0 licensed. H3ddle downloads an INT8 ConvRot repackaging together with
the official image encoder. Live denoising previews use madebyollin's TAEF1
decoder, MIT licensed.

**Qwen3-TTS** — reference-voice speech generation by the Qwen team,
Apache-2.0 licensed. H3ddle downloads a safetensors conversion of the 12 Hz
0.6B Base release.

**Powered by Stability AI.** Music, sound effects, and ambience come from
[Stable Audio 3 Small SFX](https://huggingface.co/stabilityai/stable-audio-3-small-sfx),
© Stability AI Ltd., used under the Stability AI Community License — free for
research and non-commercial use, and for commercial use below USD $1M annual
revenue after registering with Stability AI. The weights H3ddle installs are
repackaged from Stability's own MLX release into safetensors; tensor values are
unchanged, and the licence and NOTICE ship inside the package. Its text encoder
is **T5Gemma**, provided under and subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).

Model outputs belong to you, as far as each licence allows. Note that the
Stability licence forbids using outputs to train competing foundational
generative models.

## Repository boundary

Do not add model weights or private implementation references. Read
`AGENTS.md`, `docs/architecture.md`, and `docs/product-contract.md` before making
architectural changes.
