# H3ddle

H3ddle is an open-source native macOS workspace for local MiniMax H3
generation. Video, images, and audio are produced entirely on your own machine
and assembled on a deliberately small two-track timeline: one visual lane and
one audio lane.

The project is an independent SwiftUI/AppKit implementation over a vendored
Metal engine. Model weights are never stored in this repository or bundled with
the app; they are downloaded from pinned, checksummed Hugging Face revisions
into Application Support.

## Current state

Generation, on the local Metal engine:

- prompt-to-video with synchronized audio, still images, and standalone audio;
- start and end keyframes, and ordered reference images through a Ref2VA
  transformer, both working on the optimized single-file packages;
- four managed model packages — standard and step-distilled, each with or
  without reference support — pinned by revision and SHA-256, sharing their
  common weights by hardlink so a second package costs only what differs;
- prompts composed in the three-field schema MiniMax H3 was trained on;
- resolutions named by short edge from 352p to 1088p, derived from the chosen
  aspect ratio;
- block caching, which replays cached tail-block residuals on stable denoising
  steps for about 40% less work at standard step counts;
- denoising previews decoded through a 9 MB tiny autoencoder in roughly 180 ms,
  showing the model's current estimate of the finished frame; and
- a remaining-time estimate projected from each run's own measured pace.

Editing and output:

- a two-track program timeline with one visual lane and one audio lane;
- canvas objects with direct gesture editing, text items, visual effects and
  transitions, and undo/redo;
- drag-and-drop import of existing video, image, and audio files;
- H.264, H.265, or ProRes export with loudness normalization; and
- Download and Copy statistics on any finished generation.

Stills render one representative frame of a short H3 clip, at either the
detailed 22-frame chunk or a 5-frame chunk that is about three times quicker.
Audio renders the joint model for its soundtrack and writes a 32 kHz stereo WAV
directly, with no FFmpeg process involved.

Known limitation: prompts asking for ambience rather than speech tend to return
speech. This is H3's own bias toward dialogue, not a defect in how the app
drives it — the canvas size was ruled out by measuring 32, 64, 128, and 256
square, and the audio lane's sigma schedule was checked against the reference
and is correct. Dedicated sound-effect and ambience generation is in progress.

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

Open **Model not set** in the toolbar to choose an existing directory containing
the released `FL2VA` model tree, or download the pinned optimized package into
`~/Library/Application Support/H3ddle/Models`. Model weights stay outside the
repository and app bundle. `Ref2VA` is optional in the current protocol.

Managed packages come from `Comfy-Org/MiniMax-H3` for the base weights and
`PulpCut` for the two step-distilled transformers converted for this project,
plus pinned official tokenizer and configuration metadata. Every source uses an
immutable Hub revision and every file is verified by SHA-256 before install.
Packages share their text encoder, VAEs, preview decoder, and metadata, so
installing a second one downloads only the transformer that differs.

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
`Comfy-Org/MiniMax-H3`. The two step-distilled transformers under `PulpCut`
are conversions of those weights made for this project and carry the same
licence.

**MiniMax H3 TAE** — the 9 MB preview decoder behind live denoising previews,
by Kijai, Apache-2.0.

**Powered by Stability AI.** Sound effects and ambience come from
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
