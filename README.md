# H3ddle

H3ddle is an open-source native macOS workspace for local MiniMax H3 video
generation. Generated video, images, and audio are assembled on a deliberately
small two-track timeline: one visual lane and one audio lane.

The project is an independent SwiftUI/AppKit implementation. Model weights are
not stored in this repository or bundled by the scaffold.

## Current state

The first scaffold includes:

- a native macOS editor shell;
- ordered visual items and explicitly timed audio items;
- append, disable, and remove domain behavior;
- a provider-neutral Generate Studio with explicit prototype fallbacks;
- a versioned JSON-lines engine protocol and bundled native helper;
- persistent local-model selection and H3/Metal inventory validation;
- a pinned, resumable managed-model download with disk preflight, progress,
  cancellation, SHA-256 verification, and atomic installation;
- real prompt-to-video dispatch when a valid model and FFmpeg are available;
- reserved AVFoundation export boundaries; and
- public-boundary, unit, engine, and Xcode build checks.

Generated video plays with native macOS controls in the editor and can be copied
to a user-selected location. Export composition is still a placeholder.
Standalone audio still uses the prototype provider; `h3.c` does not expose
audio-only output. Image generation uses the community still recipe: the
shortest trained 22-frame H3 chunk, then the last decoded frame as a PNG.

## Requirements

- Apple Silicon Mac
- macOS 15 or newer
- Xcode 26 or newer
- XcodeGen 2.46 or newer
- FFmpeg and FFprobe for real generation

## Build

```sh
xcodegen generate
open H3ddle.xcodeproj
```

Open **Model not set** in the toolbar to choose an existing directory containing
the released `FL2VA` model tree, or download the pinned optimized package into
`~/Library/Application Support/H3ddle/Models`. Model weights stay outside the
repository and app bundle. `Ref2VA` is optional in the current protocol.

The first managed package is the approximately 53.9 GB FL2VA INT8 ConvRot
variant from `Comfy-Org/MiniMax-H3`, plus pinned official tokenizer/configuration
metadata. Both sources use immutable Hub revisions. This package is ready for
prompt-only FL2VA generation: Qwen and all 50 DiT layers stream from SSD, then
the F32 AudioVAE and F16 VideoVAE decode synchronized output. A native M1 Pro
smoke test completed the entire path through a 22-frame H.264/AAC file with
1.49 GiB peak tracked DiT residency. Visual-reference conditioning for this
optimized layout remains future work.

The app currently exposes a native 256×256 preview with four denoising passes.
This is the smallest canvas and denoising budget intended to produce a
recognizable result; it is still below H3's 20-pass quality default. The
duration control is a request because H3 rounds it up to a legal temporal
shape. Denoising reports progress for every transformer layer, VideoVAE work is
aggregated across temporal chunks, and Cancel terminates the job-specific
helper immediately so mapped weights and Metal allocations are released even
while a long GPU operation is active. Generated videos can be played in the
editor or copied to a user-selected location, and both live and completed
generation time are shown in the UI. Use the CLI for deliberate larger-canvas
or higher-step experiments.

Run all non-UI checks and build the application with:

```sh
Scripts/ci.sh
```

## Repository boundary

Do not add model weights or private implementation references. Read
`AGENTS.md`, `docs/architecture.md`, and `docs/product-contract.md` before making
architectural changes.
