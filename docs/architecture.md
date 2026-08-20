# Architecture

H3ddle is a native macOS application with three strict boundaries.

```text
SwiftUI application
├── H3ddleCore              versioned project and visual / audio / T1 timeline
├── H3ddleGeneration        provider-neutral generation workflow
├── H3ddleMedia             preview, program compositor, AVFoundation/VideoToolbox export
├── H3ddleModels            pinned catalogs, download and installation
└── H3ddleEngineProtocol    JSON-lines IPC contract
        └── Engine service  h3.c, Metal, FFmpeg and model lifetime
```

## Timeline

The visual track is the program backbone. Its items are ordered and derive
their start time from the preceding visual items. A transition pulls the
incoming clip over the outgoing tail so the two clips overlap for the
transition duration. Visual clips store a shared canvas object transform
(fit or cover, then translation, uniform scale, and continuous rotation).
The UI supports append, duplicate, reorder, disable, trim, split, canvas
placement, rotate, overlapping cut transitions, clip effects, and remove.

The audio track shares the program clock but stores explicit start times. New
generated or imported audio still appends at the current audio-track end.
Clips can be duplicated after the source and dragged to a new start time or
order. Imported files are copied into an app-managed media folder. Disabling an
audio item preserves its position and renders silence. Removing it also preserves
later start times, leaving a gap rather than silently desynchronizing the mix.

T1 is an overlay lane above V1. Titles store explicit start times, may overlap,
and draw after the visual. Preview lasts `max(visual, audio, text)`. Export lasts
`max(visual, text)` when T1 is included; a title-only program still cannot
export. Audio shorter than that end produces silence. Trailing audio is detected
against the export end and must be surfaced before export.
Generated video may contain native audio, controlled independently from the
dedicated audio track. `ProgramCompositor` places the current visual and any T1
overlays on the program canvas so preview and export can share one draw path. Film grain and
chroma key use stitchable Metal Core Image kernels compiled at first use. The program
monitor displays those composed frames; `AVPlayer` is only used for audio and
as a video-frame source.

## State and persistence

Domain models are Codable and carry an explicit schema version. They do not
depend on UI or media frameworks. Project persistence will use versioned JSON
and app-managed media directories rather than SwiftData.

## Generation

`GenerationProvider` is the app-facing abstraction. Tests and unsupported
generation kinds use a fake provider. `EngineGenerationProvider` translates
validated video and still requests into engine commands and consumes progress,
completion, failure, and cancellation events without loading weights in the app
process. Stills follow the community recipe: one 22-frame H3 chunk, keep the
last decoded frame, skip FFmpeg mux. Audio follows the community 32×32
soundtrack recipe: run the joint model at the mechanical minimum canvas,
then skip the video decoder outright and write the audio VAE's own samples
as a WAV. Only video output needs FFmpeg.

The engine service is deliberately single-job. Keeping `h3_ctx` in that process
allows later interactive cache reuse while still permitting the app to reclaim
all inference memory by terminating the service.

Audio generation remains provider-neutral. The current H3 public API produces joint
video and audio, not a standalone audio-only result. A real provider must expose
capabilities and either use a dedicated local audio model or explicitly derive
audio from an audiovisual render; see decision 0004.

## Model access

The app stores a security-scoped bookmark for the user-selected model folder.
Validation happens in the bundled helper using `h3_load_dir`, so both the model
inventory and active Metal device come from the same boundary used for inference.
The helper and runtime Metal source are copied into `Contents/Helpers`; model
weights are never copied into the app.

Managed packages use immutable manifests containing the source repository,
revision, optional per-file source overrides, expected sizes, SHA-256 hashes,
memory guidance, and upstream license link. `H3ddleModels` streams downloads to
an Application Support staging
directory, resumes with HTTP byte ranges, verifies each file, then moves the
whole package into place on the same volume. Cancellation preserves partial
files. CI exercises the same transport with tiny local fixtures and never needs
model weights or network access.

Download completion and engine compatibility are separate states. The first
curated INT8 package is marked `ready` only after the native path passed the
optimized Qwen stack, compact AdaLN schedule, all 50 streamed DiT layers, both
single-file VAE decoders, and H.264/AAC muxing on an M1 Pro. The DiT keeps two
I8/F32 layer slots and peaked at 1.487 GiB of tracked Metal residency in the
real forward probe. The VideoVAE expands one F16 transformer block at a time;
it rereads weights per spatial/temporal tile and is a correctness-first path
rather than the final throughput design. This optimized package supports
prompt-only FL2VA generation, not visual-reference conditioning.

The native app adapter currently applies an optimized-checkpoint preview
preset: native 256×256 rendering, four denoising passes, all 50 transformer
blocks, and no denoiser reuse. The previous 64×64/two-pass plumbing smoke was
below h3.c's recognizable-output floor and is not a user-facing preset. Live
denoising stills are optional and remember the last choice; each still is a
full VideoVAE pass and does not change the encoded video. The first model
inspection starts one helper, compiles Metal once, and keeps `h3_ctx` with
cache enabled until the app quits, the model folder changes, idle timeout,
or memory pressure. Cancellation is cooperative; if the helper does not
acknowledge within a short window the process is killed and the next job
respawns it.
