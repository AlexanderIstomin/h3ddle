# AuK native Metal integration plan

Status: revised after the Mac feasibility and speed experiments, 2026-09-13.
This is an implementation plan; AuK native inference is not implemented yet.
It supersedes the earlier suggestion of a packaged Python inference helper.

## Required execution architecture

All shipping AuK neural-network inference must run through native C/Metal in
the existing out-of-process engine service. No Python interpreter, PyTorch,
Transformers, `uv`, Python worker, or Python fallback may be required by the
app or model package. Native CPU code may handle tokenization, file I/O,
scheduling and scalar bookkeeping. Tensor inference uses Metal, with narrowly
scoped Objective-C/MPSGraph adapters where needed.

Follow the existing [YuE2 backend](../../Engine/Sources/YuE2/yue2.c),
[C bridge](../../Engine/Sources/YuE2/yue2_bridge.c) and
[validation workflow](../experiments/2026-09-13-yue2-native.md): Swift engine
request → validated C ABI → native model pipeline → Metal operations → atomic
audio output, with progress, cancellation and deterministic ownership.
Reuse compatible `h3_gpu` buffers, safetensors loading, matrix multiplication,
normalization, RoPE, attention and convolution primitives. Validate AuK shapes,
masks and precision independently; YuE2's model implementation is not a
drop-in AuK engine. Extend shared kernels only with regression coverage.

YuE2 currently releases allocations after each job. Start AuK with that clear
ownership model for parity, then add bounded model residency because the AuK
tests justify it. Keep residency in the serialized engine service, with
eviction before another large model loads, on memory pressure, cancellation
cleanup and helper shutdown. Do not introduce an application-wide singleton.

Python remains permitted only for developer reference export, offline weight
conversion, and experiments outside the app bundle. Installation must yield
weights, tokenizer/configuration data, a versioned manifest and notices—not an
inference environment. A clean runtime smoke test must succeed without any
development environment installed or available through PATH.

## Decisions from measured results

All measurements below are three-second requests on one M1 Pro with 32 GiB,
using PyTorch 2.7.1/MPS. They establish candidates for the native port, not
native performance guarantees. See the [full speed report](../experiments/2026-09-13-auk-speed.md)
for timing boundaries, output differences and small-sample limitations.

| Experiment | Observed result | Revised decision |
| --- | --- | --- |
| Resident FP32 Base, 32 steps | Mean 102.02 s uncached; 21.17 s exact-input cached; samples and latents identical | Add bounded exact-input caches after native parity; the 4.82x gain applies only to repeats |
| BF16 Qwen + FP32 DiT/VAE, Base 32 | Warm mean 24.26 s, about 4.20x baseline; highest recorded driver memory 16.50 GiB | Preferred native precision candidate; preserve FP32 reference fixtures |
| Same mixed loader, Base 16 | Warm mean 11.35 s, about 8.99x original baseline; outputs changed | Experimental preview after quality review; retain 32-step default |
| FP32 DiT 8 steps, fixed conditioning | Sampling 4.83 s versus warmed 32-step 19.04 s; greater output deviation | Research option, not initial product default |
| BF16 DiT | Mean sampling 32.82 s versus FP32 22.99 s, with timing drift | No speed justification; do not make it the default |
| FP16 DiT | First generation produced NaN/Inf latents | Reject this path unless a later numerical fix passes independent validation |
| Mixed Flash, released four steps | Warm uncached 6.65–10.76 s; cached 5.66 s | Preserve released grid; no controlled warm FP32 comparison supports a speedup factor |

The mixed Base loader took 23.74 s to load in a fresh process, compared with
112.76 s for the earlier full-FP32 loader. Disk/shader caches were not cold.
Flash's mixed-precision waveform correlation to the earlier FP32 sample was
0.98876, a larger change than Base's 0.99999. Correlation, finite samples and
absence of clipping do not establish intelligibility, speaker similarity or
editing quality. Listening and task-specific evaluation remain release gates.

Use BF16 Qwen storage without expanding the whole model to FP32. Keep DiT,
VAE and solver state FP32 initially, and preserve numerical sensitivity in
normalization, RoPE and learned layer fusion. Native BF16 load/compute behavior
must be measured on M1; PyTorch autocast choices do not specify a Metal kernel
implementation or guarantee its speed. Avoid transient full-model duplicate
allocations: restoring FP32 during one experiment exceeded the MPS memory cap.
Do not advertise 16 GiB compatibility or a production minimum from these runs.

## Native model pipeline

Proposed new source directory: `Engine/Sources/AuK/`, with a public internal
header, C bridge, checkpoint loader, conditioner, DiT, flow solver and VAE.
Add the bridge to `H3Native.h`, register sources in `Engine/Package.swift`, and
dispatch from `H3ddleEngineService`. These are planned changes, not existing APIs.

1. Native input processing: decode/resample reference audio, reproduce Qwen's
   audio feature extraction and tokenizer, and construct exact task templates,
   masks and positions. Foundation/AVFoundation adapters stay out of domain
   models. Compare tokens and preprocessing tensors to the pinned reference.
2. Qwen2.5-Omni-3B Thinker text/audio path: implement its audio tower, text
   transformer and requested hidden-state extraction. Port AuK's normalized,
   learned weighted layer fusion. Qwen3-TTS embeddings are not interchangeable.
   Extract only verified dependencies from the checkpoint; exclude unused
   vision/Talker/token-to-waveform tensors. Omit a logits head only after
   proving hidden-state inference does not depend on it.
3. Reference VAE encoder: implement the actual posterior and sampling policy,
   yielding 64-channel, 50 Hz latents. Record reference randomness separately
   from target noise so caching cannot silently change seeded behavior.
4. AuK denoiser: port all 10 double-stream and 20 single-stream blocks, width
   1536 and 24 heads, including text/reference/target attention, masks, position
   encoding and conditioning. Compare individual blocks before full sampling.
5. Native flow solver: retain Base's reference 32 steps, CFG 2 and sway −1.
   Keep Flash CFG 0 and its fixed grid
   `[0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1]`.
   Flash must not inherit arbitrary Base step controls or YuE2's midpoint
   solver semantics. Add Base 16 only after the quality gate; keep 8 experimental.
6. Native VAE decoder: port the released BigVGAN-based flow VAE to 24 kHz mono
   output. Validate lengths, convolution padding and any later tiling at seams.
   Emit measured sample duration and write output atomically; reject nonfinite
   tensors and leave no successful asset after cancellation or failure.

Cache reference latents and complete conditioner outputs only when keys match
source content/range, instruction, preprocessing, checkpoint and converter
revisions, precision policy and relevant seeds. Bound cache bytes and invalidate
on model replacement. Do not transplant YuE2 AR-prefix KV caching into AuK's
joint denoiser: later-layer text/reference states depend on the changing target.
Only demonstrably invariant computations can be cached across solver steps.
Quantization, layer skipping, sparse attention and approximate block reuse
remain separate experiments, with no promised speedup or default enablement.

## Full feature scope

Expose all 16 upstream task families through typed options and deterministic
instruction templates. A task is advertised only after its native acceptance
cases pass; two successful MPS samples do not validate the full table.

| Task | Required inputs and controls | Acceptance focus |
| --- | --- | --- |
| Zero-shot TTS | Script, reference voice, target duration | Intelligibility and speaker similarity |
| Instruction TTS | Script, voice description, target duration | Text accuracy and described voice |
| Speech content editing | Source, replace/insert/remove text and anchors | Requested words and preservation of other content |
| Lyric editing | A cappella source, original/new lyrics | Lyric accuracy and melody preservation; offer vocal extraction first for a mix |
| Pitch editing | Source, signed semitones | Pitch change, intelligibility and timing |
| Rate editing | Source, positive rate factor | Requested rate and duration; estimate source duration divided by factor |
| Volume editing | Source, signed dB | Loudness change and clipping |
| Emotion editing | Source, preset or custom emotion | Emotion and content preservation |
| Timbre editing | Source, voice description | Timbre change with content preservation |
| De-accent | Accented source | Pronunciation change and intelligibility |
| Nonverbal editing | Source, add/remove breath/laugh/cough, placement | Event and location accuracy |
| Whisper conversion | Source, whisper or normal direction | Both directions and preserved words |
| Enhancement | Source, denoise/dereverb/cleanup/restoration instruction | Noise/reverb and telephone, muffling, clipping or dropout restoration cases |
| Speech separation | Overlapping speakers, selected speaker order | Target retention and interference suppression |
| Music separation | Mix, singing-only or all-human-voices selection | Vocals and accompaniment rejection; not a general instrument stem splitter |
| Target speaker extraction | Mixture, phrase identifying the speaker | Correct speaker selected and retained |

Use the native Generation Studio for the two TTS modes and an audio-editing
panel for source-based tasks. Reuse voice-library recordings, not another
model's cached embeddings. Keep task capability, Base/Flash selection and
valid controls visible; reject unsupported combinations before loading weights.
Start with evaluated English/Chinese templates. Do not inherit another speech
model's language list or claim Portuguese support without testing it.

Source+target context and memory limits must come from native measurements.
An upstream UI's 30-second limit is not proof of a universal model limit.
Begin with short validated clips, report limits in engine capabilities, and
test increasing reference and target lengths separately before raising them.
Automatic long-form chunking requires separate continuity and voice tests.

Prompt enhancement, transcription helpers, Gradio/ComfyUI integration and
training utilities are upstream workflow accessories rather than additional
AuK inference heads. Core tasks must work offline through native templates.
Any later automatic transcription or prompt enhancement needs its own native
backend and validation. Offline fine-tuning/conversion may use external developer
tools; imported checkpoints require a supported native manifest. No upstream
Python demo, training environment or SGLang server becomes an app dependency.

## Protocol, assets and model installation

Add Codable/Sendable task and option value types to the framework-independent
protocol/domain layers. Extend the versioned handshake with AuK variants,
validated tasks, duration/context limits and precision/step capabilities.
Current protocol is 19 at review time; allocate the next version when the
implementation lands rather than colliding with ongoing changes.

Persist the task, original instruction, source asset/range, reference identity,
checkpoint revisions, seed policy, requested duration and sampling/precision
policy in queued jobs and generation recipes. Validate finite numeric controls,
required source inputs, mutually exclusive options and compatible manifests.
Test queue restoration, migrations and unsupported-version rejection.

Edits create new assets without overwriting originals. Store actual output
duration and explicit audio start time on the common timeline clock. Preserve
audio append/duplicate/reorder behavior. If region editing promises untouched
audio outside a selection, implement native trim/splice with seam tests; do
not assume whole-clip generative editing preserves samples outside the edit.

Share Qwen and VAE package data between Base and Flash, keep one DiT variant
active, verify pinned hashes and tensor layouts, and preserve licences/notices.
Weights stay outside the repository. Model-folder inspection must distinguish
valid native assets from an incomplete reference environment. A managed
download can follow the initial local-folder integration; it must not install
a Python runtime.

AuK is MIT, but the required Qwen2.5-Omni-3B checkpoint has the
[Qwen research licence](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/f75b40e3da2003cdd6e1829b1f420ca70797c34e/LICENSE).
Treat commercial distribution/use as a separate licensing gate; converting to
Metal or asking users to download weights does not remove those terms. An
alternative encoder is not a drop-in licensing fix and would require separate
training and validation. Research feasibility work can continue independently.

## Implementation order and completion gates

1. **Reference fixtures and package contract.** Pin the revisions recorded in
   the feasibility report. Add developer-only export/check tools modeled on
   YuE2's native harness. Export fixed tokens, features, hidden states, fusion,
   reference posterior, target noise, block outputs, solver states and WAVs.
   Define explicit per-stage tolerances before accepting optimized results.
2. **Native component parity.** Complete tokenizer/audio preprocessing,
   conditioner, VAE and DiT/solver checks with Metal API validation enabled.
   Use identical exported tensors to separate RNG differences from numerical
   errors. YuE2's tolerances do not automatically apply to AuK. No app exposure
   until Base and Flash complete valid native end-to-end requests.
3. **Helper and full task integration.** Add the versioned protocol, validated
   options, package discovery, asset provenance and native UI. Run all 16 task
   cases on both advertised variants with listening and task-appropriate
   measures such as word error, speaker similarity, pitch/rate/loudness error,
   and separation quality. Retain failed capabilities as unavailable.
4. **Measured optimization.** Compare fresh-process load, warm uncached and
   exact-input cached runs using fixed fixtures and repeated alternating
   baselines. Record conditioning, sampling, decoding, full latency, peak
   memory, output quality and real-time factor across several clip lengths.
   Validate BF16 Qwen, then residency/caches, then Base 16 independently.
   Native throughput targets are set from these results, not the MPS ratios.
5. **Production acceptance.** Run through the bundled helper with restricted
   PATH, no Python environment, offline operation and `MTL_DEBUG_LAYER=1`.
   Check cancellation during each stage, memory-pressure eviction, repeated
   jobs, model switching, corrupt weights, crash recovery and partial-file
   cleanup. Inspect app resources and launch paths for runtime dependencies.
   Require quality and licensing gates before a general release.

Normal CI remains weight-free: fake providers, request validation, queue and
recipe persistence, timeline/export behavior and native shape/mask/solver
tests. Run `Scripts/ci.sh`; real-model parity and performance tests are an
explicit optional suite with external fixtures, following YuE2's pattern.

The immediate deliverable is a native parity harness and complete C/Metal
pipeline, followed by feature integration. A working Python probe or a faster
isolated kernel does not satisfy the native engine completion gate.

## Evidence

- [Mac feasibility and pinned assets](../experiments/2026-09-13-auk-mac.md)
- [Speed experiments and reproducible comparisons](../experiments/2026-09-13-auk-speed.md)
- [YuE2 native implementation and measured parity](../experiments/2026-09-13-yue2-native.md)
- [AuK model card](https://huggingface.co/tencent/AuK)
- [Pinned upstream implementation and cookbook](https://github.com/Tencent-Hunyuan/AuK/tree/0dfd4d39015078351217b09a40e263355aaa646b)
