# H3ddle engine

The engine is a separate process so model lifetime, Metal allocation, crashes,
and cancellation do not destabilize the SwiftUI application. Communication is
versioned JSON Lines over standard input and output.

The service builds the pinned `Engine/Vendor/h3.c` revision as a native library.
It currently supports:

1. a capability handshake that checks whether FFmpeg is available;
2. model inventory and Metal-device validation through `h3_load_dir`;
3. one prompt-to-video generation job at a time;
4. H3 progress callbacks mapped to `EngineEvent` values; and
5. cooperative cancellation while native inference is active.

The app bundles the service in `Contents/Helpers` and `h3_shaders.metal` in
`Contents/Resources/H3Engine`. It passes explicit Homebrew FFmpeg and FFprobe
locations when they are available.
Reference inputs and denoising-preview transport are intentionally not exposed
by protocol version 3.

The app uses the real service for video when a validated model is selected.
Image and standalone-audio generation remain behind the prototype provider. CI
builds and handshakes with the service but does not require model weights.

## Native quantized-weight spike

The pinned engine can validate and load row-major `I8` safetensors weights with
one `F32` scale per output channel. A weight-only Metal path keeps activations
and outputs in BF16, so pre-M5 Macs retain the smaller serialized and resident
weight without expanding the complete matrix to BF16. M1 and newer GPUs use a
simdgroup-matrix kernel; the existing M5 path can instead quantize activations
and use Metal 4 TensorOps.

`tests/test_quantized_weights.c` checks exact serialized byte/scale reload and
compares the portable Metal result with a CPU reference on every supported Mac.
On M5 it additionally checks the stored representation against h3.c's runtime
quantizer. Set `H3_BENCH_PORTABLE_INT8=1` when running the test directly to
compare the portable kernel with the BF16 path at an H3 projection shape.

The engine recognizes H3ddle's pinned optimized single-file layout, inventories
it through protocol v3, and enables prompt-only FL2VA generation. `make real-int8
H3_OPTIMIZED_MODEL=/path/to/package` loads all four block-0 INT8 matrices and
F32 scales directly from the 20.97 GB checkpoint. It applies the checkpoint's
H256 ConvRot transform, runs attention and MLP projections on Metal, and
compares selected rows with independent CPU calculations. The same probe
precomputes all 50 compact AdaLN projections from the checkpoint's F32 curve
and F16 `[output, 8]` weights.

The optimized Comfy checkpoint keeps Q, K, and V projection rows in the normal
contiguous `[Q all heads | K all heads | V all heads]` order. This differs from
the grouped per-head ordering used by the released MiniMax tree. The adapter
selects the contiguous unpacker for the optimized checkpoint; interpreting it
as grouped mixes the attention roles and produces block-like video plus
broadband audio noise. The real block-0 probe exercises this checkpoint-specific
path before its attention calculation.

This proves the optimized transformer's storage/compute and timestep-schedule
boundaries on an M1 Pro. The optimized Qwen language path also reads only the
prompt's BF16 embedding rows, streams one I8/F32 layer at a time with one-layer
prefetch, applies ConvRot to all seven projections, and completes all 50 layers
with a measured 0.909 GiB peak Metal weight/activation residency for a two-token
probe.

The optimized VAE adapters now execute both packaged checkpoints directly.
The F32 AudioVAE accepts the compact checkpoint's baked convolution weights
and remains stage-streamed, measuring 0.141 GiB peak Metal residency for a
one-token decode. The F16 VideoVAE reads its checkpoint-owned latent statistics
and converts one transformer block at a time directly into shared F32 Metal
storage. A complete 36-block, five-frame probe measures 0.252 GiB peak
residency and 9.031 GiB cumulative allocations. Larger tiled decodes remain
functional but reread the F16 weights for each tile/chunk, so native F16
kernels or a cross-tile layer schedule are still needed for practical speed.
The optimized DiT now streams two alternating I8/F32 layer slots. A real
50-layer forward measured 1.487 GiB peak Metal residency, and a two-step native
smoke test completed tokenizer, Qwen, denoising, both VAEs, and H.264/AAC muxing.
The managed package is therefore enabled for prompt-only FL2VA generation;
visual conditioning remains unsupported for this layout.
