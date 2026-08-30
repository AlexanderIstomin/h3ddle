# Native FastH3 integration

## Decision

Support FastVideo's Dense and learned-VSA FastH3 previews as distinct, versioned H3 model
profile. Convert its merged Diffusers checkpoint into H3ddle's input-major
INT8 ConvRot layout and preserve the full trained timestep function by storing
exact AdaLN lookup rows for the fixed four-call T2VA serving schedule.

Do not treat H3ddle's existing Sol attention as FastH3 VSA. VSA uses package
format 2 and its own parity-tested tile-64 Metal implementation.

## Why a direct LoRA merge is invalid

The compact H3ddle transformer replaces MiniMax H3's 2,688-dimensional time
embedding and per-block `[96768, 2688]` AdaLN projections with an
eight-dimensional pruned curve. FastH3 fine-tunes the original full matrices.
The adapter therefore cannot be multiplied into the compact template without
changing the trained function.

The four-call serving schedule has only seven distinct clean-time rows across
the video-shift-12 and audio-shift-3 streams:

```text
0, 1/37, 1/10, 1/13, 1/4, 1/5, 1/2
```

The converter evaluates FastH3's full time embedding and each trained AdaLN
matrix at those rows, rounds at the same BF16 boundary as FastVideo, and adds
the results to the optimized transformer. This costs about 68 MB rather than
restoring the roughly 10 GB full AdaLN/time path.

The Diffusers release stores the two SwiGLU input projections as
`[value, gate]`; h3.c's Comfy-derived fused kernels consume `[gate, value]`.
Their tensor shapes are identical, so the converter explicitly swaps the two
14,336-row halves before quantization. A shape-only conversion misses this and
produces structured chromatic noise despite completing without an engine error.

## Package contract

The transformer carries these metadata fields:

```text
h3.generation_profile = fasth3
h3.default_steps = 4
h3.sigma_schedule = serving
h3.conditioning = t2va
h3.fasth3.version = 1 (Dense) or 2 (VSA)
```

Both formats carry U32 version/step marker tensors, the seven F32 time rows, 50
BF16 block lookup tables, and one BF16 final lookup table. The C loader rejects
unknown versions, non-four-step schedules, Beta schedules, reference
conditioning, mismatched time rows, and partial/missing lookup tables.
Format 2 additionally requires 50 input-major INT8 learned gate projections,
their F32 scales, a tile-size marker of 64, and a sparsity marker of 0.9.

The app propagates the model profile over the versioned engine protocol. The
engine accepts only video output for FastH3, forces all 50 blocks, and disables
core reuse, block cache, token reduction, and audio refinement. These controls
could otherwise invalidate the checkpoint's four-call contract.

FastH3 is also restricted to 124–362 frames and a minimum 480-pixel short
edge. Those are the released 5–15 second and smallest documented Apple-Silicon
shapes. Both the Swift worker and native C entry point reject smaller plumbing
previews before generation.

## VSA implementation

FastVideo VSA is trained around all of the following behavior:

- segment-pure prefix tiles and spatial 4×4×4 video-token tiles;
- FP32 per-head Q/K/V tile means using true sizes for edge tiles;
- dense prefix query rows and always-selected prefix keys;
- exact top-k video-key tiles at the configured sparsity;
- sparse attention normalized only over selected token tiles; and
- a separate dense pooled attention result, broadcast per query tile and
  multiplied by a learned per-token, per-head, per-channel gate.

Sol attention uses contiguous 64-token blocks, a statistical threshold, local
and sink routing, and a different approximation for omitted blocks. It cannot
consume FastH3's learned gate and is not numerically equivalent.

The native path builds the segment/video geometry once, uploads true ragged
tile sizes and the inverse row map, and reuses one activation arena across all
50 blocks. Each block projects its learned gate from the same ConvRot-modulated
input as QKV. Four ordered GPU stages pack Q/K/V/gate, pool Q/K/V in FP32,
select video top-k while producing dense pooled attention, and evaluate exact
sparse token attention plus the learned correction. Output is written directly
to packed H3 row order, avoiding a separate untile pass.

The deterministic GPU-vs-CPU test exercises ragged segment tiles, multiple
video tiles, prefix-dense routing, video top-k, exact token softmax, and the
gate branch. Its initial Apple-GPU parity measured cosine 0.9999986, mean
absolute error 0.000066, and maximum error 0.000707 after BF16 output rounding
on the expanded ragged-3D/top-k-greater-than-one case.

## Expected performance

Dense FastH3 halves the number of DiT calls relative to H3ddle Turbo (four
instead of eight), while retaining the existing input-major INT8 ConvRot,
fused QKV/RoPE, fused MLP, and native decode paths. End-to-end gain must be
measured on Apple Silicon; it will not equal the 14× B200 headline because that
headline compares four-call 90%-sparse VSA against a 49-call dense baseline.
The native VSA path retains H3ddle's input-major INT8 ConvRot projections and
SSD overlap, while reducing exact video-query attention to every prefix tile
plus 10% of video tiles. FP32 pooling, top-k, and the trained correction add a
linear/tile-level cost; end-to-end Apple-Silicon measurements remain the
acceptance criterion.

## Local validation

The corrected Dense package completed a real 480×480, 124-frame, four-call run
on a 32 GB M1 Pro. Euler denoising took 744.529 seconds; the complete DiT phase
took 745.339 seconds and peaked at 3.396 GiB of tracked storage. The input-major
streamer read 72.182 GiB at 3.167 GiB/s, with only 0.003 seconds of unhidden I/O
wait, so dense attention—not model reads—is the limiting path at this shape.
The decoded representative frame was coherent and prompt-related. A direct
256×256, 22-frame request is now rejected before generation by both the app
worker and native engine.

A controlled 512×512, 124-frame, four-call comparison on the same M1 Pro used
the same prompt, soundscape, seed, and cache settings. Dense completed in
958.174 seconds and VSA in 696.043 seconds: VSA saved 262.132 seconds, a 27.4%
end-to-end reduction or 1.377× throughput. Both outputs were 5.175-second H.264
at 24 fps with stereo AAC. This is the local end-to-end result, not a claim that
every pipeline phase is 1.377× faster; subsequent runs persist phase timing and
sampled peak helper memory so the attention-specific gain can be separated from
model load, encoders, VAE decode, and muxing.
