# LTX-2.5 input-major transformer and F32 video attention

## Decision

Ship the distilled LTX-2.5 INT8 ConvRot transformer with every quantized
projection stored input-major, and automatically use F32 MPSGraph attention
for sufficiently long video sequences when its scratch allocation fits a
conservative memory budget.

Neither optimization requires an environment variable. The managed package
installs the repacked checkpoint at the canonical transformer path. The engine
reads the versioned tensor marker, validates every projection shape, and picks
the matching Metal kernel. F32 attention is enabled by default above 512 video
rows when four full-width F32 scratch tensors occupy no more than one sixteenth
of physical memory. The environment switches remain diagnostic opt-outs only.

## Exact repack

`Scripts/repack-ltx-input-major.py` transposes 1,344 INT8 projections across
48 dual-stream blocks. It preserves per-output scales, metadata, and all 5,885
other tensors byte-for-byte, then adds a versioned scalar layout marker. The
verification pass checks every converted matrix against the exact transpose,
checks every untouched tensor as raw bytes, and validates the complete tensor
set and metadata.

The released artifact is 21,504,034,388 bytes with SHA-256
`b39322c2d03cb85509b148b19f602275a88df8f86be48f28e0c38ba2b25f2dfb`,
pinned in the app to Hugging Face revision
`7597fb305b4cab9e7ff2c1d1e9551279c2932f0f`.

## Matched application result

The three runs used a 32 GiB M1 Pro, the same prompt and seed, a 512x512
five-second video, 65 frames, eight passes, and the LTX-2.5 Distilled package.
The generated images and videos were visually identical.

| path | denoising | total | change from regular |
|---|---:|---:|---:|
| regular output-major + BF16 attention | 458.0 s | 655.7 s | baseline |
| input-major + BF16 attention | 401.5 s | 598.3 s | -12.3% denoising; -8.8% total |
| input-major + F32 video attention | 369.2 s | 565.5 s | -19.4% denoising; -13.8% total |

The input-major step saved 57.4 seconds overall. F32 video attention then
saved another 32.8 seconds. Together they saved 90.2 seconds and made the
complete generation 1.16x faster. The video VAE remained about 172 seconds,
so it is now more than 30% of the total and the largest remaining phase.

## Compatibility and fallback

An output-major checkpoint with no marker continues to use the regular kernel.
A marked checkpoint with a missing or invalid schema is rejected instead of
producing a plausible wrong sample. On small sequences or when the memory guard
does not pass, attention stays BF16. Diagnostic builds may force the regular
checkpoint or disable F32 attention, but these controls are not part of the
released app configuration.
