# The video VAE was reading its weights six times

Once the DiT's projections were retuned (see
`2026-08-17-int8-gemm-tile-for-h3.md`), the video VAE decode became the larger
half of a generation: 180 s of a 241 s run, against 55 s of denoising. This is
what it was doing, and what it does now.

## It is not arithmetic

Stubbing each family of operations inside a real decode, the way the DiT's
share was found:

| | of the 174 s decode |
|---|---|
| the four block projections | ~22 s (12%) |
| attention | ~5 s (3%) |
| norms, SwiGLU, ConvRot, residual adds | ~0 s |
| **unaccounted** | **~147 s (85%)** |

The progress events said where it went: **216 of them for a stack of 36
layers**, because the decoder runs once per spatial tile and the clip tiles
3x2. Six passes over the same 36 blocks, and each pass did this:

```c
for (int index = 0; index < LAYERS; index++) {
    load_block(vae, index, ...);      /* read F16, widen to f32, upload */
    run_block(vae, index, ...);       /* ~0.125 s of actual work */
    free_block(&vae->blocks[index]);  /* discard it */
}
```

0.81 s a block, of which 0.125 s was arithmetic. The other 0.685 s was
fetching weights that had already been fetched, five more times to come.

## The fix already existed

`run_resident_tile` and `load_resident_weights` were written for exactly this
— their own note reads "avoids rereading 9 GiB per spatial tile" — but
`checkpoint_streams_f16` sent every F16 checkpoint down the streaming path
regardless of whether the machine had room. The package ships
`minimax_h3_video_vae_fp16.safetensors`, so it always streamed.

The dtype was the wrong question. The right one is whether residency fits:

| | 2-pass 640x352 clip |
|---|---|
| streaming | 230 s, 241 s |
| resident | 106 s, 105 s, 110 s |

**2.2x, with the frames bit-identical** — decoded to raw RGB and hashed, the
same `25dc1f79...` every configuration has produced.

## How the choice is made

Resident if `9.0 GiB + 8 GiB of headroom` fits in `hw.memsize`, so a 32 GB
machine takes it and a 16 GB one — the minimum these packages ask for — keeps
streaming and behaves exactly as before. `H3_VAE_RESIDENT=1` forces residency,
`=0` forces streaming.

The 9.0 GiB is counted from the decoder's own shape — 36 layers of qkv, out,
w1 and w2, plus the ends, as f32 — rather than from the files. Two earlier
attempts to measure it from disk were wrong in opposite directions: the
weights live in a `vae/` subdirectory so a flat scan of the package returned
zero, and a recursive one would have added the 21 GB transformer and the 27 GB
text encoder to an estimate of a 5 GB decoder.

## Two things this leaves

**The int8 video VAE would take the resident path automatically**, being a
non-F16 checkpoint, and `h3.c` already prefers
`vae/minimax_h3_video_vae_int8_convrot.safetensors` when the package has one.
No package ships it. A local conversion of it existed and was deleted during a
disk cleanup as unreferenced — correct on the evidence at the time, and worth
re-deriving now.

**Residency does not remove the reread, it amortizes it.** One pass over 9 GiB
still happens, and at ~24 s it is now the largest single item in the decode.
Holding the weights as bf16 rather than widening to f32 would halve both the
residency and that first read, at the cost of a bf16-weight variant of
`h3_gpu_linear_f32`.

## Measuring

Same discipline as the tile work, and it earned its keep twice here. Stub
inside a real run rather than benchmarking alone; interleave and swap order.
And compare *frames*, not files — an mp4 embeds a creation timestamp, so
identical output hashes differently, and a truncated capture reports a
difference that is not there. Both of those happened during this measurement
and both were briefly believed.
