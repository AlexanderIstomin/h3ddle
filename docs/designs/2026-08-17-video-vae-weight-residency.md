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

Resident if the stored projections plus 8 GiB of headroom fit in `hw.memsize`.
Native F16 needs about 4.5 GiB and therefore fits the supported 16/24 GiB
machines; `H3_VAE_NATIVE_F16=0` restores the 9.0 GiB expanded estimate, which
needs a 32 GiB machine. `H3_VAE_RESIDENT=1` forces residency and `=0` forces
streaming.

Both estimates are counted from the decoder's own shape — 36 layers of qkv,
out, w1 and w2, plus the ends — rather than from the files: 4.5 GiB stored as
F16 or 9.0 GiB expanded to F32. Two earlier attempts to measure this from disk
were wrong in opposite directions: the weights live in a `vae/` subdirectory
so a flat scan of the package returned zero, and a recursive one would have
added the 21 GB transformer and the 27 GB text encoder to the decoder.

## Two things this leaves

**The int8 video VAE would take the resident path automatically**, being a
non-F16 checkpoint, and `h3.c` already prefers
`vae/minimax_h3_video_vae_int8_convrot.safetensors` when the package has one.
No package ships it. A local conversion of it existed and was deleted during a
disk cleanup as unreferenced — correct on the evidence at the time, and worth
re-deriving now.

**Residency does not remove the initial read, it amortizes it.** One pass over
the projection weights still happens. Native-F16 storage makes that pass about
4.5 GiB while retaining F32 arithmetic.

## Follow-up: remove repeated reads without residency

The low-memory path no longer has to choose between nine-gigabyte residency
and rereading the stack for every spatial tile. It now retains only one hidden
state per tile, loads block 0 once and runs it over every tile, then does the
same for block 1. Scratch activations remain shared and the per-tile arithmetic
sequence is unchanged. Extra hidden states are capped at 256 MiB; a larger
canvas is split into batches.

On the M1 Pro 512x512x22 oracle, the production auto-tile plan made four
288-pixel spatial tiles. The old streamed path loaded 144 blocks and decoded
in 63.44 seconds; layer-major streaming loaded 36 and decoded in 29.48 seconds,
a 2.15x speedup. The complete RGB buffers were byte-identical. Peak live GPU
storage rose from 0.676 to 0.728 GiB, while cumulative allocation fell from
36.540 to 9.507 GiB. `H3_VAE_LAYER_MAJOR=0` restores the old traversal for A/B
diagnosis; the resident path is unchanged.

## Done: native F16 projection weights

The decoder retains the checkpoint's large projection matrices as
IEEE F16 and explicitly widens them to F32 inside MPSGraph before matrix
multiplication. Inputs, biases, accumulation, and outputs remain F32. Small
shapes that the expanded path sends to a direct Metal kernel stay expanded;
this preserves the existing dispatch and its exact low-bit behavior.
`H3_VAE_NATIVE_F16=0` restores expanded F32 storage for diagnosis.

The first app A/B appeared to reject this path, but the setting had been placed
in Xcode's launch-argument list rather than its environment table and was never
active. The corrected local 512x512x22 oracle is byte-identical across all RGB
frames. Expanded F32 loaded/decoded in 7.78/20.73 seconds with 9.454 GiB peak
Metal residency; native F16 took 0.88/20.12 seconds with 4.942 GiB peak. That
is 28.51 versus 21.00 seconds for the VAE phase, a 7.51-second reduction and
roughly half the memory.

The corrected cold app A/B, with prefetch disabled so it could not hide the
load, confirmed the result. Expanded F32 loaded/decoded in 10.9/23.0 seconds;
native F16 took 2.5/23.1 seconds. The VAE phase fell from 33.9 to 25.6 seconds,
an 8.3-second reduction, and both images looked identical. Native F16 is now
the default. Automatic prefetch remains limited to 32 GiB or above because it
overlaps decoder residency with the final transformer pass; its gain and the
native-F16 load reduction are not additive.

## Done: final-pass decoder prefetch

`H3_VAE_PREFETCH=1` prepares the resident still-image decoder on a background
thread while the final transformer pass runs. It changes no tensor values or
operation order. On a cold two-pass 512x512 M1 Pro run, denoising was stable at
67.89 seconds without prefetch and 67.48 seconds with it; the synchronous work
after denoising fell by about ten seconds. The emitted PPM frames had identical
SHA-256 hashes. The cold 32 GiB app A/B confirmed it at eight passes: transformer
time was 268.4 versus 268.3 seconds, decode stayed 17.1 seconds, VAE load fell
from 10.1 to 0.0 seconds, and total time fell from 302.9 to 292.6 seconds.

The trade is temporary overlap with the decoder's Metal storage: originally
9.45 GiB, now about 4.94 GiB with native F16. Prefetch remains conservative
and defaults on only at 32 GiB or above. `H3_VAE_PREFETCH=0` restores
synchronous loading, while `=1` can force the path for diagnosis.

## Measuring

Same discipline as the tile work, and it earned its keep twice here. Stub
inside a real run rather than benchmarking alone; interleave and swap order.
And compare *frames*, not files — an mp4 embeds a creation timestamp, so
identical output hashes differently, and a truncated capture reports a
difference that is not there. Both of those happened during this measurement
and both were briefly believed.
