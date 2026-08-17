# Bringing the retuned int8 GEMM to H3 and LTX-2.5

Deferred until the Z-Image lane is integrated. Written down because the
measurement is done and the work is not, and because it is worth more than
anything remaining in the image lane.

The filename says H3 because that is what it was written for. LTX-2.5 was
measured afterwards and settles a converter default rather than a migration,
so it is a section here rather than a note of its own.

## What was found

Z-Image's port drove `h3_gpu_linear_i8_weight_bf16` hard enough to notice that
its tile geometry is badly chosen for the shapes these models actually run.
The kernel computes an 8x8 output tile per simdgroup, so a Kx8 slice of weight
serves only eight rows and the matrix is re-read `rows / 8` times. Retiling to
64x40 and storing the weight input-major instead of output-major measured, on
**H3's own DiT shapes** (5376 hidden, 56 heads, 14336 FFN, 4096 rows):

| projection | 8x8, as shipped | 64x40, input-major | |
|------------|-----------------|--------------------|---|
| qkv        | 1977 ms         | 264 ms             | 7.49x |
| attn out   | 694 ms          | 91 ms              | 7.65x |
| fc1        | 2684 ms         | 353 ms             | 7.60x |
| fc2        | 1603 ms         | 183 ms             | 8.74x |
| **block**  | **6958 ms**     | **891 ms**         | **7.81x** |

Flat across row counts — 7.22x at 512 rows, 7.33x at 1024, 7.81x at 4096 —
because both tiles re-read the weight in proportion to rows and only the ratio
between them matters. Z-Image measured the same ~7.5x from the same starting
point.

Two changes produce it, and they are independent:

- **the tile.** Eighteen geometries were swept against four shapes; 64x40 won
  all of them. Registers cap the tile and the cliff is sharp and not
  predictable from arithmetic: 64x40 is 106 floats a lane and wins, while 96x32
  and 64x48 at 124-128 floats lose about half.
- **the layout.** The tile reads two weights a lane one output column apart.
  Stored `[output][input]` those land `input_dim` bytes apart and consecutive
  lanes `2 * input_dim` apart, so every lane of a simdgroup takes its own cache
  line on the dominant traffic. Stored `[input][output]` both reads are
  adjacent. Worth 9% on top of the tile.

Also available, separately measured: **attention in f32 rather than bf16**
above about 512 rows. MPSGraph's bf16 SDPA reaches 0.97 TFLOP/s on shapes of
this kind where its f32 path reaches 2.21, and casting three inputs up and the
result back still wins once the sequence is long enough for the quadratic
saving to cover the linear cost. Worth about a tenth of a Z-Image forward at
4128 tokens.

## What has to change

Nothing has moved. The retuned kernel was added as a separate entry point
(`h3_gpu_linear_i8_weight_bf16_square`) rather than by changing the existing
one, precisely so the video and speech paths kept the geometry they were tuned
against until somebody measured them.

Callers still on the 8x8 tile:

- `h3_dit.c` — four sites, the video DiT
- `h3_text_encoder.c`
- `h3_video_vae.c` — via the f32 variant, which has no retuned counterpart yet

Callers of `h3_gpu_sdpa_bf16` that may want the f32 switch: the video DiT, the
vision encoder, the video VAE, SA3's DiT. **Not** the TTS talker, which decodes
one token at a time and so never reaches the threshold.

For each model that moves, its package converter must write the int8 matrices
input-major. Transposing is a layout change only — values are untouched, and
ConvRot rotates the activation rather than the weight, so quantization is
unaffected. Three things had to agree when Z-Image did this, two of them
quietly: any permutation of output channels (Z-Image permutes q and k rows for
its rope convention) and any fusion along output channels (w1/w3 for SwiGLU)
both address what is now the minor axis, so they move from whole-row operations
to within-row ones. Neither would have raised an error — both would have
produced a plausible wrong result.

Note also that `h3_weight_load_i8_linear` infers the scale count from the
matrix's first dimension. That stops being the output count after transposing,
so weight and scales must be loaded separately.

## Do this first

**Nobody has measured what fraction of H3 video generation is the DiT GEMM.**
7.8x on a third of the runtime is 1.4x overall, and this is not a hypothetical
worry: every in-situ figure in the Z-Image port came in below its isolated one,
twice by enough to reverse a decision.

The technique that worked is stubbing operations inside a real run rather than
benchmarking them alone — replace an op with a copy, measure the whole forward,
attribute by difference. It took about twenty minutes for Z-Image and found
that attention was 33% of a forward where an isolated benchmark had implied it
was free.

Two measurement rules earned the hard way, both of which apply here:

- **Interleave and swap the order.** This machine drifts about 20% over
  minutes, which is the same size as most of these effects. A comparison run as
  sequential blocks measures the drift. One conclusion in the Z-Image work was
  recorded backwards for exactly this reason and had to be reversed.
- **`gpu_seconds` undercounts whenever MPSGraph is in the graph**, because it
  times only the root command buffer and MPSGraph schedules children.
  `command_wait_seconds` is the honest one.

## And LTX-2.5, which is not converted yet

The same harness at LTX's shapes, read from the released checkpoint rather
than the paper. M1 Pro, 4096 video tokens and 256 audio ones:

| shape | K → N | 8x8, as shipped | 64x40, input-major | | TFLOP/s |
|-------|-------|-----------------|--------------------|---|---------|
| v qkv | 4096 → 4096 | 274.3 ms | 39.6 ms | 6.93x | 3.47 |
| v out | 4096 → 4096 | 270.9 ms | 39.6 ms | 6.84x | 3.47 |
| v ff1 | 4096 → 16384 | 1015.3 ms | 152.7 ms | 6.65x | 3.60 |
| v ff2 | 16384 → 4096 | 1419.4 ms | 161.9 ms | 8.77x | 3.40 |
| a2v q | 4096 → 2048 | 133.5 ms | 20.9 ms | 6.38x | 3.29 |
| a2v out | 2048 → 4096 | 108.5 ms | 20.1 ms | 5.39x | 3.41 |
| a qkv | 2048 → 2048 | 3.3 ms | 1.4 ms | 2.39x | 1.57 |
| a ff1 | 2048 → 8192 | 13.3 ms | 3.1 ms | 4.26x | 2.76 |
| a ff2 | 8192 → 2048 | 16.9 ms | 5.0 ms | 3.38x | 1.72 |

The video stream reaches 3.4-3.6 TFLOP/s, which is where the tile runs on the
shapes it was actually swept against. That is the finding worth more than the
ratios: LTX's widths are close enough to H3's that the geometry is at its
tuned efficiency here, not merely better than what it replaced. Weighted by
how often each shape runs, a block's projections go 4545 ms to 635 ms, 7.2x.

Trust the second run. The first measured a 45% spread between `a2v q` and
`v2a k`, which are the same shape; interleaving within a shape does not cover
shader compilation and first-touch cost across the sweep.

The audio stream is a different regime and should not be quoted at the video
stream's ratio. At 256 rows the tile is four deep, so occupancy decides rather
than the weight re-read the sweep was reasoning about, and throughput roughly
halves. It still wins everywhere, and it is a small share of a block, so this
is a reason not to generalise the number rather than a reason to re-sweep.

**None of the migration hazards above apply.** LTX has no converter yet, so
this is a default to set rather than a layout to change, and two of the three
traps are absent outright: the feed-forward is plain GELU at 4x — `ff.net.0.proj`
is [16384, 4096] and `net.2` consumes all 16384 — so there is no w1/w3 fusion
to survive the transpose. What does need deciding at converter-writing time is
whether to fuse q, k and v into one N=12288 GEMM; they share an input, so it is
worth doing, and it is a fusion along output channels, which is exactly the
class that becomes a within-row operation after transposing.

The third trap has already touched work in the tree: `test_ltx_int8_format.c`
loads nine real LTX projections through `h3_weight_load_i8_linear`, which
infers the scale count from the matrix's first dimension. Input-major storage
changes that test's loader path along with everything else's.

Two things this measurement does not settle. **The attention share is probably
larger for LTX than for H3** — eight attentions a block over a video sequence
in the thousands — so the f32 SDPA switch above may be worth more here than
the tile is, and the stubbing technique should decide that before anyone
assumes the GEMM is the target. And **the absolute number deserves attention on
its own**: 635 ms a block across 48 blocks is about 31 s per denoise step on
this machine at 4096 video tokens, GEMM alone, with attention and the
elementwise work on top. That is with the tuned kernel. Untuned it is 3.7
minutes a step, so for LTX this is not an optimization but the difference
between a slow feature and an unusable one — and even after it, step count is
what decides whether the lane ships.

## Gate

Whatever moves, gate it the way the Z-Image port was gated: against a reference
capture at f32, before and after, with the number unchanged. A retiling is the
same arithmetic in a different order and should not move the output at all —
Z-Image's stayed at 3.25e-02 through five successive kernel changes, which is
what made each one safe to keep.
