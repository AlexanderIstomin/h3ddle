# LTX-2.5 generates end to end

A prompt in, frames out, on the released weights, through the C engine.

```
"a red sports car driving fast along a coastal road at sunset"
  -> tokenizer          (embedded in the text encoder checkpoint)
  -> Gemma 4 tower      48 layers, 49 hidden states      5.5 s
  -> aggregation        video 4096 / audio 2048
  -> connector          8 blocks per stream, 128 span    4.4 s
  -> DiT                48 blocks x 8 sampler steps      265 s
  -> video VAE          9 frames of 256x256              6.5 s
```

Two prompts, same seed and same code, produce a red sports car on a road at
sunset and a ginger cat asleep on a sunlit ledge. That is the check that
matters: it says the conditioning reaches the model, not merely that the
arithmetic is self-consistent.

## What the run found that nothing else could

**The rope reads pixel coordinates, not latent ones.**

`get_pixel_coords` scales each axis by the VAE's compression — 8 in time, 32 in
each spatial axis — before `precompute_freqs_cis` ever sees it. With
`causal_temporal_positioning: True` the time axis is then shifted so the first
latent frame, which a causal encoder builds from a single pixel frame, spans
`[0, 1)` rather than `[0, 8)`:

```c
begin = latent * scale;  end = (latent + 1) * scale;
if (axis == time) { begin += 1 - scale; end += 1 - scale; clamp both at 0; }
```

The audio grid is further off than that: its bounds are **timestamps in
seconds**, not indices at all. `_get_audio_latent_time_in_sec` multiplies the
latent frame by the 4× downsample, applies the same causal offset, and scales
by hop over sample rate.

### Why five layers of verified tests all missed it

Every DiT anchor — the block, the top level, the end-to-end pass, the sampler
loop — built a positions grid and handed **the same grid to both the engine and
the reference**. Latent coordinates agreed with themselves, exactly, at every
tolerance. The tests pinned the rope's *arithmetic*, which was correct; nothing
pinned which coordinates the pipeline is supposed to supply, because nothing
upstream of the tests existed to supply them.

The cost of getting it wrong is not subtle once you see it. Against a
`positional_embedding_max_pos` of 2048, an 8-wide latent frame in latent
coordinates occupies `8/2048` of the rope's range: every column has
substantially the same positional encoding and the model cannot tell one from
the next.

### And it produced a plausible image anyway

This is the part worth remembering. The first generation was not noise. It was
smooth, coloured, temporally consistent, and evolved sensibly across the
denoising steps — it looked like a working model at a resolution too small to
do anything useful. What gave it away was **32-pixel blocking, exactly one
latent cell wide**, which is only a suspicious detail if you know the VAE's
spatial factor is 32.

A component test cannot produce that signal. An image can.

## What is verified, and what this does not prove

Every stage has its own runner holding it against LTX's own code, and those
still stand. What this adds is the seams between them, and one number:
generation works.

It does not prove the conventions still unpinned at the pipeline level:

- **audio duration alignment.** Both streams denoise, but the audio token count
  here is chosen, not derived from the video duration. The audio latent decodes
  to a mel; whether that mel is the right *length* for these 9 frames is
  unverified;
- **the resolution ladder.** 256×256 at 9 frames is far below what LTX-2.5
  targets. It works, and the sampler's shift is token-dependent so the schedule
  adapts, but nothing here says a 768×512 clip is right;
- **the negative/CFG path.** The distilled model is run without guidance, which
  is what distillation is for, but the guiders in `ltx_core.components` are
  unported.

## The cost, measured

At 256×256 and 9 frames — 128 video tokens — 8 steps take 265 s, of which
**54 to 76 s is reloading the 48-block stack, once per step**. That fraction is
fixed per step and independent of resolution, so it shrinks in relative terms
as the clip grows and grows in absolute terms with the step count.

The tower and connector together are 10 s and run once. The VAE decode is 6.5 s.
Nothing outside the DiT is close to the critical path.

## Reproducing it

```
python tokenize_prompt.py "..." prompt.safetensors     # ids from the embedded tokenizer
h3_real_ltx_text_test      ENCODER prompt.safetensors states.bin
h3_real_ltx_connector_test DIT     states.bin          context.bin
h3_ltx_generate            DIT     context.bin         latent.bin [STEPS] [SEED]
h3_ltx_decode              VAE     latent.bin          frames.bin
python frames_to_png.py    frames.bin sheet.png
```

The text runner now accepts an anchor carrying only `input_ids` and skips the
phases that compare against a fixture — there is no golden tower output for an
arbitrary caption, and producing one would need the 12B model at F32.

The two-phase split between the tower and the connector is not scaffolding: the
connector's weights live in the DiT file while the tower's live in the text
encoder, and holding both open at once costs 37 GB.

## The per-step reload, and what H3 already does about it

The 54–76 s of per-step weight loading is not a new problem and should not get
a new solution. `h3_dit.c` already answers it twice for H3, and the LTX driver
built here does neither.

**Residency** — load all 48 blocks once and keep them. `h3_video_vae.c` shows
the shape the engine settled on: measure what residency would cost from the
checkpoint, compare against `hw.memsize` with a headroom allowance, take it
only if there is room left over, and let an env var force either way.

**That one does not apply to LTX on this machine.** The DiT is 21.5 GB at int8
against 32 GiB of RAM. Under the same 8 GiB headroom rule it fits only on
paper; in practice the activations, the OS and the rest of the engine would put
it into paging, which costs more than the reread it saves.

**SSD streaming with double-buffered prefetch** — which is exactly why H3 has
it. Two preallocated slots, and before block *N* runs, a `pthread` fills slot
`N+1`:

```c
stream_job = (h3_dit_stream_job){ .dit = dit, .layer = future,
                                  .slot = dit->stream_ready_slot ^ 1u };
pthread_create(&stream_thread, NULL, read_stream_layer_thread, &stream_job);
run_block(...);                     /* GPU works while the CPU reads      */
h3_gpu_submit(dit->gpu);
pthread_join(stream_thread, NULL);  /* and the wait is what gets measured */
```

Three details worth copying rather than rediscovering:

- **the slots are preallocated**, so the worker only fills tensors and never
  allocates. On unified memory that fill is a memcpy into a buffer the GPU can
  already see;
- **it wraps around**: `future = first_active_block(dit)` at the end of the
  stack, so the *next step's* first block loads during the current step's last.
  The per-step boundary is not a stall;
- **the sources are sorted by path then file offset**, so a block's reads walk
  the checkpoint forwards instead of seeking.

It also accounts for itself — `stream_wait_seconds` separates time actually
lost from time successfully hidden, which is the number that says whether
prefetch is working.

### What that is worth here

At 128 video tokens the driver spends 198 s in blocks and 54 s loading, so
perfect overlap saves at most 20–28%. At 768 tokens a step is 160 s while the
load stays fixed, so the same absolute saving is a smaller share. **Prefetch
matters most at small resolutions and short clips**, which is the opposite of
where the total time is worst — worth having, but not the thing that decides
whether the lane ships.

The LTX blocks are all identically shaped, so the two-slot scheme applies
directly with no per-block size negotiation.

## Cost at two sizes, measured

| | tokens | first steps | steady state | 8 steps |
|---|---|---|---|---|
| 256×256, 9 frames | 128 | ~33 s | ~31 s | 265 s |
| 512×512, 17 frames | 768 | 158 s | **52 s** | 827 s |

**The first steps of a run cost three times the steady state**, and it is not
the file reads — `load_seconds` is only 73 s of the 827. It is warmup: MPS
graph construction and kernel caches on first use at each shape. A benchmark
that times step one and extrapolates is wrong by 3x.

At steady state, six times the tokens costs 1.7x the time. That is strongly
sublinear, which says fixed per-step overhead dominates at the small size and
the 256x256 figure is nearly all overhead. It also means prefetching weights is
worth less than the 20-28% estimated above once warmup is separated out.

The sampler's shift is token-dependent, so the two runs get genuinely different
schedules; that is the scheduler working, not a discrepancy.

## The int8 tile, which is where the speed actually was

The driver was calling `h3_gpu_linear_i8_weight_bf16` — the plain 8x8 simd
path — for all six projections in a block, because that is what the component
runners it was built from use. `h3_gpu_linear_i8_weight_bf16_square_output_major`
takes the same arguments on the same on-disk layout and needs no
re-quantization:

| 512×512, 8 steps | blocks | loading | total |
|---|---|---|---|
| plain int8 | 732.7 s | 73.0 s | **827 s** |
| square tile | 147.8 s | 74.3 s | **242 s** |

**5x in the blocks, 3.4x end to end, and the latents are bit identical.**

### This re-orders everything after it

The obvious next move looked like an **input-major converter**: the input-major
tile is faster still. Its own note prices that at **9%**, and it costs a
conversion pipeline over a 21.5 GB checkpoint plus a second copy on disk. Nine
percent of 242 s is 13 s.

Meanwhile weight loading was 73 s against 827 — 9% — and is now 74 s against
242, or **31%**. The prefetch `h3_dit.c` already implements went from marginal
to the largest remaining lever, and it needs no new file format.

The ordering is: **use the tuned kernel that exists (free, 3.4x), then hide the
loading with the prefetch that exists, and only then consider a converter.**

Both of the first two are now done. Prefetch, measured A/B at 512×512 over 4
steps with the modes alternated so they see the same machine:

| | blocks | loading | total |
|---|---|---|---|
| serial | 71.7 s | 43.3 s | 126.1 s |
| **prefetch** | 45.8 s | 0.4 s | **56.5 s** |
| serial | 45.0 s | 28.1 s | 82.4 s |
| **prefetch** | 44.9 s | 0.5 s | **55.7 s** |

Block time is unchanged between modes, so the worker costs the GPU nothing.
Loading essentially disappears, and prefetching is also the *stable*
configuration — the serial runs swing with whatever else the machine is doing
while the prefetched ones do not, because the slack absorbs it.

### A measurement I got wrong, and how

The first attempt at this said the opposite and said it confidently: 242 s
serial against 320 s prefetched, with a tidy mechanism — unified memory, a
bandwidth-bound GEMM, a worker stealing the bus. It was written up as a
negative result.

Another generation was running on the machine at the time.

**One timing run on a shared machine is not a measurement**, and a mechanism
invented to explain one is worse than no explanation, because it is memorable
and it forecloses the question. The A/B above alternates for exactly that
reason, and the tell was available without knowing the cause: block time
differed between the two runs when the thing being changed does not touch the
GPU at all.

### What that leaves for a converter

Now that loading is ~0.5 s, the input-major converter's 9% is the only thing
left to buy — and the file layout finding below means a converter's *real*
value would have been the other thing it could fix, which prefetch has now
made moot.

### The file layout, which is worse than the matrix layout

A block's 388 MB is not contiguous. The checkpoint is sorted by tensor name and
grouped by size class, so a block's hundred-odd tensors interleave with every
other block's and span the whole 21.5 GB at **1.8% density**. Reading a block is
a hundred scattered seeks, not one sequential read.

That is a better argument for a converter than the input-major one — rewriting
the file block-major would make each block a single 388 MB sequential read. But
prefetch already hides the cost entirely, so it buys nothing on top.

The component runners still call the plain path. That is deliberate for now —
their recorded tolerances and timings were measured against it, and the output
is bit identical either way, so there is nothing numerical to re-establish when
they are switched.

## The 512 result, and an artifact that was not one

At 512×512 with 17 frames the output is properly good: a red sports car on a
coastal road, sun low over the ocean, guardrail, rocks, road markings. It is
recognisably the prompt rather than merely suggestive of it, which the 256×256
run was.

The first 8-step clip had what looked like a serious defect: frame 0 clean,
frames 1 to 12 ghosted with several cars overlaid, frames 13 to 16 clean again.
The boundary was suspiciously exact — three latent frames decode as 1 + 8 + 8
pixel frames, and the corrupted span was almost exactly the middle latent
frame's eight. Three structural causes were written down to test: the temporal
rope, the keyframe marker, and the unpinned audio duration alignment.

**It was none of them. The model was drawing motion blur.**

Sixteen steps rather than eight cleaned up the back half and left frames 1–9
smeared. Then the same scene with `driving fast` changed to `parked still` —
same seed, same resolution, same 16 steps, same code — came out flawless across
all seventeen frames, with a gentle camera drift and no ghosting anywhere. The
per-frame contrast tells the same story without looking at the images: 72→86
with a dip for the moving car, a flat 68.8–70.0 for the parked one.

So the pipeline was right and the prompt was being obeyed. Worth recording for
two reasons:

- **an exact structural boundary is not evidence of a structural bug.** The
  ghosting aligned with the latent frame boundary because motion blur has to
  align with it too — that is the only temporal resolution the latent has;
- **the cheap discriminator was a second prompt, not a code change.** Two and a
  half minutes, and it ruled out three hypotheses at once. All three would have
  meant reading reference code and instrumenting the driver.

Step count is a real quality lever independently of this: 16 steps is visibly
better than 8 on the moving clip.

## Cost at two sizes, measured

| | tokens | first steps | steady state | 8 steps |
|---|---|---|---|---|
| 256×256, 9 frames | 128 | ~33 s | ~31 s | 265 s |
| 512×512, 17 frames | 768 | 158 s | **52 s** | 827 s |

**The first steps of a run cost three times the steady state**, and it is not
the file reads — `load_seconds` is only 73 s of the 827. It is warmup: MPS
graph construction and kernel caches on first use at each shape. A benchmark
that times step one and extrapolates is wrong by 3x.

At steady state, six times the tokens costs 1.7x the time. That is strongly
sublinear, which says fixed per-step overhead dominates at the small size and
the 256x256 figure is nearly all overhead. It also means prefetching weights is
worth less than the 20-28% estimated above once warmup is separated out.

The sampler's shift is token-dependent, so the two runs get genuinely different
schedules; that is the scheduler working, not a discrepancy.

## The int8 tile, which is where the speed actually was

The driver was calling `h3_gpu_linear_i8_weight_bf16` — the plain 8x8 simd
path — for all six projections in a block, because that is what the component
runners it was built from use. `h3_gpu_linear_i8_weight_bf16_square_output_major`
takes the same arguments on the same on-disk layout and needs no
re-quantization:

| 512×512, 8 steps | blocks | loading | total |
|---|---|---|---|
| plain int8 | 732.7 s | 73.0 s | **827 s** |
| square tile | 147.8 s | 74.3 s | **242 s** |

**5x in the blocks, 3.4x end to end, and the latents are bit identical.**

### This re-orders everything after it

The obvious next move looked like an **input-major converter**: the input-major
tile is faster still. Its own note prices that at **9%**, and it costs a
conversion pipeline over a 21.5 GB checkpoint plus a second copy on disk. Nine
percent of 242 s is 13 s.

Meanwhile weight loading was 73 s against 827 — 9% — and is now 74 s against
242, or **31%**. The prefetch `h3_dit.c` already implements went from marginal
to the largest remaining lever, and it needs no new file format.

The ordering is: **use the tuned kernel that exists (free, 3.4x), then hide the
loading with the prefetch that exists, and only then consider a converter.**

Both of the first two are now done. Prefetch, measured A/B at 512×512 over 4
steps with the modes alternated so they see the same machine:

| | blocks | loading | total |
|---|---|---|---|
| serial | 71.7 s | 43.3 s | 126.1 s |
| **prefetch** | 45.8 s | 0.4 s | **56.5 s** |
| serial | 45.0 s | 28.1 s | 82.4 s |
| **prefetch** | 44.9 s | 0.5 s | **55.7 s** |

Block time is unchanged between modes, so the worker costs the GPU nothing.
Loading essentially disappears, and prefetching is also the *stable*
configuration — the serial runs swing with whatever else the machine is doing
while the prefetched ones do not, because the slack absorbs it.

### A measurement I got wrong, and how

The first attempt at this said the opposite and said it confidently: 242 s
serial against 320 s prefetched, with a tidy mechanism — unified memory, a
bandwidth-bound GEMM, a worker stealing the bus. It was written up as a
negative result.

Another generation was running on the machine at the time.

**One timing run on a shared machine is not a measurement**, and a mechanism
invented to explain one is worse than no explanation, because it is memorable
and it forecloses the question. The A/B above alternates for exactly that
reason, and the tell was available without knowing the cause: block time
differed between the two runs when the thing being changed does not touch the
GPU at all.

### What that leaves for a converter

Now that loading is ~0.5 s, the input-major converter's 9% is the only thing
left to buy — and the file layout finding below means a converter's *real*
value would have been the other thing it could fix, which prefetch has now
made moot.

### The file layout, which is worse than the matrix layout

A block's 388 MB is not contiguous. The checkpoint is sorted by tensor name and
grouped by size class, so a block's hundred-odd tensors interleave with every
other block's and span the whole 21.5 GB at **1.8% density**. Reading a block is
a hundred scattered seeks, not one sequential read.

That is a better argument for a converter than the input-major one — rewriting
the file block-major would make each block a single 388 MB sequential read. But
prefetch already hides the cost entirely, so it buys nothing on top.

The component runners still call the plain path. That is deliberate for now —
their recorded tolerances and timings were measured against it, and the output
is bit identical either way, so there is nothing numerical to re-establish when
they are switched.

## The 512 result, and a temporal artifact worth chasing

At 512×512 with 17 frames the output is properly good: a red sports car on a
coastal road, sun low over the ocean, guardrail, rocks, a road surface with
markings. It is recognisably the prompt rather than merely suggestive of it,
which the 256×256 run was.

But the clip has a clear defect. **Frame 0 is clean, frames 1 to about 12 are
ghosted — several cars overlaid — and frames 13 to 16 are clean again.**

The temporal structure is the first place to look, because it is not uniform.
Three latent frames decode to 17 pixel frames as 1 + 8 + 8: a causal encoder
makes the first latent frame cover a single pixel frame and the rest cover
eight. So the rope midpoints are 0.5, 5 and 13 — deliberately unevenly spaced
— and the corrupted span is roughly the second latent frame's eight.

Candidates, cheapest first:

- **step count.** Eight steps on a distilled model is meant to be enough, but
  ghosting is what too few steps looks like. A 16-step run settles it and costs
  half an hour;
- **audio duration alignment.** The audio token count is still chosen rather
  than derived, so the cross-modal attention may be relating video frames to
  audio at the wrong rate — and the cross-modal path is exactly what would
  smear content along time;
- **the keyframe marker.** It is applied to the first latent frame's tokens,
  which is what makes frame 0 special — and frame 0 is the one that came out
  clean.

Recording it rather than guessing further: the pipeline is proven, and this is
a quality question with three testable causes.


## The audio seam, and the third rope bug (2026-08-18)

`tests/ltx_audio_decode.c` finally reads the audio latent that
`h3_ltx_generate` had been writing since the day it was written:

```
latent [R, 128] -> unpatchify [8, R, 16] -> denormalize
                -> audio VAE decoder -> log-mel [2, 4R-3, 64]
                -> vocoder -> (4R-3) * 160 samples of 16 kHz stereo
```

It made a sound immediately, and the sound was wrong. **A listener said so;
nothing else could.** Every number looked healthy — loud, tonal, and clearly
prompt-dependent, with a sports car 22x louder than a sleeping cat.

### The time axis is in seconds

`create_initial_state` divides axis 0 by the fps right after
`get_pixel_coords`. This engine was feeding pixel frames.

That is the **third** time this one axis has been wrong: latent cells, then
pixel frames, then seconds, each a separate bug in the same four lines. It
survived the second fix because the picture barely shows it — per-frame content
is not what a time scale controls — and it did its damage exactly where it was
heard. The audio grid was already seconds (0 to 0.7) while the video grid was
frames (0 to 17), so video filled 85% of the rope's range and audio 3.5% of it,
and **the two streams attend to each other through those positions**.

The "audio duration alignment" candidate listed above was therefore right in
substance and too shallow in detail: the rate relating the streams was wrong,
but because their *units* differed, not because a count was mischosen. Worth
noting that the ghosting it was proposed to explain was later written off as
motion blur — a conclusion that should now be re-tested rather than trusted.

### The audio length was never a free parameter

Two lines away in the same file:

```python
AudioLatentShape.from_video_pixel_shape:
    duration = float(shape.frames) / float(shape.fps)
    frames   = round(duration * sample_rate / hop_length / downsample)   # 25/s
```

At 3 latent frames and 24 fps that is 18 rows, against the 24 hardcoded here:

```
before   video 0.708 s   audio 0.930 s    31% too long
after    video 0.708 s   audio 0.690 s    within one audio frame
```

`fps` is not in the checkpoint — `VideoPixelShape.fps` is supplied by the
caller — so it is now a compile-time knob defaulting to 24, and both the rope
scale and the row count follow from it rather than being chosen separately.

### What this says about how to work on the rest

Two days went into component anchors, each one green, before any modality was
rendered end to end. The bug was found within an hour of the seam existing.
A latent that has never been decoded has never been checked, and an artefact
no person has looked at or listened to has not been verified — the numbers
here were not merely insufficient, they were actively reassuring.
