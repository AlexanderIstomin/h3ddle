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

| | tokens | per step | 8 steps |
|---|---|---|---|
| 256×256, 9 frames | 128 | ~33 s | 265 s |
| 512×512, 17 frames | 768 | ~160 s | ~21 min |

Six times the tokens for about five times the time — close to linear, which
says attention is not yet dominating at this size. The sampler's shift is
token-dependent, so the two runs get genuinely different schedules; that is the
scheduler working, not a discrepancy.
