# YuE2 music

YuE2 runs natively in H3ddle's C/Metal engine helper. App inference needs only
macOS frameworks and model weights: no Python, PyTorch, Transformers, `uv`, or
Hugging Face CLI. An older prototype package's runtime directory is ignored.

In Models, choose Audio and download YuE2-3B · Music (about 7.8 GB).
The model uses the non-commercial CC BY-NC 4.0 license. Existing local YuE2
folders can also be added. Open Audio generation, select
Music and YuE2-3B, and enter a style. Lyrics are optional; leave them blank for an instrumental
request (the helper adds “instrumental, no vocals” to the style). Choose 8, 16 or 32 synthesis
steps; 32 is the default. Steps and lyrics are saved with
queued jobs and generation recipes.

Maximum length is 1–360 seconds, matching the reference’s default 9000 semantic
frames at 25 Hz. The model can finish earlier. Full melody/chord planning remains
enabled. Bounded-memory Metal attention supports the model’s 24576-token context;
acoustic synthesis uses the reference context chunk boundaries, and scratch/KV
buffers grow only as needed. Long lyrics/styles that exceed the combined planning
and semantic budget are rejected with a message to shorten them. The native random-number generator is reproducible,
but its seed does not reproduce the former Python worker's exact song.

All transformer operations and VAE convolutions run on Metal. The helper
releases other model caches before loading YuE2 and releases YuE2 tensors after
each job. No network access or subprocess is needed during music generation.
H3-specific block caching, layer skipping and beta schedules do not apply.

## Developer installation

Existing prototype folders work without reinstalling or converting weights.
For a new folder, install the Hugging Face `hf` CLI in your development
environment and run:

```sh
python3 Scripts/setup-yue2-package.py \
  --destination "$HOME/Library/Application Support/H3ddle/LocalModels/YuE2-3B"
```

This development script downloads pinned model/VAE files, verifies SHA-256
hashes and preserves their licenses. It does not install an inference runtime.
Choose this folder in H3ddle's Audio model library, or use the in-app catalog download.

The [native implementation and validation notes](experiments/2026-09-13-yue2-native.md)
record the numerical comparisons and tested limitations. Python remains only
in development reference/export tools outside the app bundle.

After building the app, run the optional production-provider test with:

```sh
MTL_DEBUG_LAYER=1 \
H3DDLE_YUE2_SMOKE_PACKAGE="$HOME/Library/Application Support/H3ddle/LocalModels/YuE2-3B" \
  swift test --package-path Packages/H3ddleKit --no-parallel --filter yue2RealWorkerSmoke
```

Normal CI uses fake inference, protocol persistence, package recognition and
native request-validation tests without weights. UI tests verify step selection,
Reset, durations above 30 seconds, and queue restoration with both sung and empty
lyrics. The weight-free long-context attention oracle runs with:

```sh
python3 Scripts/build-yue2-check.py --output /tmp/yue2-check
MTL_DEBUG_LAYER=1 /tmp/yue2-check --attention Engine/Vendor/h3.c/h3_shaders.metal
```

With the optional weights, `--capacity PACKAGE SHADERS` checks allocation for 9000 frames and the full
24576-token context, then evaluates one 1100-frame acoustic velocity with real
model weights. This is a numerical capacity check, not a music-quality benchmark.

The weights retain CC BY-NC 4.0. The package preserves upstream licenses and
third-party notices; see the [model repository](https://huggingface.co/m-a-p/YuE2-3B).

## Progress and native attention

YuE2 has separate score-planning, semantic-token, acoustic-synthesis, decoding
and saving stages. Completing the token budget does not mean the audio is ready.
The app reserves 35–97% for acoustic synthesis and holds the bar below 100% until
the output is complete. Remaining time starts with a duration-and-step projection and switches to
measured acoustic work once synthesis has established its pace.

Acoustic attention uses Apple's native MPSGraph SDPA (Metal), with no interpreter
or external runtime. `H3_YUE2_REFERENCE_ATTENTION=1` selects the slower scalar
kernel for numerical comparisons. Each completed acoustic transformer layer
reports progress and checks cancellation; work is not reported merely when it is
enqueued. The development capacity harness also checks cancellation after the
first acoustic layer.

After reusing the AR context across stages, a two-minute, 8-step request completed
in 9m39s on an M1 Pro versus 11m22s in the pinned Python pipeline (15% less time).
A complete fixed-token 30-second workload used 21–29% less computation time at
8/16/32 steps and passed the numerical checks. These are measured workloads on
that Mac, not a universal timing guarantee. Long music remains a substantial
workload. See the native experiment report for stage timings and comparison
scope. To validate a captured JSONL event trace,
set `H3DDLE_YUE2_PROGRESS_TRACE` and run `GenerationProgressTrackerTests`.

Remaining time starts with a duration-and-step projection, then follows measured
acoustic synthesis speed. It is approximate and varies with hardware and score
complexity; decoding and saving are shown as finishing.
