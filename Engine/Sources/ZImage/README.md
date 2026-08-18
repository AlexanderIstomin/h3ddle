# Z-Image-Turbo

Text to image, 6.15B parameters, eight forwards at guidance 0. The engine
reads the package `Scripts/convert-z-image-package.py` builds.

## What runs where

The thirty-four DiT blocks and the VAE decoder run on the GPU. The embedders,
the head, the sampler and the tokenizer stay on the CPU: together they are well
under one percent of the arithmetic, and moving them would double the new code
for no measurable return. The sequence crosses the bus once per step in each
direction — 63 MB at 4128 tokens, a couple of milliseconds.

The text encoder is Qwen3-4B and runs on `qwen_block_forward` from
`../Qwen3TTS` unchanged. That block is shape-general across its third model
now: H3's encoder at 5120 wide, the TTS talker at 1024, this at 2560.

| file | what |
|------|------|
| `zimage_block.c`   | one S3-DiT block, CPU, f32 — the reference the GPU is gated against |
| `zimage_dit.c`     | embedders, head, unpatchify; runs blocks on CPU or device |
| `zimage_gpu.c`     | the thirty-four blocks on Metal |
| `zimage_encoder.c` | Qwen3-4B, 35 of 36 layers (see below) |
| `zimage_vae.c` | the autoencoder both ways: decode for every render, encode for img2img |
| `zimage_vae.c`     | the decoder, CPU, f32 — the reference |
| `zimage_vae_gpu.c` | the decoder on Metal |

## Things that are true and not obvious

**The DiT reads `hidden_states[-2]`.** The second-to-last encoder block,
un-normed, so 35 of 36 layers run and `model.norm` never does. Verified against
transformers' indexing rather than read off the pipeline source.

**The refiners do not run on the unified sequence.** `noise_refiner` sees the
image tokens alone and `context_refiner` the caption tokens alone, each
attending only within its own half; only then are the two concatenated for the
trunk. Running either on the joint sequence is numerically plausible and wrong.

**`cap_embedder` and `context_refiner` take no conditioning**, so the caption
half is identical at every sampler step and is computed once. The trunk writes
through those rows, so the refined copy is kept aside and restored each step.

**The sequence is [image, caption]**, image first — the opposite of most
joint-attention models.

**The block's adaLN is a bare linear; the head's is SiLU then linear.** Reading
one and assuming the other gives a wrong picture with no error anywhere.

**The rope pairs adjacent channels**, where `qwen_block_forward` pairs i with
i + head_dim/2. On the GPU this is a permutation of the q and k rows at load
rather than a kernel, because attention is invariant to permuting the head
dimension of q and k together.

**The sampler**: sigmas are the pipeline's `linspace(1, 1/N, N)` then a *static*
shift of 3.0 — the `mu` the reference computes is dead code under
`use_dynamic_shifting: false`. The model is fed `1 - sigma`, and its output is
negated before the Euler step.

## Correctness

Every stage is gated against a reference the goldens in `Scripts/zimage/`
produce, at f32, before the next stage was built:

| stage | RMS relative |
|-------|--------------|
| text encoder, 36 layers | 4.49e-06 |
| VAE decoder, CPU | 1.47e-06 |
| VAE decoder, GPU | 1.80e-06 |
| one S3-DiT layer | 5.6e-07 |
| whole forward, bf16 weights | 1.34e-05 |
| sampler schedule | bit-identical |

The GPU DiT is compared against the *int8* reference rather than the bf16 one,
so quantization error is not folded into the same number as the port's error.

## Harness

`harness/` holds the checks and benchmarks. They are not in the build — each
has its own `main`, and they need model files and goldens that cannot live in
the repository. Build one directly, for example:

    clang -O2 -fobjc-arc -o zimage_gpu_check \
        harness/zimage_gpu_check.c zimage_gpu.c zimage_block.c \
        ../Qwen3TTS/qwen_block.c ../Qwen3TTS/qwen_weights.c \
        ../../Vendor/h3.c/h3_gpu.m ../../Vendor/h3.c/h3_weights.c \
        ../../Vendor/h3.c/h3_safetensors.c \
        -I../Qwen3TTS -I../../Vendor/h3.c -I. \
        -framework Foundation -framework Metal \
        -framework MetalPerformanceShaders \
        -framework MetalPerformanceShadersGraph -framework Accelerate -lm

## Not done

Nothing is wired to the engine protocol or the app yet — this is the port, not
the feature. The sampler and the head are CPU code inside `zimage_dit.c` rather
than a service entry point.
