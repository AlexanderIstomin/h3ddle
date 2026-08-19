# Z-Image-Turbo

Text to image, 6.15B parameters, eight forwards at guidance 0. The engine
reads the package `Scripts/convert-z-image-package.py` builds.

## What runs where

The Qwen3-4B prompt encoder, thirty-four DiT blocks and VAE run on the GPU.
The prompt encoder streams one layer of BF16 weights at a time so its complete
checkpoint does not have to remain device-resident beside the image model. The
CPU `qwen_block_forward` path remains as a shaderless reference harness only;
production does not silently fall back to it when Metal fails.
The same boundary covers the DiT and VAE: `zimage_generate` requires Metal,
while the full host implementation is reachable only through the explicitly
named `zimage_generate_cpu_reference` harness entry point.

The small DiT embedders, head, sampler and tokenizer stay on the CPU: together
they are well under one percent of the arithmetic. The DiT sequence crosses
the bus once per step in each direction — 63 MB at 4128 tokens, a couple of
milliseconds.

| file | what |
|------|------|
| `zimage_block.c`   | one S3-DiT block, CPU, f32 — the reference the GPU is gated against |
| `zimage_dit.c`     | embedders, head, unpatchify; runs blocks on CPU or device |
| `zimage_gpu.c`     | the thirty-four blocks on Metal |
| `zimage_encoder.c` | CPU reference for Qwen3-4B, 35 of 36 layers (see below) |
| `zimage_encoder_gpu.c` | production Qwen3-4B prompt encoder on Metal |
| `zimage_tae.c` | TAEF1 live denoising-preview decoder on Metal |
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
i + head_dim/2. The GPU's Z-Image QKV boundary reads adjacent pairs directly
but writes them in the engine's half-paired order, because attention is
invariant to permuting the head dimension of q and k together. The checkpoint
weights therefore load unchanged.

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
| text encoder, Metal vs f32 CPU, 35 layers | 6.51e-03 |
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

The prompt encoder has a CPU-versus-Metal correctness and timing harness:

    clang -O2 -fblocks -fobjc-arc -o zimage_encoder_gpu_compare \
        harness/zimage_encoder_gpu_compare.c zimage_encoder.c \
        zimage_encoder_gpu.c ../Qwen3TTS/qwen_block.c \
        ../Qwen3TTS/qwen_weights.c ../../Vendor/h3.c/h3_gpu.m \
        ../../Vendor/h3.c/h3_safetensors.c \
        -I../Qwen3TTS -I../../Vendor/h3.c -I. \
        -framework Foundation -framework Metal \
        -framework MetalPerformanceShaders \
        -framework MetalPerformanceShadersGraph -framework Accelerate -lm

The preview decoder has a model-backed dispatch and output-range smoke test:

    clang -O2 -fobjc-arc -o zimage_tae_check \
        harness/zimage_tae_check.c zimage_tae.c \
        ../../Vendor/h3.c/h3_gpu.m ../../Vendor/h3.c/h3_weights.c \
        ../../Vendor/h3.c/h3_safetensors.c \
        -I../../Vendor/h3.c -I. \
        -framework Foundation -framework Metal \
        -framework MetalPerformanceShaders \
        -framework MetalPerformanceShadersGraph -framework Accelerate -lm

    ./zimage_tae_check /path/to/taef1.safetensors \
        ../../Vendor/h3.c/h3_shaders.metal

## Deliberate CPU work

The sampler and small DiT embedders and head remain CPU code inside
`zimage_dit.c`. The prompt encoder, DiT blocks and VAE use Metal in the app.
