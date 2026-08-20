# Compact Ref2VA hybrid loader

## Goal

Avoid storing and downloading a second 20.97 GB Turbo transformer solely to
enable ordered Ref2VA inputs. This is an experimental quality/storage trade,
not a way to accelerate a transformer pass.

The public
[ComfyUI_MinimaxH3HybridLoader](https://github.com/scottmudge/ComfyUI_MinimaxH3HybridLoader)
reports that FL2VA and Ref2VA share almost all of their structure and offers a
"subjective" recipe that starts with FL2VA and replaces Ref2VA AdaLN weights in
blocks 25–49. H3ddle implements that recipe independently in its native loader.

## Checkpoint format

`Scripts/build-h3-hybrid-adaln.py` copies only these 50 F16 tensors from the
compact Ref2VA checkpoint:

- `blocks.25...49.adaln_proj.linear.weight`
- `blocks.25...49.adaln_proj.linear.bias`

It adds three singleton U32 markers: format version, first block, and block
count. The runtime validates all markers plus every dtype and shape before it
creates a Metal device. The source checkpoint is never modified and incomplete
or differently shaped overlays fail instead of silently falling back.

The FL2VA base supplies blocks 0–24, the compact timestep curve, text refiner,
patch projections, attention/MLP core, final AdaLN, and output heads. This is
deliberate: using Ref2VA's distinct timestep table or final head would no
longer be the tested recipe.

For the Turbo INT8 ConvRot checkpoints measured on 2026-08-20:

| reference component | bytes | reduction |
|---|---:|---:|
| full input-major Ref2VA | 20,970,380,012 | — |
| blocks 25–49 AdaLN overlay | 43,551,180 | 99.79% |

The overlay built from the standard Ref2VA checkpoint and the overlay built
from the Turbo Ref2VA checkpoint are byte-identical: both have SHA-256
`c3d80a9a2d17a30caf83e933262473cbf0b1ba7de4d29556646e9a92ab5f17aa`.
One hosted overlay can therefore serve both the standard and Turbo FL2VA base
checkpoints without duplicating storage.

The installed filename is
`diffusion_models/minimax_h3_ref2va_pruned_int8_convrot_hybrid_adaln_25_49.safetensors`.
When both files exist, `H3_REF2VA_HYBRID=1` selects the overlay for controlled
A/B tests. When the full Ref2VA file is absent, the compact overlay is selected
automatically. Prompt-only generations always keep the FL2VA path.

## First native A/B

Machine: Apple M1 Pro, 32 GB. Both runs used the same Turbo + References model,
input-major weights, reference portrait, prompt, 512×512 canvas, 22-frame still
detail, eight beta-schedule passes, 50 blocks, core reuse 1, and seed 65859680.

| measurement | full Ref2VA | hybrid | difference |
|---|---:|---:|---:|
| DiT load | 1.135 s | 0.877 s | -0.258 s |
| Euler denoise | 236.641 s | 236.650 s | +0.009 s (+0.004%) |
| total DiT | 237.806 s | 237.555 s | -0.251 s |
| SSD bytes read | 144.005 GiB | 144.005 GiB | none |
| VAE decode | 47.198 s | 47.227 s | +0.029 s |

The runtime therefore behaves as expected: the overlay changes precomputed
conditioning but the 50-block core has the same size and cost. Both decoded
images were coherent, detailed portraits that preserved the reference's
identity, red shirt, and lighting. Their SSIM was 0.728 and PSNR 19.42 dB,
showing that the recipe intentionally changes pose and expression rather than
reproducing the full Ref2VA pixels.

One image is not enough evidence to replace the managed model. Before shipping
the compact package, test several seeds and reference types (single/multiple
images, video, and audio), compare identity and prompt adherence against full
Ref2VA, and make the storage trade explicit in the model description.

The automatic fallback was also exercised with a lean package containing the
FL2VA core and overlay but no full Ref2VA checkpoint. A real 256×256, two-pass
reference generation selected the hybrid without an environment override and
completed normally. A standard, non-Turbo checkpoint completed the same smoke
test with the explicit override. These checks establish loader compatibility;
they do not replace the broader visual-quality matrix above.
