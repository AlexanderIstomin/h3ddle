#!/usr/bin/env python3
"""Export what Z-Image's VAE decoder produces, for the C port to match.

Unlike the text encoder this should agree at f32, not within a bf16 argument:
the decoder is a convolution stack the engine runs in f32 end to end, exactly
as the speech codec's vocoder did.

Intermediates are exported alongside the image because the stack is deep — a
conv_in, a mid block with attention, four up blocks, and a head — and a single
comparison at the end cannot say which of them went wrong.

  zimage_vae_golden.py --model <base checkout> --out golden.safetensors [--size 16]
"""

import argparse

import torch
from diffusers import AutoencoderKL
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=16,
                        help="latent side; the image comes out 8x this")
    args = parser.parse_args()

    # f32 is the default here; passing dtype= is a modelling-library
    # keyword the config path forwards into the constructor.
    vae = AutoencoderKL.from_pretrained(f"{args.model}/vae")
    vae.eval()
    print(f"latent channels {vae.config.latent_channels}, "
          f"blocks {vae.config.block_out_channels}, "
          f"scaling {vae.config.scaling_factor}, shift {vae.config.shift_factor}")

    # A fixed spread rather than a real encode: the decoder is being checked,
    # not the pair, and a seeded normal exercises every channel evenly.
    torch.manual_seed(7)
    latent = torch.randn(1, vae.config.latent_channels, args.size, args.size)

    taps = {}

    def tap(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            taps[name] = value.detach().float().clone()
        return hook

    decoder = vae.decoder
    handles = [
        decoder.conv_in.register_forward_hook(tap("conv_in")),
        decoder.mid_block.register_forward_hook(tap("mid_block")),
        decoder.up_blocks[0].register_forward_hook(tap("up_block_0")),
        decoder.up_blocks[2].register_forward_hook(tap("up_block_2")),
        decoder.up_blocks[3].register_forward_hook(tap("up_block_3")),
    ]
    with torch.no_grad():
        image = vae.decode(latent).sample
    for handle in handles:
        handle.remove()

    tensors = {"latent": latent, "image": image.float(), **taps}
    for name, value in tensors.items():
        print(f"  {name:12} {tuple(value.shape)}")
    print(f"\nimage range [{image.min():.3f}, {image.max():.3f}]")

    save_file({name: value.contiguous().clone() for name, value in tensors.items()},
              args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
