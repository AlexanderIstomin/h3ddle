#!/usr/bin/env python3
"""Convert Stability's MLX Stable Audio 3 Small SFX release into the
safetensors package layout h3.c reads.

Stability publishes two copies of this model. The gated
`stable-audio-3-small-sfx` repo is fp32, 3.5 GB, and requires every user to
accept a licence on Hugging Face before downloading. The ungated
`stable-audio-3-optimized` repo carries the same weights already reduced to
f16 NumPy archives, needs no account, and omits the encoder half that
text-to-audio never runs. We convert from the ungated copy so the app can
fetch weights without asking the user to log in anywhere.

Only the container changes. Tensor names, shapes and values are copied
through untouched, so the engine can be checked against Stability's own MLX
implementation tensor for tensor. The one optional exception is --half, which
casts the f32 decoder to f16; it is off by default because Stability ships
that decoder in f32 while shipping everything else in f16, which reads as a
deliberate choice about audio quality rather than an oversight.

The Stability AI Community License requires anyone redistributing these
weights to pass on the agreement, state what they changed, and credit
Stability in the product. The package therefore carries LICENSE.md,
LICENSE_GEMMA.md and a NOTICE naming this conversion; the "Powered by
Stability AI" credit is the app's responsibility, not this script's.

Usage:
  convert-sa3-package.py --mlx-dir sa3-mlx --tokenizer tokenizer.json \
      --out MiniMax-Stable-Audio-3-SFX [--half]
"""

import argparse
import json
import os
import shutil
import struct
import sys

import numpy as np

# MLX archive -> package file. The encoder is deliberately absent: it only
# matters for audio-to-audio, and leaving it out saves the user 215 MB.
PARTS = [
    ("MLX/dit_sm-sfx_f16.npz", "dit.safetensors"),
    ("MLX/t5gemma_f16.npz", "text_encoder.safetensors"),
    ("MLX/same_s_decoder_f32.npz", "decoder.safetensors"),
]

LICENSES = ["LICENSE.md", "LICENSE_GEMMA.md"]

NOTICE = """This Stability AI Model is licensed under the Stability AI Community License,
Copyright (c) Stability AI Ltd. All Rights Reserved

Gemma is provided under and subject to the Gemma Terms of Use found at
ai.google.dev/gemma/terms

Modifications by H3ddle: the weights were repackaged from Stability's MLX
NumPy archives (stabilityai/stable-audio-3-optimized) into safetensors. Tensor
names, shapes and values are unchanged{half}. No weights were retrained,
merged, pruned or quantized.
"""

DTYPES = {
    np.dtype("float16"): "F16",
    np.dtype("float32"): "F32",
    np.dtype("int32"): "I32",
    np.dtype("int64"): "I64",
}


def write_safetensors(path, tensors, metadata):
    """Write a safetensors file without pulling in torch."""
    header = {"__metadata__": metadata}
    offset = 0
    ordered = sorted(tensors)
    for name in ordered:
        array = tensors[name]
        dtype = DTYPES.get(array.dtype)
        if dtype is None:
            raise SystemExit(f"{name}: unsupported dtype {array.dtype}")
        size = array.nbytes
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size

    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # The spec wants the data section 8-byte aligned; pad the header to suit.
    padding = (-len(blob)) % 8
    blob += b" " * padding

    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for name in ordered:
            handle.write(np.ascontiguousarray(tensors[name]).tobytes())


def convert(source, destination, half, metadata):
    """Returns (tensor count, parameter count, side files to write).

    The archives carry two byte blobs alongside the weights: META, the text
    encoder's architecture as JSON, and TOKENIZER_MODEL, the SentencePiece
    model. Neither is a weight. The first belongs in the safetensors header
    where the engine already looks; the second is a file in its own right.
    """
    extras = {}
    with np.load(source) as archive:
        tensors = {}
        metadata = dict(metadata)
        for name in archive.files:
            array = archive[name]
            if array.dtype == np.uint8:
                if name == "META":
                    metadata["config"] = bytes(array).decode("utf-8")
                elif name == "TOKENIZER_MODEL":
                    extras["tokenizer.model"] = bytes(array)
                else:
                    raise SystemExit(f"{name}: unexpected byte blob")
                continue
            if half and array.dtype == np.float32:
                array = array.astype(np.float16)
            tensors[name] = array
        count = len(tensors)
        params = sum(a.size for a in tensors.values())
        write_safetensors(destination, tensors, metadata)
    return count, params, extras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-dir", required=True,
                        help="checkout of stabilityai/stable-audio-3-optimized")
    parser.add_argument("--tokenizer", required=True,
                        help="t5gemma tokenizer.json (HF fast-tokenizer format)")
    parser.add_argument("--out", required=True, help="package directory to write")
    parser.add_argument("--half", action="store_true",
                        help="also cast the f32 decoder to f16 (halves its size)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    metadata = {
        "source_repo": "stabilityai/stable-audio-3-optimized",
        "model": "stable-audio-3-small-sfx",
        "license": "Stability AI Community License",
        "conversion": "mlx npz to safetensors, values unchanged",
        "decoder_precision": "f16" if args.half else "as-released",
    }

    total_params = 0
    for relative, output in PARTS:
        source = os.path.join(args.mlx_dir, relative)
        if not os.path.exists(source):
            raise SystemExit(f"missing {source}")
        target = os.path.join(args.out, output)
        # The decoder is the only f32 archive, so --half only ever hits it.
        count, params, extras = convert(source, target, args.half, metadata)
        total_params += params
        size = os.path.getsize(target)
        print(f"{output:<24} {count:>4} tensors  {params/1e6:>7.1f}M params  "
              f"{size/1e6:>7.1f} MB")
        for name, blob in extras.items():
            with open(os.path.join(args.out, name), "wb") as handle:
                handle.write(blob)
            print(f"{name:<24} {len(blob)/1e6:>28.1f} MB")

    shutil.copyfile(args.tokenizer, os.path.join(args.out, "tokenizer.json"))
    print(f"{'tokenizer.json':<24} "
          f"{os.path.getsize(args.tokenizer)/1e6:>28.1f} MB")

    for name in LICENSES:
        source = os.path.join(args.mlx_dir, name)
        if os.path.exists(source):
            shutil.copyfile(source, os.path.join(args.out, name))
        else:
            print(f"warning: {name} not found in {args.mlx_dir}", file=sys.stderr)

    note = NOTICE.format(
        half=", except that the decoder was cast from f32 to f16"
        if args.half else "")
    with open(os.path.join(args.out, "NOTICE"), "w") as handle:
        handle.write(note)

    print(f"\n{total_params/1e6:.1f}M parameters total -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
