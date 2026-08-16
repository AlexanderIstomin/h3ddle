#!/usr/bin/env python3
"""Convert a Qwen3-TTS release into the safetensors package layout h3.c reads.

The release is two checkpoints — the backbone and a speech tokenizer — holding
five subsystems between them. Four of those are needed to speak; the fifth is
half the tokenizer and is dropped here, which is most of the reason this script
exists.

What comes out, one file per subsystem so each can be validated on its own
against the goldens and demand-paged independently:

  talker.safetensors          28 layers, plus the 151936 x 2048 text embedding
  code_predictor.safetensors  the 5-layer model that emits code groups 1..15
  speaker_encoder.safetensors ECAPA-TDNN over a 128-band mel
  codec_decoder.safetensors   RVQ, an 8-layer transformer, and the vocoder

What is deliberately left behind: `encoder.*` in the speech tokenizer, which is
a Mimi codec encoder. It exists to turn a reference clip into codes for
in-context cloning. We clone from the ECAPA speaker embedding instead — the two
were compared by ear and the embedding held up — so the encoder is 31 unused
quantizers and a conv stack we would otherwise have to port and ship.

Two transforms, and only two, change any values:

  * The codebooks are stored EMA-style, as a running sum and a usage count.
    The reference divides one by the other on every decode; we do it once here
    so the engine can index a plain table. Reading `embedding_sum` raw and
    skipping this yields noise that sounds like a subtle decoder bug.
  * The speaker encoder is widened from bf16 to f32, which is exact. It is
    10M parameters, so the size costs nothing, and it removes a precision
    question from a net that pools statistics over a whole clip.

Everything else is copied byte for byte, so the engine can be checked against
the reference tensor by tensor. The talker keeps its released bf16 — it is the
precision h3.c runs natively — and the codec decoder keeps its released f32,
on the same reasoning that kept Stable Audio's decoder wide.

Usage:
  convert-qwen3-tts-package.py --model-dir qwen3-tts-base \
      --out Qwen3-TTS-12Hz-0.6B-Base
"""

import argparse
import json
import os
import shutil
import struct
import sys

import numpy as np

# Subsystem -> (output file, prefix to strip). Order matters: the code
# predictor lives under the talker's prefix, so it has to match first.
ROUTES = [
    ("talker.code_predictor.", "code_predictor.safetensors"),
    ("talker.", "talker.safetensors"),
    ("speaker_encoder.", "speaker_encoder.safetensors"),
]

# The mel front end the speaker encoder expects. These live in the reference's
# Python rather than in any config file, so record them where the engine can
# read them back instead of leaving them to be rediscovered.
MEL = {
    "n_fft": 1024, "num_mels": 128, "sampling_rate": 24000,
    "hop_size": 256, "win_size": 1024, "fmin": 0, "fmax": 12000,
}

NOTICE = """Qwen3-TTS by Alibaba Cloud, licensed under the Apache License 2.0.

Modifications by H3ddle: the weights were repackaged from the released
safetensors into one file per subsystem. Tensor values are unchanged except
that the codec's EMA codebooks were folded to plain embeddings
(embedding_sum / max(cluster_usage, 1e-5)), which is the same division the
reference performs at decode time, and the speaker encoder was widened from
bfloat16 to float32, which is exact. The Mimi codec encoder was omitted: it
serves in-context voice cloning, which this application does not use. No
weights were retrained, merged, pruned or quantized.
"""

DTYPE_SIZES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8,
               "I8": 1, "I16": 2, "I32": 4, "I64": 8, "U8": 1, "BOOL": 1}

CODEBOOK_EPSILON = 1e-5


def read_safetensors(path):
    """Returns (ordered dict of name -> (dtype, shape, bytes), metadata).

    Tensors are kept as raw bytes so that anything passing through untouched
    never has to survive a round trip through a dtype NumPy cannot spell —
    bf16 in particular.
    """
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
        blob = handle.read()
    metadata = header.pop("__metadata__", {})
    tensors = {}
    for name, entry in header.items():
        start, end = entry["data_offsets"]
        tensors[name] = (entry["dtype"], entry["shape"], blob[start:end])
    return tensors, metadata


def write_safetensors(path, tensors, metadata):
    header = {"__metadata__": metadata}
    offset = 0
    ordered = sorted(tensors)
    for name in ordered:
        dtype, shape, payload = tensors[name]
        expected = DTYPE_SIZES[dtype]
        for dimension in shape:
            expected *= dimension
        if expected != len(payload):
            raise SystemExit(f"{name}: {len(payload)} bytes, expected {expected}")
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + len(payload)]}
        offset += len(payload)

    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)     # the data section wants 8-byte alignment
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for name in ordered:
            handle.write(tensors[name][2])


def to_float32(tensor):
    """Widen a stored tensor to f32. bf16 is the top half of an f32, so the
    widening is a shift rather than a conversion, and is exact."""
    dtype, shape, payload = tensor
    if dtype == "F32":
        return np.frombuffer(payload, dtype=np.float32).reshape(shape)
    if dtype == "BF16":
        raw = np.frombuffer(payload, dtype=np.uint16).astype(np.uint32) << 16
        return raw.view(np.float32).reshape(shape)
    if dtype == "F16":
        return np.frombuffer(payload, dtype=np.float16).astype(np.float32).reshape(shape)
    raise SystemExit(f"cannot widen {dtype}")


def as_tensor(array):
    array = np.ascontiguousarray(array, dtype=np.float32)
    return ("F32", list(array.shape), array.tobytes())


def fold_codebooks(tensors):
    """Replace every (embedding_sum, cluster_usage) pair with one embedding.

    Returns the number folded, so a release that changes this layout fails
    loudly here rather than sounding wrong later.
    """
    folded = 0
    for name in [n for n in tensors if n.endswith("_codebook.embedding_sum")]:
        stem = name[: -len("embedding_sum")]
        usage_name = stem + "cluster_usage"
        if usage_name not in tensors:
            raise SystemExit(f"{name}: no matching cluster_usage")
        total = to_float32(tensors.pop(name))
        usage = to_float32(tensors.pop(usage_name))
        embedding = total / np.maximum(usage, CODEBOOK_EPSILON)[:, None]
        tensors[stem + "embedding"] = as_tensor(embedding)
        folded += 1
    return folded


def write_tokenizer(model_dir, out_dir):
    """Assemble a fast tokenizer.json from the released slow-format files.

    The engine requires type BPE, a null unk_token and an NFC normalizer; it
    accepts merges either as "left right" strings or as pairs.
    """
    with open(os.path.join(model_dir, "vocab.json")) as handle:
        vocab = json.load(handle)
    with open(os.path.join(model_dir, "merges.txt")) as handle:
        merges = [line.rstrip("\n") for line in handle
                  if line.strip() and not line.startswith("#version")]
    with open(os.path.join(model_dir, "tokenizer_config.json")) as handle:
        added = json.load(handle).get("added_tokens_decoder", {})

    tokenizer = {
        "version": "1.0",
        "added_tokens": [
            {"id": int(identifier), "content": entry["content"],
             "single_word": entry.get("single_word", False),
             "lstrip": entry.get("lstrip", False),
             "rstrip": entry.get("rstrip", False),
             "normalized": entry.get("normalized", False),
             "special": entry.get("special", True)}
            for identifier, entry in sorted(added.items(), key=lambda x: int(x[0]))
        ],
        "normalizer": {"type": "NFC"},
        "model": {"type": "BPE", "unk_token": None, "vocab": vocab,
                  "merges": merges},
    }
    path = os.path.join(out_dir, "tokenizer.json")
    with open(path, "w") as handle:
        json.dump(tokenizer, handle, ensure_ascii=False, separators=(",", ":"))
    print(f"{'tokenizer.json':<28} {len(vocab):>4} entries  "
          f"{len(merges)} merges  {len(tokenizer['added_tokens'])} added  "
          f"{os.path.getsize(path) / 1e6:>7.1f} MB")


def report(name, tensors):
    params = 0
    size = 0
    for dtype, shape, payload in tensors.values():
        count = 1
        for dimension in shape:
            count *= dimension
        params += count
        size += len(payload)
    print(f"{name:<28} {len(tensors):>4} tensors  {params / 1e6:>7.1f}M params  "
          f"{size / 1e6:>7.1f} MB")
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True,
                        help="checkout of Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--out", required=True, help="package directory to write")
    args = parser.parse_args()

    backbone = os.path.join(args.model_dir, "model.safetensors")
    codec = os.path.join(args.model_dir, "speech_tokenizer", "model.safetensors")
    for path in (backbone, codec):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.model_dir, "config.json")) as handle:
        config = json.load(handle)
    with open(os.path.join(args.model_dir, "speech_tokenizer",
                           "config.json")) as handle:
        codec_config = json.load(handle)

    common = {"source_repo": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
              "license": "Apache-2.0"}

    # ---- backbone: talker, code predictor, speaker encoder ---------------
    source, _ = read_safetensors(backbone)
    buckets = {output: {} for _, output in ROUTES}
    for name in sorted(source):
        for prefix, output in ROUTES:
            if name.startswith(prefix):
                buckets[output][name[len(prefix):]] = source[name]
                break
        else:
            raise SystemExit(f"{name}: no route for this tensor")

    speaker = buckets["speaker_encoder.safetensors"]
    for name in list(speaker):
        speaker[name] = as_tensor(to_float32(speaker[name]))

    total = 0
    metadata = dict(common, config=json.dumps(config["talker_config"]))
    total += report("talker.safetensors", buckets["talker.safetensors"])
    write_safetensors(os.path.join(args.out, "talker.safetensors"),
                      buckets["talker.safetensors"], metadata)

    predictor_config = config["talker_config"]["code_predictor_config"]
    metadata = dict(common, config=json.dumps(predictor_config))
    total += report("code_predictor.safetensors",
                    buckets["code_predictor.safetensors"])
    write_safetensors(os.path.join(args.out, "code_predictor.safetensors"),
                      buckets["code_predictor.safetensors"], metadata)

    metadata = dict(common, config=json.dumps(config["speaker_encoder_config"]),
                    mel=json.dumps(MEL), precision="widened from bf16 to f32")
    total += report("speaker_encoder.safetensors", speaker)
    write_safetensors(os.path.join(args.out, "speaker_encoder.safetensors"),
                      speaker, metadata)

    # ---- speech tokenizer: the decoder half only -------------------------
    source, _ = read_safetensors(codec)
    decoder = {}
    dropped = 0
    for name in sorted(source):
        if name.startswith("encoder."):
            dropped += 1
            continue
        if not name.startswith("decoder."):
            raise SystemExit(f"{name}: neither encoder nor decoder")
        decoder[name[len("decoder."):]] = source[name]
    folded = fold_codebooks(decoder)
    if folded != codec_config["decoder_config"]["num_quantizers"]:
        raise SystemExit(f"folded {folded} codebooks, expected "
                         f"{codec_config['decoder_config']['num_quantizers']}")

    metadata = dict(common, config=json.dumps(codec_config["decoder_config"]),
                    codebooks="EMA sums folded to embeddings")
    total += report("codec_decoder.safetensors", decoder)
    write_safetensors(os.path.join(args.out, "codec_decoder.safetensors"),
                      decoder, metadata)
    print(f"{'(mimi encoder omitted)':<28} {dropped:>4} tensors dropped")

    # ---- tokenizer -------------------------------------------------------
    # Qwen3-TTS ships the slow format, vocab.json plus merges.txt, but the
    # engine's tokenizer reads the fast tokenizer.json. Assembling one here
    # means no engine change: this vocabulary and merge table are byte for
    # byte the ones H3 already tokenizes with, so h3_tokenizer handles it
    # unmodified. The TTS special ids (151671-151673) never come from text —
    # the prompt builder uses them as integers — so they only need to be
    # present, not reachable through the merges.
    write_tokenizer(args.model_dir, args.out)

    for name in ("generation_config.json",):
        source_path = os.path.join(args.model_dir, name)
        if os.path.exists(source_path):
            shutil.copyfile(source_path, os.path.join(args.out, name))
            print(f"{name:<28} {os.path.getsize(source_path) / 1e6:>28.1f} MB")
        else:
            print(f"warning: {name} not found", file=sys.stderr)

    with open(os.path.join(args.out, "NOTICE"), "w") as handle:
        handle.write(NOTICE)

    print(f"\n{total / 1e6:.1f}M parameters total -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
