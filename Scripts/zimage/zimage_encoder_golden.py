#!/usr/bin/env python3
"""Export what Z-Image's text encoder produces, so the C port can be checked
against it rather than against a reading of the config.

Qwen3-4B is the same decoder block H3's text encoder and the TTS talker
already run, at a third set of shapes — 2560 wide, 36 layers, 32 query heads
against 8 key/value heads. What has to be established is that it really is the
same arithmetic, and where the encoder's output is taken from.

Both candidates are exported, because the pipeline's choice between them is not
visible in any config: the final block's output before `model.norm`, and
`last_hidden_state` after it. Getting this wrong is silent — the shapes agree
and only the values differ.

  zimage_encoder_golden.py --model <base checkout> --out golden.safetensors
"""

import argparse

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3Model

# Deliberately mixed: ASCII, an accent, and Chinese, because the model is
# bilingual and a tokenizer fault would otherwise hide behind plain English.
PROMPT = ("A rain-slicked Tokyo alley at night, neon reflected in the puddles, "
          "一只黑猫 sitting under a café awning.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--f32", action="store_true",
                        help="run the reference in f32, so only the port's "
                             "own error is measured")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(f"{args.model}/tokenizer")
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    print(f"prompt: {ids.shape[1]} tokens")
    print("first ids:", ids[0, :12].tolist())

    # The dtype is the point of the comparison, not an incidental setting: at
    # bfloat16 the reference carries the same activation noise the C port is
    # being measured for, and the two errors are then indistinguishable.
    dtype = torch.float32 if args.f32 else torch.bfloat16
    model = Qwen3Model.from_pretrained(f"{args.model}/text_encoder", dtype=dtype)
    model.eval()
    print(f"loaded: {model.config.num_hidden_layers} layers, "
          f"{model.config.hidden_size} wide, "
          f"{model.config.num_attention_heads}/{model.config.num_key_value_heads} heads, "
          f"head_dim {model.config.head_dim}, theta {model.config.rope_theta}")

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    # hidden_states[0] is the embedding, [i] the output of block i-1, and the
    # last entry is *not* normed; last_hidden_state is.
    states = out.hidden_states
    tensors = {
        # f32, not i32: the reader is typed and a 151936 vocabulary is
        # exactly representable well below f32's 2^24 integer limit.
        "input_ids": ids.float(),
        "embedding": states[0].float(),
        "block_00": states[1].float(),
        "block_17": states[18].float(),
        "block_35_prenorm": states[-1].float(),
        "last_hidden_state": out.last_hidden_state.float(),
    }
    for name, value in tensors.items():
        print(f"  {name:20} {tuple(value.shape)} {value.dtype}")

    # The two candidates differ, and by how much decides whether picking the
    # wrong one would ever be noticed.
    a, b = tensors["block_35_prenorm"], tensors["last_hidden_state"]
    delta = (a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()
    print(f"\nprenorm vs normed: {delta:.3f} RMS relative — "
          f"{'distinguishable' if delta > 0.05 else 'nearly identical'}")

    # In f32 the last hidden state and the final entry of hidden_states are
    # literally the same storage, which safetensors refuses to write twice.
    save_file({name: value.contiguous().clone() for name, value in tensors.items()},
              args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
