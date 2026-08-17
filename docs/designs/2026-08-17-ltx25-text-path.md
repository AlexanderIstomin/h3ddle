# LTX-2.5's text conditioning path, against the released files

Every stage of this path has a check against LTX's or Google's own code, at toy
dimensions. None of them have been run together, and none on the released
weights. This is what the released files say, gathered before writing the
runner, because three assumptions taken from the reference implementation have
already turned out to disagree with what shipped.

## Where the weights actually are

Not where the two-phase split would want them.

| | file | size |
|---|---|---|
| tokenizer | `tokenizer_json`, inside the text encoder | 32 MB |
| embedding, 48 Gemma layers, final norm | text encoder | 15.4 GB |
| aggregation projections | text encoder, `text_embedding_projection.*` | 2.3 GB |
| **connector, both streams** | **the DiT file** | **2.0 GB, 8 blocks each** |

The connector living in the DiT checkpoint is the awkward one. The conditioning
path cannot be completed without opening the 21.5 GB file, so either the
connector runs after the DiT is resident, or its 2 GB is read out separately.
Reading it separately is the one that fits in memory, see below.

The aggregation weight is `[4096, 188160]`, and 188160 is 3840 x 49 — the
embedding output plus one per layer, which is the count the aggregation check
was built against.

## The memory constraint decides the shape

The DiT is 21.5 GB and the text encoder 15.4 GB, on a 32 GB machine. They
cannot both be resident, so the entry point is two-phase by construction rather
than as an optimization: encode, free, denoise. H3 already pays a fixed 176 s
for its own encoder load, so there is a pattern to follow.

That makes the connector's placement a real decision rather than a detail. It
needs the DiT file open while the encoder's output is still live.

## What the released config confirms

Read from the safetensors `__metadata__`, not from the library. Everything
here was already implemented against the reference; this is the confirmation,
and it is worth having in writing because the one time these disagreed it was
silent.

| | |
|---|---|
| 48 layers, hidden 3840, intermediate 15360 | as implemented |
| 16 query heads; 8 KV heads of 256 sliding, 1 of 512 global | as implemented |
| `full_attention` at 5, 11, 17, 23, 29, 35, 41, 47 | as implemented |
| `attention_k_eq_v` true — global layers have no `v_proj` | as implemented |
| `hidden_size_per_layer_input` 0 | as implemented |
| sliding rope: theta 10,000, whole head | as implemented |
| global rope: theta 1,000,000, `partial_rotary_factor` 0.25 | as implemented |
| `hidden_activation` `gelu_pytorch_tanh` | as implemented |
| `use_bidirectional_attention` `vision` — so text is causal | as implemented |

The partial rotary factor is the one worth naming. A 512-wide head rotates 128
of its channels, so 64 of 256 frequency pairs move and 192 stay at identity.
Rotating the whole head would be the natural reading and would corrupt all
eight global layers while producing perfectly well-formed conditioning.

`rope_type` is `proportional` on the global layers and `default` on the
sliding ones. Only the second is modelled today. Whether `proportional` means
anything beyond the theta already applied has **not** been established, and it
is the one open question in this table.

## What is checked, and what that leaves

Verified at toy dimensions, each against the vendor implementation with a
mutation pass: the tokenizer, the Gemma block, the rotary application, the
aggregation, and the connector. Verified on released weights: the tokenizer,
against 22 reference cases through the embedded `tokenizer_json`; and nine int8
projections through `h3_weight_load_i8_linear`, four from the DiT and five from
the tower.

What that leaves for the runner is assembly, and the seams are where this will
fail: 48 layers wired in the right order with the right per-layer type, the 49
hidden states collected in the order the aggregation expects, and ConvRot
applied to each activation with the group the marker gives. Projections sharing
an input must agree on that group, which `h3_text_encoder.c` already checks for
Qwen and which is worth checking here rather than assuming.

## Do this first

Resolve `rope_type: proportional`. It is the only entry above that is not
already implemented and confirmed, it affects eight of forty-eight layers, and
getting it wrong yields conditioning that is plausible rather than obviously
broken — which is the failure mode this whole path has been built to avoid.
