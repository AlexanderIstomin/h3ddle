# 0005: Curate pinned managed-model packages

## Status

Accepted.

## Context

The original BF16 H3 release is impractical on most supported Macs. Optimized
community checkpoints differ in quantization, tensor names, directory layout,
and required runtime support. Offering an unrestricted repository picker would
therefore let the app download many large but unusable packages. Model transfers
also need to survive cancellation and app restarts without placing weights in
the source tree or app bundle.

## Decision

H3ddle exposes a small catalog of immutable package manifests. Each manifest
pins a repository revision and records the selected file paths, exact sizes,
SHA-256 hashes, upstream license, minimum-memory guidance, and engine
compatibility state.

Downloads stream into a package-specific staging directory under Application
Support, resume with HTTP byte ranges, verify locally, and install by moving the
verified staging directory on the same volume. The initial catalog entry selects
the FL2VA pruned INT8 ConvRot transformer, INT8 text encoder, and two VAEs from
`Comfy-Org/MiniMax-H3`, plus the tokenizer and runtime configuration files from
the pinned official `MiniMaxAI/MiniMax-H3` revision. It is explicitly marked as
generation-ready for prompt-only FL2VA after real native probes cover optimized
Qwen, the full streamed DiT, both VAEs, and final audiovisual muxing. Readiness
does not imply visual-reference conditioning support.

## Consequences

- The approximately 53.9 GB package can generate prompt-only audiovisual clips
  without downloading the repository's other checkpoints.
- Published files changing in place fail verification instead of silently
  changing runtime behavior.
- Cancellation retains useful partial data, while atomic installation prevents
  the engine from observing an incomplete managed package.
- A new checkpoint or quantization requires a reviewed manifest and engine
  compatibility decision.
- Users can still select an existing local model folder independently.
