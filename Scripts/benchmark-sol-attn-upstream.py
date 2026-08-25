#!/usr/bin/env python3
"""Compare NVIDIA's unmodified MPS Sol-Attn with H3ddle-style dense SDPA.

This is an investigation tool, not an application dependency. Point it at a
pinned Sana checkout and run it with a recent PyTorch nightly that exposes
``torch.mps.compile_shader``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sparse-backends",
        type=Path,
        required=True,
        help="Sana techniques/sparse_backends directory",
    )
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[976, 1550, 4128, 15485]
    )
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tau", type=float, nargs="+", default=[1.3])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=65859680)
    parser.add_argument(
        "--capture-prefix",
        type=Path,
        help="load PREFIX.{json,q.bf16,k.bf16,v.bf16} instead of random inputs",
    )
    return parser.parse_args()


def timed(callable, iterations: int) -> tuple[float, object]:
    samples: list[float] = []
    result = None
    for _ in range(iterations):
        started = time.perf_counter()
        result = callable()
        torch.mps.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), result


def main() -> int:
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if not hasattr(torch.mps, "compile_shader"):
        raise SystemExit("this benchmark needs torch.mps.compile_shader")
    sys.path.insert(0, str(args.sparse_backends))
    from sol_attn.mps.metal import _routing_debug_mps, sol_attn_tiled_mps

    print(
        f"PyTorch {torch.__version__}; tau={args.tau}; "
        f"heads={args.heads}; dim={args.head_dim}"
    )
    device = torch.device("mps")
    scale = 1.0 / math.sqrt(args.head_dim)
    torch.manual_seed(args.seed)

    cases: list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    if args.capture_prefix:
        metadata_path = Path(f"{args.capture_prefix}.json")
        metadata = json.loads(metadata_path.read_text())
        tokens = int(metadata["tokens"])
        heads = int(metadata["heads"])
        head_dim = int(metadata["head_dim"])
        if heads != args.heads or head_dim != args.head_dim:
            raise SystemExit("capture shape disagrees with --heads/--head-dim")
        shape = (1, tokens, heads, head_dim)

        def captured(suffix: str) -> torch.Tensor:
            payload = bytearray(
                Path(f"{args.capture_prefix}.{suffix}.bf16").read_bytes()
            )
            expected = math.prod(shape) * 2
            if len(payload) != expected:
                raise SystemExit(
                    f"{suffix} capture has {len(payload)} bytes, expected {expected}"
                )
            return (
                torch.frombuffer(payload, dtype=torch.bfloat16)
                .reshape(shape)
                .to(device)
            )

        cases.append(
            (
                tokens,
                int(metadata["sink_tokens"]),
                captured("q"),
                captured("k"),
                captured("v"),
            )
        )
    else:
        for tokens in args.tokens:
            shape = (1, tokens, args.heads, args.head_dim)
            cases.append(
                (
                    tokens,
                    0,
                    torch.randn(shape, device=device, dtype=torch.bfloat16),
                    torch.randn(shape, device=device, dtype=torch.bfloat16),
                    torch.randn(shape, device=device, dtype=torch.bfloat16),
                )
            )

    for tokens, sink_tokens, query, key, value in cases:
        sink_blocks = max(8, (sink_tokens + 63) // 64)

        def dense():
            # H3ddle promotes long BF16 attention to F32 because MPSGraph's
            # BF16 SDPA is slower on the measured M1 shapes.
            output = F.scaled_dot_product_attention(
                query.float().transpose(1, 2),
                key.float().transpose(1, 2),
                value.float().transpose(1, 2),
                scale=scale,
            )
            return output.transpose(1, 2).to(torch.bfloat16)

        def sparse(tau: float):
            return sol_attn_tiled_mps(
                query,
                key,
                value,
                scale=scale,
                tau=tau,
                thresh_type="diag",
                sink_blocks=(0, min(sink_blocks, (tokens + 63) // 64)),
                query_block_size=64,
            )

        # Compile and warm both paths outside measurement.
        dense_reference = dense()
        sparse_output = sparse(args.tau[0])
        torch.mps.synchronize()

        dense_seconds, dense_reference = timed(dense, args.iterations)
        dense_f32 = dense_reference.float()
        for tau in args.tau:
            sparse_seconds, sparse_output = timed(
                lambda: sparse(tau), args.iterations
            )
            routes, _, _, _ = _routing_debug_mps(
                query,
                key,
                scale=scale,
                tau=tau,
                thresh_type="diag",
                sink_blocks=(0, min(sink_blocks, (tokens + 63) // 64)),
            )
            density = routes.float().mean().item()
            sparse_f32 = sparse_output.float()
            cosine = F.cosine_similarity(
                dense_f32.reshape(1, -1), sparse_f32.reshape(1, -1)
            ).item()
            mae = (dense_f32 - sparse_f32).abs().mean().item()
            print(
                f"tokens={tokens:6d} tau={tau:4.1f} "
                f"dense={dense_seconds:8.4f}s sol={sparse_seconds:8.4f}s "
                f"speedup={dense_seconds / sparse_seconds:5.2f}x "
                f"routes={density:6.2%} cosine={cosine:.6f} mae={mae:.6g}"
            )
            del sparse_output, routes
        del query, key, value, dense_reference
        torch.mps.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
