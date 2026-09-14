#!/usr/bin/env python3
"""Replay a verified YuE2 sample to compare attention tiling and solver steps.

Uses the same score, semantic tokens, seed, weights, and midpoint solver.
All outputs are experimental; fewer steps require a listening comparison.
Setup: docs/experiments/2026-09-13-yue2-mac.md.
"""

import argparse
import json
import os
from pathlib import Path
import time
import traceback

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--vae", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path,
                        help="artifacts directory from probe-yue2-mac.py")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "stage": "load", "trials": []}
    pipe = None

    def save():
        temporary = args.output / "benchmark.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output / "benchmark.json")

    try:
        import importlib.metadata
        import numpy as np
        import soundfile as sf
        import torch
        from yue2 import YuE2Pipeline
        from yue2.nar import synthesize
        from yue2.storage import verify_result

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS GPU access is required")
        if importlib.metadata.version("yue2-infer") != "0.1.5":
            raise RuntimeError("Requires yue2-infer 0.1.5")
        source = verify_result(args.reference)
        config = json.loads((args.reference / "config.json").read_text())
        if config["generation"]["ode_steps"] != 32:
            raise ValueError("Use a reference generated with the original 32-step solver")
        request = json.loads((args.reference / "request.json").read_text())
        prefix = np.load(args.reference / "prefix.npy", allow_pickle=False).tolist()
        tokens = np.load(args.reference / "semantic.npy", allow_pickle=False).tolist()
        reference = np.load(args.reference / "latent.npy", allow_pickle=False)
        pipe = YuE2Pipeline.from_pretrained(
            str(args.model.resolve()), vae=str(args.vae.resolve()),
            device="mps", backend="torch-eager", local_files_only=True,
        )
        if source["weights"] != pipe.weights:
            raise ValueError("Benchmark weights must match the reference")
        report.update(reference_identity=source["identity"], reference_config=config,
                      dependencies={name: importlib.metadata.version(name)
                                    for name in ("torch", "yue2-infer", "numpy")},
                      seed=request["seed"], semantic_tokens=len(tokens),
                      weights=pipe.weights)
        model = pipe._load_model(for_nar=True)

        def solve(steps, tile):
            return synthesize(model, prefix, tokens, request["seed"],
                              steps=steps, query_chunk_size=tile).numpy()

        report["stage"] = "warmup"
        save()
        for tile in (256, 1024):
            print(f"Warmup: tile={tile}", flush=True)
            solve(1, tile)
            torch.mps.synchronize()

        # ABBA helps distinguish scheduling gains from warmup or thermal drift.
        # Reduce steps only after comparing both attention schedules at 32.
        schedule = [(32, 256), (32, 1024), (32, 1024), (32, 256)]
        for index in range(6):
            if index == 4:
                averages = {tile: np.mean([t["synthesis_seconds"] for t in report["trials"]
                                           if t["query_tile"] == tile]) for tile in (256, 1024)}
                faster_tile = min(averages, key=averages.get)
                report["selected_query_tile"] = faster_tile
                schedule.extend([(16, faster_tile), (8, faster_tile)])
            steps, tile = schedule[index]
            name = f"trial-{index + 1}-steps-{steps}-tile-{tile}"
            report["stage"] = name
            save()
            print(f"Starting {name}", flush=True)
            torch.mps.synchronize()
            start = time.perf_counter()
            latents = solve(steps, tile)
            torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            if latents.shape != reference.shape or not np.isfinite(latents).all():
                raise RuntimeError("Invalid synthesized latents")
            difference = latents.astype(np.float64) - reference
            rmse = float(np.sqrt(np.mean(difference ** 2)))
            reference_rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
            trial = {"name": name, "steps": steps, "query_tile": tile,
                     "synthesis_seconds": elapsed,
                     "latents_identical_to_reference": bool(np.array_equal(latents, reference)),
                     "latent_rmse": rmse, "latent_relative_rmse": rmse / max(reference_rms, 1e-12),
                     "mps_allocated_bytes_after": torch.mps.current_allocated_memory(),
                     "mps_driver_bytes_after": torch.mps.driver_allocated_memory()}
            np.save(args.output / f"{name}.npy", latents)
            report["trials"].append(trial)
            save()
            print(json.dumps(trial), flush=True)

        # Decode after timing every synthesis run, retaining GPU weights across
        # trials and avoiding CPU/GPU reloads inside the timing comparison.
        report["stage"] = "decode"
        save()
        reference_audio, sample_rate = sf.read(args.reference / "audio.flac", always_2d=True)
        for trial in report["trials"]:
            latents = np.load(args.output / f"{trial['name']}.npy", allow_pickle=False)
            start = time.perf_counter()
            audio = pipe.decode(latents)
            torch.mps.synchronize()
            if audio.shape != reference_audio.shape or not np.isfinite(audio).all():
                raise RuntimeError("Invalid decoded audio")
            trial["decode_seconds"] = time.perf_counter() - start
            trial["audio_rms"] = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            trial["audio_peak"] = float(np.max(np.abs(audio)))
            trial["audio_seconds"] = len(audio) / sample_rate
            trial["audio_correlation_to_reference"] = float(np.corrcoef(audio.ravel(), reference_audio.ravel())[0, 1])
            if trial["audio_rms"] < 1e-6:
                raise RuntimeError("Effectively silent output")
            sf.write(args.output / f"{trial['name']}.wav", audio, sample_rate, subtype="PCM_24")
            save()
        report.update(status="passed", stage="complete")
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        save()
        if pipe is not None:
            pipe.close()


if __name__ == "__main__":
    main()
