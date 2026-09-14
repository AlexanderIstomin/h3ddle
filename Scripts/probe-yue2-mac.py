#!/usr/bin/env python3
"""Run the optional YuE2 0.1.5 MPS feasibility probe with local weights.

Requires the upstream wheel and its pinned dependencies in a separate venv.
Never downloads weights or changes the application. See the feasibility note
in docs/experiments for setup, limitations, and measured results.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import time
import traceback

# Unsupported GPU operators must fail visibly rather than silently falling
# back to CPU. Upstream intentionally samples random values on CPU for MPS.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--vae", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=positive_int, default=250)
    parser.add_argument("--steps", type=positive_int, default=32,
                        help="Acoustic midpoint steps; fewer than 32 is experimental")
    parser.add_argument("--cot", choices=("off", "melody", "full"), default="off")
    parser.add_argument("--seed", type=int, default=831001)
    parser.add_argument("--timeout", type=positive_int, default=1800)
    parser.add_argument("--style", default="Gentle acoustic folk, warm singing voice, acoustic guitar, soft piano, relaxed tempo")
    parser.add_argument("--lyrics", default="[Verse]\nMorning light across the bay\nSilver waves carry dreams away\n[Chorus]\nStay with me beside the sea\nLet the wind sing soft and free")
    args = parser.parse_args()
    for folder in (args.model, args.vae):
        if not folder.is_dir():
            parser.error(f"Missing local model folder: {folder}")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running", "stage": "environment",
        "macos": platform.mac_ver()[0], "architecture": platform.machine(),
        "python": platform.python_version(), "automatic_mps_cpu_fallback": False,
        "max_tokens": args.max_tokens, "steps": args.steps, "cot": args.cot, "seed": args.seed,
        "stage_seconds": {}, "memory_observations": [],
    }
    started = time.perf_counter()
    pipe = None

    def save_report():
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = args.output / "probe.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output / "probe.json")

    try:
        import numpy as np
        import torch
        from yue2 import YuE2Pipeline
        from yue2.pipeline import SongResult
        from yue2.protocol import GenerationConfig, SongRequest

        report["dependencies"] = {
            name: importlib.metadata.version(name)
            for name in ("yue2-infer", "torch", "transformers", "numpy", "soundfile", "huggingface-hub")
        }
        if platform.system() != "Darwin" or not torch.backends.mps.is_available():
            raise RuntimeError("This probe requires macOS with visible MPS GPU access")
        if report["dependencies"]["yue2-infer"] != "0.1.5":
            raise RuntimeError("This probe was written for yue2-infer 0.1.5")
        report["mps_recommended_max_bytes"] = torch.mps.recommended_max_memory()

        def memory():
            report["memory_observations"].append({
                "stage": report["stage"], "elapsed_seconds": time.perf_counter() - started,
                "mps_allocated_bytes": torch.mps.current_allocated_memory(),
                "mps_driver_bytes": torch.mps.driver_allocated_memory(),
            })

        def stage(name, operation):
            report["stage"] = name
            save_report()
            print(f"Starting {name}", flush=True)
            start = time.perf_counter()
            result = operation()
            torch.mps.synchronize()
            report["stage_seconds"][name] = time.perf_counter() - start
            memory()
            save_report()
            return result

        def cancelled():
            return time.perf_counter() - started > args.timeout

        counts = {}

        def on_token(phase, token):
            counts[phase] = counts.get(phase, 0) + 1
            if counts[phase] % 25 == 0:
                memory()
                report["tokens_observed"] = dict(counts)
                save_report()

        pipe = stage("load_and_verify", lambda: YuE2Pipeline.from_pretrained(
            str(args.model.resolve()), vae=str(args.vae.resolve()),
            local_files_only=True, device="mps", backend="torch-eager",
            generation_config=GenerationConfig(ode_steps=args.steps),
        ))
        sampling = {"min_tokens": min(200, args.max_tokens), "max_tokens": args.max_tokens}
        request = SongRequest(style=args.style, lyrics=args.lyrics, cot=args.cot, seed=args.seed)
        plan = stage("plan", lambda: pipe.plan(request=request, cancelled=cancelled, on_token=on_token))
        plan.save(args.output / "plan")
        semantic = stage("semantic", lambda: pipe.generate_semantic(
            plan, sampling=sampling, cancelled=cancelled, on_token=on_token))
        np.save(args.output / "semantic.npy", np.asarray(semantic.tokens, dtype=np.int32))
        latents = stage("synthesis", lambda: pipe.synthesize(semantic, cancelled=cancelled))
        np.save(args.output / "latent.npy", latents)
        audio = stage("decode", lambda: pipe.decode(latents))
        if audio.ndim != 2 or audio.shape[1] != 2 or not len(audio) or not np.isfinite(audio).all():
            raise RuntimeError("Expected nonempty, finite stereo audio")
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < 1e-6:
            raise RuntimeError("Generated audio is effectively silent")
        config = pipe.effective_config(request, semantic_sampling=sampling)
        timing = {"abc": plan.timing, "semantic": semantic.timing,
                  "nar_seconds": report["stage_seconds"]["synthesis"],
                  "vae_seconds": report["stage_seconds"]["decode"],
                  "load": pipe.load_timing, "e2e_seconds": time.perf_counter() - started}
        from yue2.storage import identity
        request_id = identity({"request": request.to_dict(), "config": config, "weights": pipe.weights})
        song = SongResult(audio, 48000, semantic, latents, config, pipe.weights, timing, request_id)
        song.save_artifacts(args.output / "artifacts")
        song.save(args.output / "audio.wav")
        report.update(status="passed", stage="complete", audio_seconds=len(audio) / 48000,
                      sample_rate=48000, channels=2, peak_amplitude=peak, rms_amplitude=rms,
                      clipped_fraction=float(np.mean(np.abs(audio) >= 1)),
                      truncated=song.truncated, timing=timing)
        print(json.dumps({k: report[k] for k in ("status", "audio_seconds", "stage_seconds", "truncated")}), flush=True)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        save_report()
        if pipe is not None:
            pipe.close()


if __name__ == "__main__":
    main()
