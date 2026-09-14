#!/usr/bin/env python3
"""Optional AuK MPS feasibility probe; uses an isolated upstream installation.

Weights, upstream source, and generated audio stay outside the repository.
Run under Scripts/run-with-timeout.py to bound model loading and GPU calls.
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

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.9")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=831001)
    parser.add_argument("--instruction", default='Generate speech based on the following description: "A calm, clear English voice". The content to speak is: "Good morning. Welcome home."')
    args = parser.parse_args()
    if not 0 < args.seconds <= 15:
        parser.error("seconds must be positive and at most 15 for this probe")
    for path in (args.model, args.qwen):
        if not path.is_dir():
            parser.error(f"Missing model directory: {path}")
    if args.audio and not args.audio.is_file():
        parser.error("Missing source audio")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status="running", stage="environment", stage_seconds={},
                  memory_observations=[], macos=platform.mac_ver()[0],
                  architecture=platform.machine(), python=platform.python_version(),
                  seed=args.seed, target_seconds=args.seconds, instruction=args.instruction,
                  reference_audio=bool(args.audio), device="mps", source_modifications=False,
                  automatic_mps_cpu_fallback=False)
    report["mps_watermarks"] = {key: os.environ[key] for key in (
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO")}

    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        tmp = args.output / "probe.json.tmp"
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.output / "probe.json")

    try:
        import numpy as np
        import soundfile as sf
        import torch
        from auk.infer.infer_auk import AukInfer

        report["dependencies"] = {name: importlib.metadata.version(name) for name in
                                  ("auk", "torch", "torchaudio", "transformers", "torchdiffeq", "numpy", "soundfile")}
        if not torch.backends.mps.is_available():
            raise RuntimeError("Real MPS GPU access is required; CPU fallback is disabled")
        report["mps_recommended_max_bytes"] = torch.mps.recommended_max_memory()

        def observe():
            report["memory_observations"].append(dict(
                stage=report["stage"], elapsed_seconds=time.perf_counter() - started,
                allocated_bytes=torch.mps.current_allocated_memory(),
                driver_bytes=torch.mps.driver_allocated_memory()))

        def stage(name, operation):
            report["stage"] = name
            save()
            print(f"Starting {name}", flush=True)
            tick = time.perf_counter()
            result = operation()
            torch.mps.synchronize()
            report["stage_seconds"][name] = time.perf_counter() - tick
            report["stage"] = name
            observe()
            save()
            return result

        checkpoint = args.model / "auk_flash.safetensors"
        if not checkpoint.exists():
            checkpoint = args.model / "auk_base.safetensors"
        engine = stage("load", lambda: AukInfer(str(args.model / "config.yaml"),
                       str(checkpoint), qwen_path=str(args.qwen), device="mps"))
        report["variant"] = "flash" if engine.is_flash else "base"
        report["parameter_bytes"] = {
            key: sum(p.numel() * p.element_size() for p in model.parameters())
            for key, model in (("conditioner_and_dit", engine.model), ("vae", engine.vae_model))}
        # Observe actual public stages without replacing their computation.
        for obj, method, label in (
            (engine.model, "encode_text", "conditioning"),
            (engine.vae_model, "encoding_and_normalization", "reference_encode"),
            (engine.vae_model, "inference_from_latents", "decode"),
        ):
            original = getattr(obj, method)
            def measured(*a, _original=original, _label=label, **kw):
                return stage(_label, lambda: _original(*a, **kw))
            setattr(obj, method, measured)

        forwards = [0]
        def tick(_module, _inputs, _output):
            forwards[0] += 1
            report["stage"] = "sampling"
            report["transformer_forwards"] = forwards[0]
            observe()
            save()
            print(f"Transformer forward {forwards[0]}", flush=True)
        engine.model.transformer.register_forward_hook(tick)
        content = [{"type": "text", "text": args.instruction}]
        if args.audio:
            content.append({"type": "audio", "audio": str(args.audio.resolve())})
        wav, sr = stage("generate", lambda: engine.generate(
            [{"role": "user", "content": content}], gen_seconds=args.seconds, seed=args.seed))
        audio = wav.detach().cpu().numpy()
        if audio.ndim != 2 or not audio.shape[1] or not np.isfinite(audio).all():
            raise RuntimeError("Expected finite, nonempty channel-first audio")
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < 1e-6:
            raise RuntimeError("Output is effectively silent")
        sf.write(args.output / "audio.wav", audio.T, sr, subtype="FLOAT")
        reloaded, actual_sr = sf.read(args.output / "audio.wav", always_2d=True)
        if actual_sr != sr or not np.isfinite(reloaded).all() or len(reloaded) != audio.shape[1]:
            raise RuntimeError("WAV round-trip failed")
        report.update(status="passed", stage="complete", sample_rate=sr,
                      channels=audio.shape[0], audio_seconds=audio.shape[1] / sr,
                      rms_amplitude=rms, peak_amplitude=float(np.max(np.abs(audio))),
                      clipped_fraction=float(np.mean(np.abs(audio) >= 1)))
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
