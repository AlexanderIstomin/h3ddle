#!/usr/bin/env python3
"""Compare AuK residency, conditioning reuse, BF16, and Base solver counts.

Uses the pinned, external environment from the AuK feasibility note. All
changes are runtime experiment adapters; upstream installed files are untouched.
No downloads, app changes, automatic CPU fallback, or Flash schedule changes.
"""

import argparse
from contextlib import nullcontext
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
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
    parser.add_argument("--reference", type=Path, required=True,
                        help="Passed feasibility probe.json, providing exact task settings")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-smoke", action="store_true",
                        help="Test a BF16-Qwen loader and real uncached requests instead of the stage sweep")
    parser.add_argument("--dit-precision", choices=("fp32", "bf16", "fp16"), default="fp32",
                        help="Denoiser storage/compute precision in compact-smoke mode")
    parser.add_argument("--base-steps", choices=(8, 16, 32), type=int, default=32,
                        help="Base sampling steps in compact-smoke mode; Flash stays at four")
    parser.add_argument("--comparison", type=Path,
                        help="Trial prefix (.wav/.npy omitted) from the FP32 sweep")
    parser.add_argument("--comparison-wav", type=Path,
                        help="Audio-only reference when no saved FP32 latents exist")
    parser.add_argument("--stages-only", action="store_true",
                        help="Prepare fixed inputs and run stage tests; reuse an existing --comparison")
    args = parser.parse_args()
    ref = json.loads(args.reference.read_text())
    if ref["status"] != "passed" or bool(args.audio) != ref["reference_audio"]:
        parser.error("Use a passed probe and provide its source audio when required")
    if args.stages_only and (args.compact_smoke or not args.comparison):
        parser.error("stages-only needs comparison and cannot be combined with compact-smoke")
    if args.comparison and args.comparison_wav:
        parser.error("Choose comparison or comparison-wav")
    if not args.compact_smoke and args.base_steps != 32:
        parser.error("base-steps is only for compact-smoke")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", stage="verify", trials=[], conditioning_trials=[],
                  reference=ref, python=platform.python_version(),
                  precision_changes_are_experimental=True, cpu_fallback=False,
                  compact_loader=args.compact_smoke,
                  compact_dit_precision=args.dit_precision if args.compact_smoke else None,
                  stages_only=args.stages_only,
                  environment={key: os.environ.get(key) for key in (
                      "PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO",
                      "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL")})

    def save():
        temp = args.output / "benchmark.json.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(args.output / "benchmark.json")

    try:
        import numpy as np
        import soundfile as sf
        import torch
        from auk.infer.infer_auk import AukInfer

        def cast_weights(module, dtype):
            if all(parameter.dtype == dtype for parameter in module.parameters()):
                return
            device = next(module.parameters()).device
            # Conversion briefly holds old and new storage. Stage on CPU to
            # keep that transient allocation outside the Metal memory cap.
            if device.type == "mps":
                module.to("cpu")
                torch.mps.synchronize()
                gc.collect()
                torch.mps.empty_cache()
            # Rotary frequencies and other floating buffers are not weights.
            # Preserve their original values/dtypes through every A/B cast.
            buffers = [(child, key, value) for child in module.modules()
                       for key, value in child._buffers.items() if value is not None]
            module.to(dtype=dtype)
            for child, key, value in buffers:
                child._buffers[key] = value
            module.to(device=device)

        if not torch.backends.mps.is_available():
            raise RuntimeError("Real MPS access is required")
        report["dependencies"] = {name: importlib.metadata.version(name) for name in (
            "auk", "torch", "torchaudio", "transformers", "torchdiffeq", "x-transformers")}
        if report["dependencies"]["torch"] != "2.7.1":
            raise RuntimeError("This benchmark targets the pinned torch 2.7.1 environment")
        # Compare local weights with the pinned Hub manifests from feasibility.
        verified = []
        for folder, label in ((args.model, ref["variant"]), (args.qwen, "qwen")):
            manifest = json.loads((folder.parent / f"{label}-manifest.json").read_text())
            for entry in manifest["files"]:
                path = folder / entry["path"]
                if not entry["sha256"] or not path.exists():
                    continue
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != entry["sha256"]:
                    raise RuntimeError(f"Checkpoint checksum mismatch: {path.name}")
                verified.append(dict(repo=manifest["repo"], revision=manifest["revision"],
                                     file=entry["path"], sha256=digest))
        report["verified_weights"] = verified
        if args.audio:
            report["source_audio_sha256"] = hashlib.sha256(args.audio.read_bytes()).hexdigest()
        checkpoint = args.model / ("auk_flash.safetensors" if ref["variant"] == "flash" else "auk_base.safetensors")
        report["stage"] = "load"
        save()
        begin = time.perf_counter()
        from auk.model import CFMEdit
        original_to = CFMEdit.to
        own_to = CFMEdit.__dict__.get("to")
        def compact_to(model, *a, **kw):
            if a == (torch.float32,) and not kw:
                # Intercept only AuK's initialization-time whole-model FP32
                # expansion. Qwen is already loaded in BF16; VAE is separate.
                cast_weights(model.transformer, {"fp32": torch.float32,
                             "bf16": torch.bfloat16, "fp16": torch.float16}[args.dit_precision])
                return model
            return original_to(model, *a, **kw)
        try:
            if args.compact_smoke:
                CFMEdit.to = compact_to
            engine = AukInfer(str(args.model / "config.yaml"), str(checkpoint),
                              qwen_path=str(args.qwen), device="mps")
        finally:
            if args.compact_smoke:
                if own_to is None:
                    del CFMEdit.to
                else:
                    CFMEdit.to = own_to
        torch.mps.synchronize()
        report["load_seconds"] = time.perf_counter() - begin
        report["loaded_components"] = {
            name: dict(parameter_bytes=sum(p.numel() * p.element_size() for p in module.parameters()),
                       parameter_dtypes=sorted({str(p.dtype) for p in module.parameters()}))
            for name, module in (("qwen", engine.model.text_encoder),
                                 ("dit", engine.model.transformer), ("vae", engine.vae_model))}
        if engine.is_flash != (ref["variant"] == "flash"):
            raise RuntimeError("Variant differs from reference")
        default_steps = 4 if engine.is_flash else args.base_steps
        precision = {"dit": args.dit_precision if args.compact_smoke else "fp32"}
        cached = {}
        use_cache = [False]
        phase_times = {}
        latent_result = [None]
        raw_encode = engine.model.encode_text
        def original_encode(*a, **kw):
            context = torch.autocast("mps", dtype=torch.bfloat16) if args.compact_smoke else nullcontext()
            with context:
                return raw_encode(*a, **kw)
        original_prepare = engine.model.build_cond_inputs
        original_reference = engine.vae_model.encoding_and_normalization
        original_sample = engine.model.sample
        original_decode = engine.vae_model.inference_from_latents
        original_forward = engine.model.transformer.forward

        def timed(key, fn):
            torch.mps.synchronize()
            tick = time.perf_counter()
            value = fn()
            torch.mps.synchronize()
            phase_times[key] = phase_times.get(key, 0) + time.perf_counter() - tick
            return value

        def cache_call(key, original, a, kw):
            if use_cache[0]:
                return cached[key]
            value = timed(key, lambda: original(*a, **kw))
            cached[key] = value
            return value

        engine.model.build_cond_inputs = lambda *a, **kw: cache_call("preprocessing", original_prepare, a, kw)
        engine.model.encode_text = lambda *a, **kw: cache_call("conditioning", original_encode, a, kw)
        engine.vae_model.encoding_and_normalization = lambda *a, **kw: cache_call("reference_encode", original_reference, a, kw)

        def sample(*a, **kw):
            value = timed("sample_including_conditioning", lambda: original_sample(*a, **kw))
            source_frames = int(kw["lens"][0])
            latent_result[0] = value[0][:, source_frames:, :].detach().cpu().float().numpy()
            return value

        def forward(*a, **kw):
            dtype = torch.float16 if precision["dit"] == "fp16" else torch.bfloat16
            context = torch.autocast("mps", dtype=dtype) if precision["dit"] != "fp32" else nullcontext()
            with context:
                # Keep the ODE state/integration in FP32 even for BF16 matmuls.
                return original_forward(*a, **kw).float()

        engine.model.sample = sample
        engine.model.transformer.forward = forward
        engine.vae_model.inference_from_latents = lambda *a, **kw: timed("decode", lambda: original_decode(*a, **kw))

        def memory():
            return dict(allocated_bytes=torch.mps.current_allocated_memory(),
                        driver_bytes=torch.mps.driver_allocated_memory())

        def compare(candidate, reference):
            a, b = candidate.astype(np.float64), reference.astype(np.float64)
            if a.shape != b.shape or not np.isfinite(a).all():
                raise RuntimeError("Invalid output shape or nonfinite values")
            rms = float(np.sqrt(np.mean(b * b)))
            return dict(identical=bool(np.array_equal(candidate, reference)),
                        relative_rmse=float(np.sqrt(np.mean((a - b) ** 2))) / max(rms, 1e-12),
                        correlation=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))

        baseline_audio = [None]
        baseline_latent = [None]
        if args.comparison:
            baseline_audio[0], comparison_sr = sf.read(str(args.comparison) + ".wav", always_2d=True)
            if comparison_sr != engine.target_sample_rate:
                raise RuntimeError("Comparison sample rate differs")
            baseline_latent[0] = np.load(str(args.comparison) + ".npy", allow_pickle=False)
        elif args.comparison_wav:
            baseline_audio[0], comparison_sr = sf.read(args.comparison_wav, always_2d=True)
            if comparison_sr != engine.target_sample_rate:
                raise RuntimeError("Comparison sample rate differs")

        def run(name, steps=default_steps, record=True):
            report["stage"] = name
            save()
            print(f"Starting {name}", flush=True)
            phase_times.clear()
            torch.manual_seed(ref["seed"])
            content = [{"type": "text", "text": ref["instruction"]}]
            if args.audio:
                content.append({"type": "audio", "audio": str(args.audio.resolve())})
            torch.mps.synchronize()
            tick = time.perf_counter()
            wav, sr = engine.generate([{"role": "user", "content": content}],
                                      gen_seconds=ref["target_seconds"], seed=ref["seed"], nfe=steps)
            torch.mps.synchronize()
            elapsed = time.perf_counter() - tick
            audio = wav.cpu().numpy().T
            if not np.isfinite(audio).all() or np.sqrt(np.mean(audio ** 2)) < 1e-6:
                raise RuntimeError("Invalid or silent audio")
            if not record:
                return
            if baseline_audio[0] is None:
                baseline_audio[0] = audio.copy()
                baseline_latent[0] = latent_result[0].copy()
            item = dict(name=name, steps=steps, dit_precision=precision["dit"],
                        cached_conditioning=use_cache[0], seconds=elapsed,
                        phases=dict(phase_times), memory=memory(),
                        audio_seconds=len(audio) / sr, peak=float(np.max(np.abs(audio))),
                        clipped_fraction=float(np.mean(np.abs(audio) >= 1)),
                        audio_comparison=compare(audio, baseline_audio[0]),
                        latent_comparison=(compare(latent_result[0], baseline_latent[0])
                                           if baseline_latent[0] is not None else None))
            sf.write(args.output / f"{name}.wav", audio, sr, subtype="FLOAT")
            readback, read_sr = sf.read(args.output / f"{name}.wav", always_2d=True)
            if read_sr != sr or not np.array_equal(readback, audio):
                raise RuntimeError("Float WAV round-trip failed")
            np.save(args.output / f"{name}.npy", latent_result[0])
            report["trials"].append(item)
            save()
            print(json.dumps(item), flush=True)

        with torch.inference_mode():
            if args.compact_smoke:
                run("compact-first")
                run("compact-resident-1")
                run("compact-resident-2")
                use_cache[0] = True
                run("compact-cached")
                report.update(status="passed", stage="complete")
                return
            if args.stages_only:
                run("prepare-caches", steps=1 if not engine.is_flash else 4, record=False)
                use_cache[0] = True
            else:
                run("warmup-full", record=False)
                run("resident-fp32-1")
                run("resident-fp32-2")
                use_cache[0] = True
                run("cached-fp32-1")
                run("cached-fp32-2")

            # Conditioner ABBA. Qwen was loaded from BF16 and expanded to FP32,
            # so its BF16 -> FP32 restoration does not lose original weights.
            full_conditioning = cached["conditioning"]
            bf16_conditioning = None
            # Keep the inactive DiT off Metal throughout both conditioning
            # variants, including dtype conversions. Transfer time is excluded.
            engine.model.transformer.to("cpu")
            gc.collect()
            torch.mps.empty_cache()
            report["conditioning_dit_device"] = "cpu"
            for index, dtype_name in enumerate(("fp32", "bf16", "bf16", "fp32")):
                report["stage"] = f"conditioning-{dtype_name}-{index + 1}"
                save()
                print("Starting " + report["stage"], flush=True)
                dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
                cast_weights(engine.model.text_encoder, dtype)
                gc.collect()
                torch.mps.empty_cache()
                def encode():
                    context = torch.autocast("mps", dtype=torch.bfloat16) if dtype_name == "bf16" else nullcontext()
                    with context:
                        return original_encode(cached["preprocessing"], "mps")
                encode()  # warm each execution dtype, outside timed region
                torch.mps.synchronize()
                tick = time.perf_counter()
                value = encode()
                torch.mps.synchronize()
                elapsed = time.perf_counter() - tick
                if dtype_name == "bf16":
                    bf16_conditioning = (value[0].float(), value[1])
                item = dict(precision=dtype_name, seconds=elapsed, memory=memory(),
                            comparison=compare(value[0].float().cpu().numpy(),
                                               full_conditioning[0].float().cpu().numpy()))
                report["conditioning_trials"].append(item)
                print("Conditioning " + json.dumps(item), flush=True)
                save()

            # Qwen is no longer needed for these fixed-input sampling tests.
            # Free it identically for both precisions to avoid memory confounds.
            del engine.model.text_encoder
            gc.collect()
            torch.mps.empty_cache()
            engine.model.transformer.to("mps")
            report["memory_after_releasing_qwen"] = memory()
            save()

            for index, dtype_name in enumerate(("fp32", "bf16", "bf16", "fp32")):
                dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
                if dtype_name != precision["dit"]:
                    cast_weights(engine.model.transformer, dtype)
                    if dtype_name == "fp32":
                        # DiT checkpoint is FP32: restore exact original values,
                        # not the rounded BF16 values expanded back to FP32.
                        engine._load_ema_weights(engine.model, str(checkpoint))
                    precision["dit"] = dtype_name
                engine.model.transformer.clear_cache()
                gc.collect()
                torch.mps.empty_cache()
                run(f"warmup-dit-{index}", steps=1 if not engine.is_flash else 4, record=False)
                run(f"dit-{dtype_name}-{index + 1}")

            if not engine.is_flash:
                run("base-fp32-16", steps=16)
                run("base-fp32-8", steps=8)

            # Combined BF16 Qwen conditioning and BF16 DiT at released steps.
            # VAE and ODE integration remain FP32.
            cast_weights(engine.model.transformer, torch.bfloat16)
            precision["dit"] = "bf16"
            cached["conditioning"] = bf16_conditioning
            torch.mps.empty_cache()
            run("warmup-combined", steps=1 if not engine.is_flash else 4, record=False)
            run("combined-bf16")
        report.update(status="passed", stage="complete")
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
