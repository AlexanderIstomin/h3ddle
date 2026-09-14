#!/usr/bin/env python3
"""H3ddle protocol 19 worker for the pinned optional YuE2 local package.

The UI remains native. This worker owns all Python/model state and never
downloads or installs anything during generation. stdout is JSONL only.
"""

import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
from urllib.parse import unquote, urlparse
import uuid

PROTOCOL_VERSION = 19
STEPS = (8, 16, 32)


def file_path(value):
    url = urlparse(value)
    if url.scheme != "file" or url.netloc not in ("", "localhost") or url.query or url.fragment:
        raise ValueError("Expected a local file URL")
    path = Path(unquote(url.path))
    if not path.is_absolute():
        raise ValueError("Expected an absolute local path")
    return path


def validate_request(command, package):
    if command.get("protocolVersion") != PROTOCOL_VERSION:
        raise ValueError("Incompatible YuE2 engine protocol; update the app")
    uuid.UUID(command["requestID"])
    uuid.UUID(command["jobID"])
    request = command["generation"]
    if request.get("kind") != "music":
        raise ValueError("This worker generates YuE2 music only")
    if file_path(request["modelDirectory"]).resolve() != package.resolve():
        raise ValueError("The request selected a different YuE2 package")
    steps = request.get("denoisingSteps", 32)
    if type(steps) is not int or steps not in STEPS:
        raise ValueError("YuE2 synthesis steps must be 8, 16, or 32")
    duration = request.get("duration")
    if type(duration) not in (int, float) or not math.isfinite(duration) or not 1 <= duration <= 30:
        raise ValueError("This local YuE2 release supports a maximum length of 1–30 seconds")
    prompt = request.get("prompt", "")
    lyrics = (request.get("music") or {}).get("lyrics", "")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
        raise ValueError("Provide a music style prompt of at most 4000 characters")
    if not isinstance(lyrics, str) or not lyrics.strip() or len(lyrics) > 5000:
        raise ValueError("Provide lyrics of at most 5000 characters")
    seed = request.get("seed", 831001)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Invalid random seed")
    if request.get("firstFrameURL") or request.get("lastFrameURL") or request.get("referenceImageURLs"):
        raise ValueError("YuE2 does not accept image references")
    if request.get("checkpoint") or request.get("blockCache") or request.get("useBetaSchedule"):
        raise ValueError("H3 sampling options do not apply to YuE2")
    output = file_path(request["outputURL"])
    if output.suffix.lower() != ".wav" or output.exists():
        raise ValueError("Choose a new WAV output path")
    return dict(prompt=prompt.strip(), lyrics=lyrics.strip(), steps=steps,
                duration=duration, seed=seed % (2**63), output=output)


class YuE2Backend:
    def __init__(self, package):
        self.package = package
        self.pipe = None

    def close(self):
        if self.pipe is not None:
            self.pipe.close()
            self.pipe = None
            import torch
            torch.mps.empty_cache()

    def generate(self, request, cancelled, progress):
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HOME"] = str(self.package / ".cache")
        import importlib.metadata
        import numpy as np
        import soundfile as sf
        import torch
        from yue2 import YuE2Pipeline
        from yue2.protocol import GenerationConfig

        if importlib.metadata.version("yue2-infer") != "0.1.5":
            raise ValueError("Set up this package with yue2-infer 0.1.5")
        if not torch.backends.mps.is_available():
            raise ValueError("YuE2 requires Apple Silicon with Metal GPU access")
        progress("Loading YuE2", 0.01)
        if self.pipe is None:
            self.pipe = YuE2Pipeline.from_pretrained(
                str(self.package / "model"), vae=str(self.package / "vae"),
                local_files_only=True, device="mps", backend="torch-eager", progress=False,
            )
        self.pipe.generation_config = GenerationConfig(ode_steps=request["steps"])
        if cancelled():
            raise InterruptedError("Cancelled")
        max_tokens = int(math.ceil(request["duration"] * 25))
        counts = {}

        def token(phase, value):
            counts[phase] = counts.get(phase, 0) + 1
            if counts[phase] % 10 == 0:
                if phase == "abc":
                    progress("Planning melody and chords", 0.05 + 0.15 * min(counts[phase] / 4096, 1))
                else:
                    progress("Generating music", 0.2 + 0.25 * min(counts[phase] / max_tokens, 1))

        progress("Planning melody and chords", 0.05)
        plan = self.pipe.plan(style=request["prompt"], lyrics=request["lyrics"],
                              cot="full", seed=request["seed"], cancelled=cancelled, on_token=token)
        progress("Generating music", 0.2)
        semantic = self.pipe.generate_semantic(
            plan, sampling={"max_tokens": max_tokens, "min_tokens": min(200, max_tokens)},
            cancelled=cancelled, on_token=token,
        )
        progress("Synthesizing audio", 0.45)
        from yue2.nar import synthesize
        model = self.pipe._load_model(for_nar=True)
        latents = synthesize(
            model, plan.prefix, semantic.tokens, request["seed"], steps=request["steps"],
            cancelled=cancelled,
            on_progress=lambda done, total: progress("Synthesizing audio", 0.45 + 0.45 * done / total),
        ).numpy()
        if cancelled():
            raise InterruptedError("Cancelled")
        progress("Decoding audio", 0.9)
        audio = self.pipe.decode(latents)
        if cancelled():
            raise InterruptedError("Cancelled")
        audio = audio[:int(request["duration"] * 48000)]
        if audio.ndim != 2 or audio.shape[1] != 2 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError("YuE2 produced invalid audio")
        output = request["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".partial.wav")
        try:
            sf.write(temporary, audio, 48000, subtype="PCM_24")
            if cancelled():
                raise InterruptedError("Cancelled")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return len(audio) / 48000


class Server:
    def __init__(self, package, backend=None, output=None):
        self.package = package
        self.backend = backend if backend is not None else YuE2Backend(package)
        self.output = output if output is not None else sys.stdout
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.active = None
        self.cancelled = threading.Event()

    def emit(self, command, kind, **fields):
        event = dict(protocolVersion=PROTOCOL_VERSION, requestID=command["requestID"], kind=kind)
        if command.get("jobID"):
            event["jobID"] = command["jobID"]
        event.update(fields)
        with self.output_lock:
            self.output.write(json.dumps(event, allow_nan=False) + "\n")
            self.output.flush()

    def submit(self, command):
        if command.get("protocolVersion") != PROTOCOL_VERSION:
            self.emit(command, "failed", message="Incompatible YuE2 engine protocol")
            return
        kind = command.get("kind")
        if kind == "handshake":
            self.emit(command, "ready", capabilities=dict(
                engineName="H3ddle YuE2", engineVersion="0.1.5",
                features=["musicGeneration", "standaloneAudioGeneration", "cancellation"]))
        elif kind == "cancel":
            with self.lock:
                if self.active == command.get("jobID"):
                    self.cancelled.set()
        elif kind == "shutdown":
            self.cancelled.set()
            self.commands.put(None)
        elif kind == "generate":
            try:
                request = validate_request(command, self.package)
            except (ValueError, KeyError, TypeError) as error:
                self.emit(command, "failed", message=str(error))
                return
            with self.lock:
                if self.active is not None:
                    self.emit(command, "failed", message="YuE2 is already generating")
                    return
                self.active = command["jobID"]
                self.cancelled.clear()
                self.commands.put((command, request))
        else:
            self.emit(command, "failed", message="Unsupported YuE2 command")

    def run(self, input_stream):
        def read():
            try:
                for line in input_stream:
                    try:
                        command = json.loads(line)
                        uuid.UUID(command["requestID"])
                        self.submit(command)
                    except (ValueError, KeyError, TypeError) as error:
                        print(f"Invalid engine command: {error}", file=sys.stderr)
            finally:
                self.cancelled.set()
                self.commands.put(None)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        try:
            while True:
                try:
                    item = self.commands.get(timeout=120)
                except queue.Empty:
                    self.backend.close()
                    continue
                if item is None:
                    return
                command, request = item
                self.emit(command, "accepted")
                try:
                    # Preserve the JSON wire even if a dependency prints to stdout.
                    with redirect_stdout(sys.stderr):
                        duration = self.backend.generate(
                            request, self.cancelled.is_set,
                            lambda phase, fraction: self.emit(
                                command, "progress", phase=phase, fractionComplete=fraction),
                        )
                    if self.cancelled.is_set():
                        request["output"].unlink(missing_ok=True)
                        raise InterruptedError("Cancelled")
                    terminal = ("completed", dict(outputURL=request["output"].as_uri(), outputDuration=duration))
                except InterruptedError:
                    terminal = ("cancelled", dict(message="Cancelled"))
                except Exception as error:
                    terminal = ("failed", dict(message=str(error)))
                    self.backend.close()
                finally:
                    with self.lock:
                        self.active = None
                self.emit(command, terminal[0], **terminal[1])
        finally:
            self.backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    Server(args.package.resolve()).run(sys.stdin)
