#!/usr/bin/env python3
"""Prepare an optional local YuE2 folder for H3ddle's Audio model library.

Requires hf on PATH during development setup. Weights remain outside the
repository. Native app inference needs no Python runtime. --source reuses
the pinned files from the feasibility workspace.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

MODEL_REVISION = "29b3558dd46954a0cd9021dc76d5c91864a0f1c7"
VAE_REVISION = "9a94e1d0ea9f8087e98f77fa88df4a4068104d2a"
FILES = {
    "model": {
        "model.safetensors": "1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9",
        "config.json": "ad3477bbef890bf98ae196c1e4b44779494a6231c4ab66f32708eabadf265329",
        "qwen.tiktoken": "b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186",
        "weights_manifest.json": "2296f82c29dcb11aeefde07e216403a32fc86c8f6da2ec15a70102abfd936a93",
    },
    "vae": {
        "model.safetensors": "807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346",
        "config.json": "f0191bb9694009956de44e0c361a6f1334760be4c8f848e599bde242a54a0970",
        "weights_manifest.json": "017d64a4d288217a43fcac3866a0e1d55f796fa891d4dfb6de0bd170ab750026",
    },
}
for entries in FILES.values():
    entries.update({
        "LICENSE": "060985741d20e70613b4c189c7de106cabd3fb2109fbbfb6d705c9d619417dd0",
        "THIRD_PARTY_NOTICES.md": "14d3fd9f6fee86b4260b69b0979735b99ffec8c6b4db254567047a4215cd9af3",
    })


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    if destination == repository or repository in destination.parents:
        parser.error("Model weights must stay outside the repository")
    for tool in ("hf",):
        if shutil.which(tool) is None:
            parser.error(f"Install {tool} before running setup")
    destination.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, HF_HOME=str(destination / ".cache/hf"))

    def run(*command):
        subprocess.run(command, env=environment, check=True)

    def reuse(source, target, expected):
        if target.is_file() and sha256(target) == expected:
            return True
        if source is not None and source.is_file():
            if sha256(source) != expected:
                raise ValueError(f"Pinned file checksum mismatch: {source.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            return True
        return False

    for name, revision, repo in (("model", MODEL_REVISION, "m-a-p/YuE2-3B"),
                                 ("vae", VAE_REVISION, "m-a-p/YuE2-Vae")):
        target = destination / name
        target.mkdir(exist_ok=True)
        for filename, expected in FILES[name].items():
            source = args.source / name / filename if args.source else None
            if not reuse(source, target / filename, expected):
                run("hf", "download", repo, filename, "--revision", revision,
                    "--local-dir", str(target))
            if sha256(target / filename) != expected:
                raise ValueError(f"Pinned file checksum mismatch: {filename}")
        # Preserve all additional upstream license texts alongside the notices.
        run("hf", "download", repo, "--revision", revision,
            "--include", "licenses/*", "--local-dir", str(target))

    marker = {
        "schemaVersion": 1, "model": "m-a-p/YuE2-3B", "modelRevision": MODEL_REVISION,
        "vaeRevision": VAE_REVISION, "nativeEngine": "yue2-metal-v1", "license": "CC BY-NC 4.0",
    }
    (destination / "h3ddle-yue2.json").write_text(json.dumps(marker, indent=2) + "\n")
    print(f"Ready: {destination}\nIn H3ddle, open Models → Audio → Add Folder and select this folder.")


if __name__ == "__main__":
    main()
