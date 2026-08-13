#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

if rg -n --hidden \
    --glob '!.git/**' \
    --glob '!Scripts/check-public-boundary.sh' \
    '(@pulpcut/|/Users/[^/]+/WebstormProjects/pulpcut3d|PULPCUT_PRIVATE)' \
    "$repository_root"; then
    echo "Private PulpCut implementation reference found." >&2
    exit 1
fi

if rg --files -uu "$repository_root" \
    | rg -i '\.(safetensors|ckpt)$|/(Models|LocalReferences)/'; then
    echo "Model weights or private reference artifacts found." >&2
    exit 1
fi

echo "Public boundary check passed."

