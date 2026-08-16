#!/bin/sh
set -eu

# Rejects the two things that must never enter this repository: references to
# the private PulpCut implementation, and model weights or private reference
# media.
#
# This used to drive ripgrep. A missing rg exits 127, and a command that fails
# inside an `if` condition does not trip `set -e` — so on any machine without
# ripgrep the script skipped both searches and printed that the check passed.
# A guard that cannot fail is worse than no guard, because nobody looks again.
# grep and find are always present, and each search's result is now inspected
# rather than used as a branch condition.

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
status=0

references=$(grep -rIn -E \
    '(@pulpcut/|/Users/[^/]+/WebstormProjects/pulpcut3d|PULPCUT_PRIVATE)' \
    "$repository_root" \
    --exclude-dir=.git \
    --exclude=check-public-boundary.sh || true)
if [ -n "$references" ]; then
    printf '%s\n' "$references" >&2
    echo "Private PulpCut implementation reference found." >&2
    status=1
fi

artifacts=$(find "$repository_root" -path "$repository_root/.git" -prune -o \
        -type f -print \
    | grep -i -E '\.(safetensors|ckpt)$|/(Models|LocalReferences)/' || true)
if [ -n "$artifacts" ]; then
    printf '%s\n' "$artifacts" >&2
    echo "Model weights or private reference artifacts found." >&2
    status=1
fi

[ "$status" -eq 0 ] || exit 1
echo "Public boundary check passed."
