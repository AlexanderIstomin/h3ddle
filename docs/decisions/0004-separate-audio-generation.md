# 0004: Keep audio generation provider-neutral

## Status

Accepted for the initial scaffold.

## Context

The editor has a dedicated audio lane and presents audio as a generation kind.
The reviewed `h3.c` API generates video and 32 kHz stereo audio together. It
does not currently expose a standalone audio-only generation operation;
standalone audio inputs are references for audiovisual generation, not an
audio-only output mode.

## Decision

Audio remains a first-class `GenerationKind` at the application boundary, but
the UI and project model do not assume that MiniMax H3 itself produces the
standalone file. A production provider must report its capabilities and may
implement audio generation with a dedicated local model, or explicitly derive
an audio asset from an audiovisual H3 result. The H3 engine adapter must not
advertise audio-only generation until that behavior is implemented and tested.

## Consequences

- A dedicated audio provider adds model download, memory, attribution, and
  lifecycle work.
- Deriving audio from a full H3 render wastes video inference and can make audio
  iteration slow.
- Native sound on visual clips and the dedicated audio lane can overlap, so each
  visual clip needs an independent native-audio control.
- Explicit audio start times keep synchronization stable, but removals leave
  gaps and the first editor slice has no ripple edit or overlap tools.
- Visual duration remains authoritative; audio beyond it needs a visible export
  warning and a deliberate trim or duration decision.

