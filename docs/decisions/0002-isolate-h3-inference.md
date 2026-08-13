# 0002 — Isolate H3 inference in a helper process

Status: accepted

The app communicates with H3 through a versioned JSON-lines protocol. The helper
owns `h3_ctx`, Metal allocations, shader and model paths, progress callbacks,
preview frames, cancellation, and FFmpeg processes.

The process boundary provides crash recovery and a deterministic way to release
all inference memory. It also prevents H3 and FFmpeg concerns from leaking into
the application domain.

