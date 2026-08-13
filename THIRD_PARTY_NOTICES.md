# Third-party notices

## h3.c

H3ddle integrates with `h3.c`, Copyright (c) 2026 Salvatore Sanfilippo,
distributed under the MIT License. Its pinned source, licence, and additional
third-party notices live in `Engine/Vendor/h3.c`.

Model weights and their licence are separate from the `h3.c` software licence.
H3ddle does not commit model weights.

The optional managed model package downloads selected files from
`Comfy-Org/MiniMax-H3` at a pinned Hugging Face revision. The files remain
subject to the linked MiniMax H3 Community License Agreement and are fetched
only after the user confirms the download; H3ddle does not redistribute them.

## FFmpeg

The scaffold does not bundle FFmpeg. The H3 engine currently expects FFmpeg and
FFprobe executables. Release packaging must select a compliant build, preserve
its licence materials, and document corresponding-source obligations before any
binary distribution.
