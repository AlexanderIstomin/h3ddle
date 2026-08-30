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

H3ddle can locally convert the Dense FastH3 preview checkpoint from
`FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree`, pinned by the
conversion instructions to revision
`f624f08c6c279ab43534c003e556fc5b295b6558`. Those model weights remain subject
to the MiniMax H3 Community License Agreement and are not committed or
redistributed by H3ddle. The public FastVideo implementation used to document
the checkpoint format and serving schedule is Apache-2.0 licensed; no
FastVideo source code is vendored in this repository.

H3ddle can also convert the learned VSA checkpoint from
`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`, pinned to revision
`b65818d41939b5085451074fe8ca8b799f8d4921`. With written permission from the
MiniMax/Hailuo team, the converted learned-VSA transformer is distributed from
`PulpCut/FastH3-VSA-INT8-ConvRot`; it remains subject to the MiniMax H3
Community License Agreement included with that repository. The native tile
geometry, selection, and compression semantics are adapted from FastVideo's
Apache-2.0-licensed `video_sparse_attn_h3.py` implementation at commit
`b2db0c0a137e610fa2406d942a3c32c0179f047c`.

## TAEF1

The optional managed Z-Image package downloads the TAEF1 preview-decoder
weights from `madebyollin/taef1` at the pinned revision
`b1b2d00e9e440cfbf3dedb34266864da86016ceb`. TAEF1 is Copyright (c) 2023
Ollin Boer Bohan and distributed under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## FFmpeg

The scaffold does not bundle FFmpeg. The H3 engine currently expects FFmpeg and
FFprobe executables. Release packaging must select a compliant build, preserve
its licence materials, and document corresponding-source obligations before any
binary distribution.
