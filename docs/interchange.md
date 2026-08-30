# Editorial interchange document

H3ddle persists this JSON document, not `H3ddleProject` directly. The edit
model stays a 3-lane program; this file is the portable composition.

## Document

| Key | Type | Notes |
|---|---|---|
| `schemaVersion` | int | Interchange version. Current: `1`. Newer versions are rejected. |
| `id` | string | Project UUID, lowercase. |
| `revision` | int | Monotonic save counter. |
| `name` | string | Display name. |
| `settings` | object | Canvas and output. |
| `assets` | array | Media pool. Unused assets stay here. |
| `sequences` | array | One sequence named `Main`. |
| `compositions` | array | One composition per clip (layer stack). |

Unknown top-level keys are preserved.

## Settings

`width`, `height`, `fps`, `colorSpace` (`srgb`), `alphaMode` (`premultiplied`),
optional `backgroundColor` (hex or `transparent`), `masterGain`, `toneMapping`
(`none` / `agx` / `aces-filmic` / `neutral`), `exposure`, `target` (platform
id).

## Assets

`id`, `kind` (`video` / `image` / `audio`), `src` (locator, never a
user-home `file://` in saved packages), optional `name`, `duration`.

## Sequence

`id`, `name`, `duration`, `tracks`, `transitions`.

Tracks: `V1` (visual program), `T1` (text, visual kind), `A1` (audio). Clips
store `timelineIn` / `timelineOut` / `sourceIn` / `playbackRate`, optional
`compositionId`, `assetId`, `enabled`, `gain`, `effects`.

Transitions are sequence-level: `fromClipId`, `toClipId`, `kind`
(`cross-dissolve` / `dip-black` / `wipe`), `duration`, `easing`.

## Compositions and layers

Each visual or text clip has a composition with a planar camera matching
`settings` and one layer (`video`, `image`, or `text`). Canvas placement is
written both as a 3-vector transform and as lossless `canvasFit` /
`canvasTranslationX` / `canvasTranslationY` / `canvasScale` /
`canvasRotationRadians` fields.

Text layers carry a flat text engine (`value`, face, fill/stroke/shadow,
layout). H3ddle does not require third-party font revision hashes.

## Effects

Clip `effects` use catalog ids: `color.grade`, `stylize.vignette`,
`stylize.filmGrain`, `stylize.sharpen`, `blur.gaussian`, `stylize.bloom`,
`key.chroma`. Parameter keys and ranges are H3ddle's. A later adapter may
rescale to a remote catalog.

## Mapping from the 3-lane program

- V1 order + `gapBefore` + incoming transition overlap → clip `timelineIn`.
- Load reconstructs `gapBefore` from those times.
- T1 titles → clips on the `T1` track with a text composition.
- A1 items → audio clips with explicit in/out.
- Video `includesNativeAudio == false` → clip `gain` 0.
- Disabled items → `enabled: false`.
