# 0006: Overlay text and canvas objects

Status: accepted

The editor remains a visual backbone with explicit audio timing (ADR 0003).
This decision adds a third lane, T1 Text, that stores document-owned 2D titles
with explicit start times. Titles overlap on one lane; later array index draws
and hits on top. They are not `AssetReference`s and they are not a media kind.

Canvas objects (visual clips and T1 titles) share one `CanvasObjectTransform`:
normalized translation, uniform scale, and clockwise radians. Visuals still
fit or cover the unrotated source, then rotate. Overlays skip fit/cover and
blit a raster whose layout lives in `ProjectSettings` pixels.

Preview duration is `max(visual, audio, text)`. Export duration is
`max(visual, text)` when T1 is included, otherwise the visual duration. That
is an intentional exception to ADR 0003 for overlay text only. Audio does not
get the same exception: trailing audio still warns and truncates at the export
end. A title-only program exports the T1 span over the project background.

Insert of generated or imported media stays append-after-last. Text inserts at
the playhead. The T1 `+` menu, ⌘T, and an empty-canvas / empty-T1 “Add text”
context menu all insert a default title and open the 320 px Text inspector.
Selecting a title does not auto-open the panel.

The selection gizmo is an outline with handles. Duplicate and enable stay on
the timeline clip menu.

Companion: `docs/designs/2026-08-15-canvas-objects-and-text-track.md`.
