# Product contract

## Editor

- The header shows the H3ddle title on the left and the model selector plus
  Export on the right.
- One text lane (T1), one visual lane (V1), and one audio lane (A1) share one
  ruler and playhead. T1 sits above V1. Titles store explicit start times, may
  overlap, and later titles draw on top.
- Visual assets support generated or imported video and images. Audio assets
  support generated or imported audio. Import copies the file into an
  app-managed media folder and appends after the last clip on that lane.
  Finder files can be dropped onto the matching lane or its header.
  Disabling the audio track omits A1 and native clip soundtracks from export.
- Selected video, image, and audio clips expose left and right trim handles.
  Images can extend freely; video and audio stay inside the source. A leading
  trim keeps the out point. Dragging a visual clip reorders the lane; dragging
  an audio clip slides it inside neighboring bounds or, if dropped across
  another clip, reorders and packs from the lane's leftmost start. Duplicate
  (clip menu on video or image, including right-click on the monitor, Cmd-D)
  inserts a copy after the source. Audio start times stay
  absolute — later audio does not move on trim or delete. Split at playhead
  (S) divides the selected clip when the playhead is strictly inside it; the
  left half keeps the original clip, the right half continues the source.
  Delete / Backspace removes the selection. Visual delete closes the gap;
  audio delete leaves later clips where they are.
- Insert of generated or imported media is append-after-last only: a visual
  goes on the visual end, audio on the current audio end. There is no
  insert-at-playhead, replace mode, or media library for media. Imported stills
  hold for 3 seconds. Text is the exception: T1 `+`, ⌘T, or an empty-canvas /
  empty-T1 “Add text” context menu inserts a 5-second “Text” title at the
  playhead and opens the 320 px Text inspector. Selecting a title does not
  auto-open the panel; double-click on a T1 clip does.
- Native stills are the last frame of a 22-frame H3 chunk, not a separate
  image model. Display duration on the timeline is independent of that chunk.
- Native audio is the soundtrack of a 32×32 joint H3 clip, not a separate
  audio model. Duration follows the same 22+17n frame shapes as video.
- Disabling keeps an item recoverable. Removing audio does not shift later audio.
- Each visual video can include or mute its native soundtrack.
- The program canvas shows the composed visual and any T1 titles at the
  playhead, fitted to the project aspect on the project background. A selected
  visual or title can be moved, scaled (corner handles; Command scales about
  center), and rotated (top-middle disc or the outer corner; the pointer shows
  scale vs rotate; Shift snaps to 15°) on the monitor. Unselected
  titles hit visible glyphs only; a selected title uses its expanded bounds.
  Empty-canvas drag pans the viewer; Option-drag force-pans. Empty-canvas
  right-click offers Add text. Fit and cover remain snap-to-frame presets for
  visuals; Reset transform zeros translation and scale and keeps rotation. Adjacent visual
  cuts can dissolve, fade, or wipe. Applying a transition overlaps the incoming clip over the outgoing
  tail by the transition duration and shortens the program by that amount.
  The cut + opens the Transitions panel (same 320px slot as Effects);
  clicking an existing transition opens its settings. Visual clips can carry
  a filter stack (grade, vignette, grain, sharpen, blur, bloom, chroma key)
  shown as pills on one FX lane; + opens the Effects panel. The viewer can
  pan and zoom the frame.
- Project settings cover platform presets, custom aspect/resolution/frame rate,
  background, stored tone mapping and exposure, and master gain. Live AgX/ACES
  preview, live peak meters, and LUFS normalization are out of scope.

## Generation Studio

- The initiating lane determines the generation kind. Visual offers video and
  image. Audio offers audio.
- The studio is a 1B overlay split in half: a full-height prompt on the left
  and the remaining settings plus Generate on the right. Image jobs have no
  duration slider; inserted stills hold for 3 seconds. Generate replaces that
  composer with a full-width result (progress, then the finished media).
  Completed results show the wall-clock generate time.
- Visual jobs can attach a start and/or end frame, or up to 12 ordered
  reference stills. The two modes are mutually exclusive. Live native
  conditioning needs a released FL2VA folder (frames) or Ref2VA (references);
  the optimized INT8 package stays prompt-only.
- Native generation settings sit below the model selector: named presets
  (Preview / Standard / High) plus Custom, a resolution picker (256, 512, and
  the two native H3 canvases), denoising, transformer blocks, core reuse, and
  seed. Editing any preset-owned knob selects Custom and keeps that snapshot
  when the studio closes.
- Generation is cancellable and reports phase, progress, and elapsed wall time.
- Completing a job does not append. Insert to timeline registers the asset and
  appends it after the last item on its lane.
- Optimized native video generation uses the 256×256 four-pass preview preset
  by default. Long duration requests are not exposed in this overlay slice.
- The helper loads the model on first use and keeps it until the app closes,
  the model changes, or idle/memory-pressure eviction.
- Cancel asks the helper to stop; if it does not acknowledge promptly the
  helper is killed so the next generate is not queued behind a dying Metal job.

## Export

- Export opens a native sheet with quality presets (Recommended, High, Smaller,
  Master, Custom), a poster/progress column, and encode settings.
- Named presets seed from the project platform. Custom unlocks resolution,
  frame rate, format, H.264 profile, and AAC quality. Bitrate edits select
  Custom. Loudness and hardware acceleration stay additive.
- Formats are H.264, H.265, and ProRes. GIF and WebM are out of scope.
- Full video or a custom in/out range can be encoded. Export lasts until the
  later of the visual and, when T1 is included, the last title. A title-only
  program exports that T1 span over the project background. Trailing audio
  shows a warning and is truncated at that end. Disabling T1 omits overlays
  and restores visual-length export; without a visual that is an empty program.
- Encode uses AVFoundation and VideoToolbox on this Mac. Cancel aborts the
  writer. Completion reveals the file in Finder.
- Optional −14 LUFS normalization is an offline mix pass, not a live meter.

## Model management

- Users may continue to choose an existing local H3 folder.
- The app offers only explicitly tested, revision-pinned managed packages; it
  does not imply that an arbitrary Hugging Face repository is compatible.
- A download reports aggregate and current-file progress and can be paused and
  resumed without discarding completed data.
- Size and checksum verification complete before a staged package is installed.
- The model license is linked before download, and weights remain outside the
  app bundle, source repository, and project files.
- A downloaded package is not presented as generation-ready until the engine
  adapter supports its checkpoint layout.

## Clean-room boundary

PulpCut is a private source of visible measurements and behavior only. H3ddle
contains no copied implementation, private assets, internal identifiers, source
history, documentation, or dependency on private packages.
