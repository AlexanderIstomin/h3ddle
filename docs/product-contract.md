# Product contract

## Editor

- The header shows the H3ddle title on the left and the model selector plus
  Export on the right.
- One visual lane and one audio lane share one ruler and playhead.
- Visual assets support video and images. Audio assets support generated or
  imported audio.
- Both lanes begin without move or reorder interactions. Selected video,
  image, and audio clips expose left and right trim handles. Images can
  extend freely; video and audio stay inside the source. A leading trim keeps
  the out point. Audio start times stay absolute — later audio does not move.
- Insert is append-after-last only: a visual result goes on the visual end, an
  audio result on the current audio end. There is no insert-at-playhead or
  replace mode.
- Native stills are the last frame of a 22-frame H3 chunk, not a separate
  image model. Display duration on the timeline is independent of that chunk.
- Native audio is the soundtrack of a 32×32 joint H3 clip, not a separate
  audio model. Duration follows the same 22+17n frame shapes as video.
- Disabling keeps an item recoverable. Removing audio does not shift later audio.
- Each visual video can include or mute its native soundtrack.
- The program canvas shows the composed visual at the playhead, letterboxed to
  the project aspect on the project background. The viewer can pan and zoom
  that frame; clip transforms are out of scope.
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

- The export surface will reproduce the approved visible behavior independently
  using AVFoundation and VideoToolbox.
- The visual duration is authoritative.
- Trailing audio requires an explicit warning rather than silent truncation.

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
