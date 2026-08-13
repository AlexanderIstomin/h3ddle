# Product contract

## Editor

- One visual lane and one audio lane share one ruler and playhead.
- Visual assets support video and images. Audio assets support generated or
  imported audio.
- Both lanes begin without drag or reorder interactions.
- Visual generation appends video or image media to the visual end.
- Native stills are the last frame of a 22-frame H3 chunk, not a separate
  image model. Display duration on the timeline is independent of that chunk.
- Native audio is the soundtrack of a 32×32 joint H3 clip, not a separate
  audio model. Duration follows the same 22+17n frame shapes as video.
- Audio generation appends audio to the current audio end.
- Disabling keeps an item recoverable. Removing audio does not shift later audio.
- Each visual video can include or mute its native soundtrack.

## Generation Studio

- The initiating lane determines the available generation kinds.
- Visual offers video and image. Audio offers audio.
- Generation is cancellable and reports phase, progress, and elapsed wall time.
- Optimized native video generation identifies its 256×256 four-pass preview
  preset and warns that long duration requests scale transformer work sharply.
- Denoising stills are optional and persist across launches.
- The helper loads the model on first use and keeps it until the app closes,
  the model changes, or idle/memory-pressure eviction.
- Cancel asks the helper to stop; if it does not acknowledge promptly the
  helper is killed so the next generate is not queued behind a dying Metal job.
- A successful result is registered as an asset and appended to its lane.
- The latest generated visual is playable in the editor, shows its measured
  generation duration, and can be copied out of temporary storage.

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
