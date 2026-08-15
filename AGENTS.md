# H3ddle development guidance

H3ddle is a clean-room, open-source native macOS application. PulpCut may be
consulted privately for visible behavior and visual measurements, but its code,
assets, identifiers, comments, schemas, and repository history must never enter
this repository.

## Architecture

- Keep the application native: SwiftUI first, with narrow AppKit or AVFoundation
  adapters where the platform API requires them.
- Keep domain models free of SwiftUI, AppKit, and AVFoundation dependencies.
- Run local H3 inference outside the app process behind the versioned engine
  protocol.
- Keep UI development and CI independent from model weights through fake
  generation providers.
- Visual and audio tracks share one clock. Visual items are ordered; start
  times are derived from that order. Audio items store explicit start times.
  Both lanes support append, duplicate, and reorder.

## Quality

- Use Swift 6 concurrency checking. Do not bypass Sendable or actor-isolation
  errors without documenting why.
- Prefer small value types and protocols over application-wide singletons.
- Add or update tests whenever timeline, protocol, generation, or export
  behavior changes.
- Run `Scripts/ci.sh` before handing off changes.

## Public boundary

- Never commit model weights, credentials, private screenshots, absolute paths
  to private repositories, or imports from private package namespaces.
- Keep third-party code pinned and preserve its licence and notices.
- Treat generated Xcode project files as derived from `project.yml`.
