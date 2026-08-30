# 0007 — Project interchange document

Status: accepted

H3ddle edits a 3-lane `H3ddleProject`. Persistence uses a separate **editorial
interchange document**: settings, an asset pool, one sequence of typed tracks,
clips with explicit in/out, per-clip compositions, and sequence transitions.

The interchange is specified in this repository (`docs/interchange.md`) and
implemented in `H3ddleCore`. It is the subset a later studio API can accept
without a second document model. Unknown JSON keys round-trip so a richer
remote document is not stripped on save.

This is a documented exception to ADR 0001: interchange field names are
H3ddle's published format, not a source lift. The public-boundary script still
forbids private package imports and private repository paths. Implementation,
comments, and unused 3D / node / agent fields stay out of this tree.

The edit model does not become a multi-track NLE. Projection at the persist
boundary expands ordered V1 placements into explicit clip times and folds them
back on load.

`schemaVersion` on the interchange starts at 1. A private adapter may stamp a
remote schema version and object keys on push. Local files never store
absolute home-directory paths; assets use a locator string (relative media
in the package store).
