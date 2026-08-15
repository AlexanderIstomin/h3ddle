# 0003 — Two-track program with explicit audio timing

Status: accepted

The editor has one visual track and one audio track. Audio generation
targets the audio track instead of creating placeholder visual segments.

Visual time is derived from ordered items. Those items can be appended,
duplicated, or reordered; start times stay implied by order. Audio items store
absolute start times from the first schema version. New audio still appends at
the audio end; clips can then be duplicated or dragged. Removing audio
preserves later start times and therefore leaves silence rather than changing
synchronization implicitly.

The visual track defines export duration. Trailing audio is a visible validation
condition. Native audio from video remains attached to its visual item and can be
muted independently.

