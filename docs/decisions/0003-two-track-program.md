# 0003 — Two-track program with explicit audio timing

Status: accepted

The editor has one append-only visual track and one audio track. Audio generation
targets the audio track instead of creating placeholder visual segments.

Visual time is derived from ordered items. Audio items store absolute start
times from the first schema version, even though the initial UI only appends at
the audio end. Removing audio preserves later start times and therefore leaves
silence rather than changing synchronization implicitly.

The visual track defines export duration. Trailing audio is a visible validation
condition. Native audio from video remains attached to its visual item and can be
muted independently.

