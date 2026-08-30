# 0001 — Native macOS clean-room implementation

Status: accepted

H3ddle uses SwiftUI with narrow AppKit and AVFoundation adapters. It is a new
public implementation and does not reuse PulpCut source code or packages.

This keeps the application independently buildable, native to macOS, and safe to
publish. Visible product behavior may be reproduced from approved specifications.

ADR 0007 records one narrow exception: H3ddle publishes its own editorial
interchange JSON so a later API can accept H3ddle projects. That format is
specified here, not imported from a private package.

