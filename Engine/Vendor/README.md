# Vendored engine dependencies

`h3.c` is added here as a pinned Git submodule. Model weights are never vendored
or committed. The app-facing service links the public `libh3.a` API rather than
parsing the interactive CLI output.

