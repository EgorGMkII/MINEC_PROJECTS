# UI

This document describes current web UI behavior. Keep implementation-specific UI
details here, not in `AGENTS.md`.

## Apps

- `rag_web.py` is the internal/developer-facing web app.
- `rag_web_public.py` is the public wrapper around the same FastAPI app with
  public mode enabled.

## Current behavior

The page allows users to upload:

- project changes, multiple files;
- regional/base program, one file;
- federal program, optional;
- supporting/extra documents, multiple files.

The multi-file project and extra slots accumulate files when the user uploads
them in several batches. Those slots also support removing individual files and
clearing the slot.

Preview behavior:

- DOCX preview uses Mammoth to render HTML.
- PDF, TXT, and XLSX preview uses parsed plain text.

The result area shows:

- model answer;
- token usage when available;
- DOCX download action;
- retrieval debug details when available.

## Mode-specific UI

The public wrapper must not expose internal/developer controls. The internal app
may have different diagnostics or controls, but mode-specific UI should be
documented here when it changes.
