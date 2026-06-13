# Pipeline

This document describes the current code path in `summary_rag.py` and
`rag_web.py`. Code is the source of truth if behavior changes.

## Upload and parsing

The web routes in `rag_web.py` accept:

- project changes: multiple files, required;
- regional/base program: one file, required;
- federal program: optional;
- supporting/extra documents: multiple files, optional.

`rag_support.parse_upload_bytes` supports `docx`, `pdf`, `txt`, and `xlsx`.
DOCX parsing uses robust OOXML text extraction first, ignores media artifacts,
and reads text from `word/document.xml`, headers, footers, and tables. Table rows
are included as plain text with cells separated by ` | `. If OOXML extraction
cannot recover text, `python-docx` is used as a fallback.

Source loaders skip individual files that cannot be read or parsed. A broken
file in a multi-file upload should not fail the whole pipeline.

## Source model

`simple_rag.SourceText` carries:

- `source`;
- `text`;
- `group`;
- optional `title`.

Important groups are `project`, `base_program`, `regional`, `federal`, `extra`,
`extra_summary`, and `npa`.

## Summary-based pipeline order

`SummaryRAGPipeline.__init__` performs the main work in this order:

1. Normalize full joined project text.
2. Run UFAS precheck over full `project_text` plus `УФАС/UFAS.txt` rules.
3. Summarize each project-source document.
4. Build one `project_package_summary` from project document summaries.
5. Summarize supporting/extra documents.
6. Build retrieval sources from raw context sources plus extra summaries.
7. Chunk regular sources and summary sources.
8. Build final RAG chain and debug chain.

## Retrieval

Retrieval query is `project_package_summary.strip()`.

Regular retrieval sources:

- NPA documents from `npa/`;
- base regional program;
- federal documents.

Extra/supporting documents are excluded as raw text and reintroduced only as
`extra_summary` sources.

`select_documents_with_coverage` ranks chunks with BM25, enforces
`max_chunks_per_source`, and then adds the best chunk from any source that was
missing after the first pass.

## Final prompt and answer

The final prompt includes:

- selected base system prompt;
- analysis task;
- `project_package_summary`;
- summaries of individual project documents;
- summaries of supporting/extra documents;
- retrieved context.

The final prompt intentionally does not include full raw project package text.

`compose_final_answer` places the UFAS precheck block before the main model
answer, separated by a visual delimiter.

## Cache behavior

Summary artifacts and UFAS precheck are cached under
`test_cases/Сх/summaries` by default. Cache keys use source text hashes and, for
UFAS, the project text plus UFAS rules.

Changing summary prompts may not invalidate old cached summaries unless
`force_resummarize=True` or cache files are handled separately.
