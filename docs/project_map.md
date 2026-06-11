# Project map

## Purpose

The project is a FastAPI web service for self-checking draft changes to
government programs. It is meant to help a developer of changes find substantive
and methodological risks before formal coordination.

## Entry points

- `rag_web.py` - main developer FastAPI app and web UI.
- `rag_web_public.py` - public app wrapper that disables developer mode and
  reuses `rag_web.app`.
- `summary_rag.py` - current production summary-based pipeline.
- Docker entrypoint: `uvicorn rag_web:app --host 0.0.0.0 --port 8000`.

## Main files by responsibility

- Web/UI: `rag_web.py`, `rag_web_public.py`.
- Upload parsing: `rag_support.py`.
- Source collection and chunking: `simple_rag.py`.
- Main pipeline, summaries, UFAS precheck, coverage retrieval:
  `summary_rag.py`.
- Developer system prompt and parsing constants: `rag_support.py`.
- Public system prompt: `old_prompt.py`.
- Summary prompts: `summary_prompts.py`.
- UFAS prompt: `ufas_prompts.py`.
- DOCX report export: `/download-docx` route in `rag_web.py`.
- Operational notebook/tests: `tests.ipynb`.
- Hidden corpora: `npa/`, `УФАС/UFAS.txt`.

## Architectural constraints

- The current web path uses `SummaryRAGPipeline`, imported from `summary_rag.py`
  as `SimpleRAGPipeline`.
- `simple_rag.py` still contains a simple/legacy pipeline, but its active role
  in the web app is utility support for parsing, source collection, chunking,
  formatting, BM25, and LLM client creation.
- Project documents are kept both as joined `project_text` and individual
  `project_sources`.
- Base regional program, federal documents, extra documents, and NPA sources
  must stay logically separated by `SourceText.group`.
- Public mode must use the public prompt strategy documented in
  `report_contract.md`.
- Developer mode uses the internal/developer prompt strategy documented in
  `report_contract.md`.
