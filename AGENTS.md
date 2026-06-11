# Agent governance

This file defines how an agent should work in this repository. It should stay
stable and practical. Do not use it to record implementation history, temporary
decisions, or details of individual UI controls.

## Documentation boundaries

`AGENTS.md` is for:

- agent workflow and task interpretation;
- risk levels and required review depth;
- testing rules;
- architecture invariants;
- change restrictions;
- documentation update rules;
- contracts between major subsystems.

`docs/` is for:

- current project map and entrypoints;
- runtime commands and environment;
- current UI behavior;
- current pipeline behavior;
- report contract details;
- implementation notes that can change over time.

If a fact describes the current state of the system, put it in `docs/`. If a
rule describes how the agent must behave while changing the system, put it here.

## First steps

Before non-trivial changes:

1. Read the relevant docs in `docs/`.
2. Read the code for the layer being changed.
3. Identify the risk level.
4. Choose the workflow mode.
5. Keep the change scoped to the requested layer.

Code is the source of truth when documentation and implementation differ.

## Core System Architecture

Use this map as the stable starting point before changing code. More detailed
and more current implementation notes live in `docs/`.

- Current primary pipeline entrypoint: `summary_rag.py`.
- Web layer: `rag_web.py` and `rag_web_public.py` handle HTTP routes, upload
  flow, preview, result rendering, and DOCX download.
- Pipeline layer: `summary_rag.py` owns the summary-based RAG pipeline,
  precheck orchestration, summary artifacts, retrieval coverage, final chain,
  and debug payloads.
- Parsing and source layer: `rag_support.py` parses uploaded files and hidden
  corpus documents; `simple_rag.py` provides source collection, chunking,
  context formatting, BM25 helpers, and LLM client creation.
- Prompt layer: base report prompts, summary prompts, and precheck prompts are
  kept in dedicated prompt modules rather than embedded deep in web handlers.
- Retrieval layer: retrieval operates on prepared context documents with source
  metadata and should preserve source/group boundaries.
- Report generation layer: the pipeline composes the final model answer, while
  the web layer only renders it and exports DOCX.

## Workflow modes

- Explore mode: read code, docs, tests, and runtime contracts without changing
  files.
- Planning mode: prepare a scoped plan with goal, affected components, files,
  risks, and acceptance criteria.
- Patching mode: make focused changes only after enough context is available.
- Review mode: inspect the diff for architecture, behavior, prompt, retrieval,
  parsing, UI, and regression risks.
- Test mode: run the smallest meaningful checks available; explain skipped
  checks.

Recommended flow by risk:

- Low risk: Explore -> Patching -> Review.
- Medium risk: Explore -> Planning -> Patching -> Review -> Test.
- High risk: Explore -> Planning -> Review -> confirmation when needed ->
  Patching -> Test.

## Risk levels

- Low risk: documentation, comments, focused tests, small local helper cleanup,
  narrow UI wording.
- Medium risk: prompt wording, retrieval parameters, report structure, new
  validation checks, markdown/HTML rendering.
- High risk: pipeline order, parsing behavior, source grouping or data models,
  interfaces between web and pipeline, LLM client behavior, cache semantics,
  dependency changes, public/developer mode contracts.

Do not combine high-risk architecture changes with unrelated functionality.

## Architecture invariants

- Preserve the separation between web/UI, parsing, retrieval, summarization,
  prompts, report generation, and runtime/deployment concerns.
- Keep project documents represented both as full joined project text and as
  individual project sources when the pipeline depends on both forms.
- Do not send the full raw project package to the final prompt unless a task
  explicitly changes that architecture after analysis.
- Keep hidden corpus documents separate from uploaded user documents.
- Keep source grouping semantics stable unless the task is specifically about
  changing them.
- Keep precheck, summarization, retrieval, and final report generation as
  distinct conceptual stages.
- Keep public and internal/developer runtime modes separate when mode-specific
  behavior exists.
- Preserve fallback parsing behavior for damaged supported document formats
  unless a parsing task explicitly replaces it with an equivalent or better
  approach.

## Subsystem contracts

- Web/UI should collect user inputs and present results; it should not absorb
  core parsing, retrieval, summarization, or LLM analysis logic.
- Parsing should convert supported files to plain text and raise clear errors
  for unsupported formats.
- Source collection should label documents by role/group before retrieval.
- Summarization should preserve information needed by retrieval and final
  analysis, especially changes to goals, tasks, indicators, results, structure,
  financing, and supporting documents.
- Retrieval should operate over prepared context sources and expose enough debug
  data to understand selected context.
- Prompt logic should keep public/internal tone and report requirements explicit
  and reviewable.
- Report generation should distinguish factual findings from uncertainty and
  avoid inventing missing facts.
- Runtime configuration should come from documented environment variables, not
  hidden local assumptions.

## Testing strategy

- For new functionality, add a focused test when practical.
- If a test is not practical, state why and describe the manual or smoke check.
- Prefer unit tests, fixture-based tests, monkeypatch/mock for LLM calls, and
  smoke tests with deterministic fake LLMs.
- Do not run dependency installation or network-dependent checks unless the user
  asks or approves.
- Treat notebooks as operational tools unless a formal test runner is added.

## Documentation rules

Update docs when a change affects:

- architecture or component responsibility;
- pipeline behavior or order;
- contracts between web, parsing, retrieval, summaries, prompts, and reports;
- report format or checking rules;
- UI behavior;
- deployment/runtime commands or environment variables.

Keep documentation compact. Prefer updating the most specific file in `docs/`
over adding broad notes to `AGENTS.md`.

## Task interpretation

For each task, infer:

- goal;
- risk level;
- affected components;
- likely files;
- risks and possible regressions;
- acceptance criteria;
- minimal meaningful checks.

Ask for clarification only when a reasonable assumption would be risky or the
requested change conflicts with established contracts.

## Forbidden actions without explicit need

- Do not rewrite large parts of the project.
- Do not do mass refactors while adding functionality.
- Do not change multiple independent subsystems in one task.
- Do not remove existing behavior without impact analysis.
- Do not change mode-specific behavior casually.
- Do not delete hidden corpus files, fixtures, or user data.
- Do not run `git add`, `commit`, `checkout`, `reset`, or destructive commands
  unless explicitly requested.

## Self-review requirement

After meaningful changes, report:

- what changed;
- which files were touched;
- why pipeline or subsystem contracts are not broken;
- risks;
- how it was checked;
- what remains undone.
