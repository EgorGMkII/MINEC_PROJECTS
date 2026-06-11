# Change risk levels

## Low risk

- Documentation updates.
- Comments.
- Focused tests and test fixtures.
- Small local helper cleanup with no behavior change.
- Narrow UI wording changes.

## Medium risk

- Prompt wording that can affect model behavior.
- New validation/checking layer.
- Report section or formatting changes.
- Retrieval parameter changes.
- Markdown/HTML rendering changes.

## High risk

- Parsing logic and broken DOCX fallback.
- `SourceText` model or document grouping.
- Pipeline order.
- UFAS precheck placement.
- Summary cache semantics.
- LLM client interface.
- Public/developer prompt selection.
- Dependency changes.

For high-risk work, use:

Explore -> Planning -> Review -> Patching -> Test

Do not combine high-risk architecture changes with unrelated new functionality.
