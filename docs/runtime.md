# Runtime

## Local commands

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the internal/developer web app:

```powershell
python -m uvicorn rag_web:app --reload --port 8010
```

Run the public wrapper:

```powershell
python -m uvicorn rag_web_public:app --reload --port 8010
```

## Docker

The Docker image runs:

```text
uvicorn rag_web:app --host 0.0.0.0 --port 8000
```

See `README.md` for current Docker build/run examples.

## Environment

The pipeline expects OpenAI-compatible configuration through environment
variables:

- `OPENAI_API_KEY`;
- `OPENAI_BASE_URL`;
- `OPENAI_MODEL`;
- `RAG_TOP_K`;
- `RAG_CHUNK_SIZE`;
- `RAG_CHUNK_OVERLAP`.

Avoid relying on hardcoded local defaults for secrets. Runtime behavior should be
documented here when environment handling changes.
