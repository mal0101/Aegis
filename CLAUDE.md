# PolicyBridge — Claude Code Project Rules

## What This Is

PolicyBridge is an AI-powered decision support platform for Moroccan policymakers. It has 3 integrated components:

1. **AI Concept Simulator** — RAG chatbot that explains AI policy terms (algorithmic bias, AI auditing, etc.) in the Morocco context
2. **Case Study Library** — Searchable repository of international AI policy outcomes with real metrics
3. **Impact Simulator** — Predicts multi-dimensional impacts of proposed policies on Morocco using evidence from similar countries

This is a **3-day hackathon MVP**. Speed of development is priority #1. Accuracy is #2. Architectural elegance is #3.

**Zero cost. Fully open source. No API keys. Everything runs locally.**

---

## Architecture Decisions (FINAL — do not change)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backend | Python 3.11, FastAPI, Uvicorn | Fast async API, Pydantic validation |
| Vector DB | ChromaDB (persistent, file-based) | Zero infra, good enough for <5K chunks |
| Embeddings | `BAAI/bge-m3` via FlagEmbedding (1024D) | Multilingual (FR/EN/AR), 8K token input, MTEB 63.0 |
| Embedding fallback | `sentence-transformers/all-MiniLM-L6-v2` (384D) | If BGE-M3 fails to load (RAM/disk) |
| LLM | **Ollama + Mistral 7B** (local, free) | Strong French+English, 7B fits in 8GB RAM, Ollama is zero-config |
| LLM fallback | **Ollama + qwen2.5:3b** | If machine has <8GB RAM, use a 3B model |
| PDF extraction | pymupdf4llm | Fastest, preserves structure as markdown |
| Search | Hybrid: dense (ChromaDB) + BM25 (rank_bm25) + RRF | Policy docs need both semantic AND keyword matching |
| Frontend | React + Vite + Tailwind CSS | Standard, fast to build |
| Auth/DB/Cache | NONE | Hackathon scope — no PostgreSQL, no Redis, no auth |

### Why Ollama + Mistral 7B

- **Mistral** was built by a French company — best open-source model for French+English mixed content
- **Ollama** exposes an OpenAI-compatible API at `http://localhost:11434/v1` — so the code uses the `openai` Python SDK with just a different `base_url`. Minimal code change.
- **Q4 quantized**: runs in ~4.5GB VRAM (GPU) or ~6GB RAM (CPU-only). Any modern laptop handles it.
- **No API key, no billing, no rate limits, no internet required after model download.**

---

## Directory Structure

```
policybridge/
├── CLAUDE.md                        # THIS FILE — project rules
├── .env                             # Ollama config (no secrets needed)
├── .env.example                     # Template
├── .gitignore
├── requirements.txt
├── README.md
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI app, CORS, routers, lifespan
│   │   ├── config.py                # ALL config in one place
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py           # Every Pydantic model
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── pdf_extractor.py     # PDF → structured markdown chunks
│   │   │   ├── embedding_service.py # BGE-M3 singleton (fallback: MiniLM)
│   │   │   ├── vectordb_service.py  # ChromaDB wrapper
│   │   │   ├── search_service.py    # Hybrid search: dense + BM25 + RRF
│   │   │   ├── llm_service.py       # Ollama via OpenAI-compatible API
│   │   │   ├── rag_service.py       # Concept Simulator RAG
│   │   │   ├── case_study_service.py# Case study search + compare
│   │   │   └── impact_service.py    # Impact prediction engine
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── concepts.py
│   │   │   ├── case_studies.py
│   │   │   └── simulator.py
│   │   └── data/
│   │       ├── morocco_context.json # Shared Morocco context
│   │       ├── concepts.json        # 10 AI policy concepts (curated)
│   │       ├── case_studies.json    # 8 case studies with metrics (curated)
│   │       ├── pdf_manifest.json    # Metadata for each PDF (human fills)
│   │       └── pdfs/                # Raw policy PDF files (human adds)
│   ├── scripts/
│   │   └── load_data.py             # Ingest JSON + PDFs → ChromaDB
│   └── tests/
│       └── test_integration.py
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── index.html
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       └── components/ ...
└── storage/
    └── chromadb/                    # Persistent vector store (gitignored)
```

---

## Coding Rules

### General
- **No premature abstraction.** Write the simplest code that works. No factory patterns, no dependency injection frameworks, no abstract base classes.
- **One file per service.** Each service file is self-contained with its class + singleton instance.
- **Config in one place.** All paths, model names, collection names, API settings go in `config.py`. Import from there.
- **Type everything.** All function signatures have type hints. All return types specified.
- **Log, don't print.** Use `logging.getLogger(__name__)` everywhere.
- **Fail loudly at startup, gracefully at runtime.** If a model can't load, crash. If a query fails, return an error response.

### Data Flow
```
PDF files  ──→  pymupdf4llm  ──→  markdown  ──→  structure-aware chunking  ──→  BGE-M3 embed  ──→  ChromaDB
JSON files ──→  build searchable_text  ──→  BGE-M3 embed  ──→  ChromaDB

User query ──→  BGE-M3 embed  ──→  ChromaDB dense search ──┐
           ──→  BM25 keyword search  ─────────────────────┤──→  RRF merge  ──→  top-K chunks  ──→  Ollama/Mistral  ──→  response
                                                          │
           ──→  metadata filter (country, type)  ──────────┘
```

### LLM Rules (Ollama-specific)
- **Ollama must be running** before starting the backend: `ollama serve`
- The `openai` Python SDK connects to Ollama at `http://localhost:11434/v1` with `api_key="ollama"`
- **Model name in config**: `mistral` (Ollama resolves to latest Mistral instruct)
- **Fallback model**: `qwen2.5:3b` if machine can't run 7B
- **Temperature**: 0.4 for concepts, 0.3 for impact analysis (Mistral 7B benefits from lower temp)
- **Max tokens**: 1024 for concepts, 2048 for impact simulator
- **System prompts must be concise** — 7B models have less capacity than GPT-4o-mini, so keep system prompts under 500 words and put detail in the user prompt instead
- If Ollama is not running, the LLM service should raise a clear error: "Ollama is not running. Start it with: ollama serve"

### Embedding Rules
- **Primary model:** `BAAI/bge-m3` — 1024 dimensions, cosine similarity
- **Fallback model:** `sentence-transformers/all-MiniLM-L6-v2` — 384 dimensions
- The embedding service MUST detect which model loaded and set `EMBEDDING_DIM` accordingly
- ChromaDB collection names include the dimension: `concepts_1024` or `concepts_384`
- If switching models, all collections must be re-built (run `load_data.py` again)

### Chunking Rules
- **Target chunk size:** 256–512 tokens
- **Split at structural boundaries:** article, section, numbered provision
- **Never split mid-paragraph**
- **Always prepend context:** `[Country: X] [Policy: Y] [Section: Z]\n\n{text}`
- **Overlap:** 10% at article boundaries (not within articles)

### Search Rules
- **Always use hybrid search** for case studies and policy document queries
- Dense search via ChromaDB (top_k * 2 candidates)
- BM25 search via rank_bm25 (top_k * 2 candidates)
- Merge with Reciprocal Rank Fusion (k=60)
- Return top_k final results
- **Metadata pre-filtering** when country or policy_type is specified

### API Rules
- All routes under `/api/` prefix
- POST for queries/searches, GET for lists/details
- Always return `processing_time_ms` in responses
- All errors return `{"error": "message", "detail": "..."}` with appropriate HTTP status
- CORS allows `http://localhost:5173` (Vite dev server)

### Frontend Rules
- Functional components with hooks only. No class components.
- Tailwind CSS only. No CSS files.
- All API calls go through `src/api.js` (centralized client).
- Loading states for every API call. No silent failures.
- Mobile-responsive (Tailwind breakpoints).

---

## Human Intervention Points

These are the ONLY things the human must do. Everything else is automated.

1. **Install Ollama** — one command: `curl -fsSL https://ollama.com/install.sh | sh`
2. **Pull the model** — one command: `ollama pull mistral`
3. **Start Ollama** — one command: `ollama serve` (or it auto-starts on install)
4. **Place PDF files** in `backend/app/data/pdfs/`
5. **Fill `pdf_manifest.json`** with metadata for each PDF (template is generated)
6. **Run `pip install -r requirements.txt`** (dependencies)
7. **Run `python -m scripts.load_data`** (initial data ingestion)
8. **Run `npm install`** in `frontend/` (frontend dependencies)

No API keys. No accounts. No billing. No internet after initial setup.

---

## Don'ts

- **DON'T** add PostgreSQL, Redis, Celery, or any infrastructure beyond what's listed
- **DON'T** add authentication, user accounts, or sessions
- **DON'T** add API versioning (no `/v1/` prefix)
- **DON'T** create abstract base classes or interface patterns
- **DON'T** split CSS into separate files — Tailwind only
- **DON'T** add environment-specific configs (dev/staging/prod) — single config
- **DON'T** use `print()` — use `logging`
- **DON'T** hardcode the embedding dimension — always read from `embedding_service.dim`
- **DON'T** create utility files like `utils.py` or `helpers.py` — put logic where it's used
- **DON'T** add comments explaining obvious code — only comment WHY, never WHAT
- **DON'T** use any paid API (OpenAI, Anthropic, Cohere, etc.) — everything is local and free
