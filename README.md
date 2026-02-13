# PolicyBridge

AI-powered decision support platform for Moroccan policymakers. Understand AI policy concepts, explore international case studies, and simulate policy impacts — all locally, for free.

## Features

- **AI Concept Chat** — RAG chatbot that explains AI policy terms in Morocco's context
- **Case Study Library** — 8 international AI policies with real outcome metrics
- **Impact Simulator** — Predicts multi-dimensional impacts of proposed policies on Morocco

## Prerequisites

- Python 3.11+ (tested on 3.12)
- Node.js 18+
- [Ollama](https://ollama.com/) for local LLM inference

## Quick Start

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
ollama serve
```

> `ollama serve` runs in the foreground. If you see "address already in use", Ollama is already running as a system service — that's fine, skip this step.

### 2. Set up the backend

```bash
# Install Python dependencies
pip install -r requirements.txt

# Create .env (optional — defaults work out of the box)
cat > .env << 'EOF'
OLLAMA_BASE_URL=http://localhost:11434/v1
LLM_MODEL=mistral
LLM_FALLBACK_MODEL=qwen2.5:3b
LLM_TEMPERATURE=0.4
LLM_MAX_TOKENS=1024
EOF

# Load data into the vector database (concepts, case studies, PDFs)
cd backend
python -m scripts.load_data

# Start the backend
uvicorn app.main:app --port 8000
```

### 3. Set up the frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

## Architecture

| Component | Technology |
|-----------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| Vector DB | ChromaDB (file-based, persistent) |
| Embeddings | BGE-M3 (1024D) or MiniLM fallback (384D) |
| LLM | Ollama + Mistral 7B (local, free) |
| Search | Hybrid: dense + BM25 + Reciprocal Rank Fusion |
| PDF extraction | pymupdf4llm |
| Frontend | React + Vite + Tailwind CSS |

## Data

- 10 curated AI policy concepts with Morocco-specific context
- 8 international case studies (EU, Canada, Singapore, Rwanda, Brazil, UK, Tunisia, South Korea)
- 6 policy PDF documents (Nigeria, USA, China, EU, International)
- Morocco context (demographics, economy, AI ecosystem, governance)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Status + collection counts |
| POST | `/api/concepts/ask` | RAG Q&A on AI policy concepts |
| GET | `/api/concepts/list` | List all concepts |
| GET | `/api/case-studies/` | List all case studies |
| POST | `/api/case-studies/search` | Hybrid search case studies |
| POST | `/api/simulate/predict` | Predict policy impact |
| GET | `/api/simulate/templates` | Policy templates |

## Troubleshooting

- **"Ollama is not running"** — Run `ollama serve` or check if the service is active: `systemctl status ollama`
- **ChromaDB + Python 3.14** — Use Python 3.12 instead. ChromaDB has a pydantic v1 incompatibility with 3.14.
- **BGE-M3 fails to load** — The system falls back to MiniLM (384D) automatically. Re-run `python -m scripts.load_data` after switching models.
