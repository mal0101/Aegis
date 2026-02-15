# Aegis

AI-powered decision support platform for Moroccan policymakers. Understand AI policy concepts, explore international case studies, and simulate policy impacts — all locally, for free.

## Features

- **AI Concept Chat** — RAG chatbot that explains AI policy terms in Morocco's context
- **Case Study Library** — 8 international AI policies with real outcome metrics
- **Impact Simulator** — Predicts multi-dimensional impacts of proposed policies on Morocco

## Prerequisites

- Python 3.11+ (tested on 3.12)
- Node.js 18+

## Architecture

| Component | Technology |
|-----------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| Vector DB | ChromaDB (file-based, persistent) |
| Embeddings | BGE-M3 (1024D) or MiniLM fallback (384D) |
| LLM | Qrok API |
| Search | Hybrid: dense + BM25 + Reciprocal Rank Fusion |
| PDF extraction | pymupdf4llm |
| Frontend | React + Vite + Tailwind CSS |

## Data

- 10 curated AI policy concepts with Morocco-specific context
- 8 international case studies (EU, Canada, Singapore, Rwanda, Brazil, UK, Tunisia, South Korea)
- 6 policy PDF documents (Nigeria, USA, China, EU, International)
- Morocco context (demographics, economy, AI ecosystem, governance)

