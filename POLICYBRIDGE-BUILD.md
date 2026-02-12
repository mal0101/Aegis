# PolicyBridge — Complete Build Guide for Claude Code

> **Read CLAUDE.md first.** It contains architecture decisions, coding rules, and directory structure.
> This guide is sequential. Execute phases in order. Do not skip ahead.

---

## PHASE 0: Project Bootstrap

### Objective
Create the complete directory structure, install dependencies, set up configuration.

### Create directory structure
```
policybridge/
├── CLAUDE.md                    # (already provided — copy it to project root)
├── .env.example
├── .gitignore
├── requirements.txt
├── backend/
│   ├── app/
│   │   ├── __init__.py          # empty
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── models/
│   │   │   ├── __init__.py      # empty
│   │   │   └── schemas.py       # (created in Phase 4)
│   │   ├── services/
│   │   │   └── __init__.py      # empty
│   │   ├── routes/
│   │   │   └── __init__.py      # empty
│   │   └── data/
│   │       └── pdfs/            # empty dir — human adds PDFs here
│   ├── scripts/
│   │   └── __init__.py          # empty
│   └── tests/
│       └── __init__.py          # empty
├── frontend/                    # (created in Phase 5)
└── storage/
    └── chromadb/                # empty dir — auto-populated
```

### File: `requirements.txt`
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
chromadb==0.5.23
FlagEmbedding==1.3.3
sentence-transformers==3.3.1
torch>=2.0.0
openai==1.56.0
pymupdf4llm==0.0.17
pymupdf==1.25.1
rank-bm25==0.2.2
python-dotenv==1.0.1
pydantic==2.10.0
numpy>=1.24.0
```

> **Why is `openai` still here?** Ollama exposes an OpenAI-compatible API at `localhost:11434/v1`. We use the `openai` Python SDK pointed at Ollama — same code interface, zero cost, zero API key. This also means if someone wants to swap back to OpenAI later, it's a one-line config change.

### File: `.env.example`
```
# No API keys needed — everything runs locally for free
OLLAMA_BASE_URL=http://localhost:11434/v1
LLM_MODEL=mistral
LLM_FALLBACK_MODEL=qwen2.5:3b
LLM_TEMPERATURE=0.4
LLM_MAX_TOKENS=1024
```

### File: `.gitignore`
```
.env
__pycache__/
*.pyc
storage/chromadb/
node_modules/
dist/
.vite/
*.egg-info/
.pytest_cache/
```

### File: `backend/app/config.py`
```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
PROJECT_ROOT = BASE_DIR.parent.parent
CHROMADB_PATH = str(PROJECT_ROOT / "storage" / "chromadb")

# Data files
CONCEPTS_FILE = DATA_DIR / "concepts.json"
CASE_STUDIES_FILE = DATA_DIR / "case_studies.json"
MOROCCO_CONTEXT_FILE = DATA_DIR / "morocco_context.json"
PDF_MANIFEST_FILE = DATA_DIR / "pdf_manifest.json"

# Embedding
PRIMARY_EMBEDDING_MODEL = "BAAI/bge-m3"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM (Ollama — local, free, OpenAI-compatible API)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "qwen2.5:3b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ChromaDB collection names — set dynamically after embedding loads
# Format: {base_name}_{dim} e.g. "concepts_1024" or "concepts_384"
COLLECTION_CONCEPTS = "concepts"
COLLECTION_CASE_STUDIES = "case_studies"
COLLECTION_POLICY_DOCS = "policy_docs"

# Chunking
CHUNK_MIN_TOKENS = 128
CHUNK_MAX_TOKENS = 512
CHUNK_OVERLAP_RATIO = 0.1

# Search
SEARCH_TOP_K = 5
RRF_K = 60
DENSE_WEIGHT = 0.6
BM25_WEIGHT = 0.4
```

### File: `backend/app/main.py` (minimal — routers added in Phase 4)
```python
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="PolicyBridge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "PolicyBridge"}
```

### 🚨 HUMAN ACTION REQUIRED
1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
2. Pull the model: `ollama pull mistral` (~4.1GB download, one-time)
3. Verify Ollama is running: `ollama list` should show `mistral`
4. Run: `pip install -r requirements.txt`

> **Low-RAM machines (<8GB)?** Use `ollama pull qwen2.5:3b` instead and set `LLM_MODEL=qwen2.5:3b` in `.env`

### ✅ VERIFY
```bash
cd policybridge/backend
uvicorn app.main:app --reload --port 8000
# Visit http://localhost:8000/health → {"status": "healthy", "service": "PolicyBridge"}
# Visit http://localhost:8000/docs → Swagger UI loads
```

---

## PHASE 1: Data Foundation

### Objective
Create all JSON data files (curated content) and the PDF manifest template.

### What goes where
| Source | Purpose | Format |
|--------|---------|--------|
| `concepts.json` | 10 curated AI policy concepts with definitions, examples, Morocco context | JSON (hand-crafted) |
| `case_studies.json` | 8 international policies with structured outcome metrics | JSON (hand-crafted) |
| `morocco_context.json` | Morocco demographics, economy, AI ecosystem, governance | JSON (hand-crafted) |
| `pdf_manifest.json` | Metadata for each PDF the user provides | JSON (human fills template) |
| `data/pdfs/*.pdf` | Raw policy PDF documents | PDF (human provides) |

### File: `backend/app/data/morocco_context.json`
```json
{
  "country": "Morocco",
  "demographics": {
    "population": 37000000,
    "literacy_rate": 75.2,
    "languages": ["Arabic", "Darija", "Amazigh", "French"],
    "urban_population_pct": 64
  },
  "economy": {
    "gdp_usd_billions": 134,
    "gdp_per_capita_usd": 3600,
    "key_sectors": ["agriculture", "tourism", "manufacturing", "public_services", "healthcare", "education", "finance"],
    "digital_economy_pct_gdp": 3.5
  },
  "ai_ecosystem": {
    "ai_startups": 45,
    "research_centers": ["JAZARI Institute Rabat", "JAZARI Institute Benguerir"],
    "ai_researchers": 350,
    "university_programs": 12,
    "internet_penetration_pct": 84
  },
  "governance": {
    "data_protection_law": "Law 09-08 (CNDP)",
    "digital_strategy": "Digital Morocco 2030",
    "legal_system": "civil_law_french_tradition",
    "regulatory_body": "CNDP (Commission Nationale de Controle de la Protection des Donnees)",
    "ai_specific_regulation": false
  }
}
```

### File: `backend/app/data/concepts.json`

> **Claude Code: Use the EXACT concepts.json from the STEP-1-DATA-LAYER.md prompt file.**
> It contains 10 curated AI policy concepts with this schema per concept:
>
> ```json
> {
>   "id": "algorithmic-bias",
>   "term": "Algorithmic Bias",
>   "definition": "...",
>   "simple_explanation": "...",
>   "examples": ["...", "...", "..."],
>   "morocco_context": "...",
>   "policy_relevance": "...",
>   "related_concepts": ["fairness-in-ai", "training-data"],
>   "metadata": {
>     "difficulty": "intermediate",
>     "categories": ["fairness", "ethics"],
>     "sources": ["...", "..."]
>   },
>   "searchable_text": "..."
> }
> ```
>
> The 10 concepts are: `algorithmic-bias`, `explainable-ai`, `high-risk-ai`, `ai-auditing`, `training-data`, `ai-governance`, `regulatory-sandbox`, `data-governance`, `fairness-in-ai`, `transparency`.
>
> If STEP-1-DATA-LAYER.md is available, copy the concepts.json content exactly. If not, generate 10 concepts following this exact schema with rich Morocco-specific context.

### File: `backend/app/data/case_studies.json`

> **Claude Code: Use the EXACT case_studies.json from the STEP-1-DATA-LAYER.md prompt file.**
> It contains 8 case studies with this schema per case study:
>
> ```json
> {
>   "id": "eu_ai_act_2024",
>   "country": "EU",
>   "policy": {
>     "name": "EU AI Act",
>     "description": "...",
>     "type": "comprehensive",
>     "enacted_date": "2024-03-13",
>     "key_provisions": ["...", "..."],
>     "full_text_url": "..."
>   },
>   "outcomes": {
>     "measurement_period": "...",
>     "data_quality": "high",
>     "social_impact": { "trust_change_pct": 38, "bias_reduction_pct": 35 },
>     "economic_impact": { "compliance_costs_usd": 75000, "startup_growth_pct": -12 },
>     "implementation_reality": { "timeline_months": 28, "compliance_rate_pct": 60 },
>     "qualitative_insights": ["...", "..."]
>   },
>   "metadata": {
>     "gdp_ratio_to_morocco": 10.5,
>     "legal_similarity": 0.7,
>     "tech_maturity_gap": 0.3,
>     "tags": ["comprehensive", "risk-based"],
>     "sources": ["..."]
>   }
> }
> ```
>
> The 8 case studies are: `eu_ai_act_2024`, `canada_aia_2023`, `singapore_ai_framework_2023`, `rwanda_ai_policy_2023`, `brazil_ai_bill_2024`, `uk_ai_framework_2024`, `tunisia_ai_strategy_2023`, `south_korea_ai_act_2024`.
>
> If STEP-1-DATA-LAYER.md is available, copy the case_studies.json content exactly. If not, generate 8 case studies following this exact schema with realistic but clearly labeled projected outcome data.

### File: `backend/app/data/pdf_manifest.json` (TEMPLATE — human fills this)
```json
{
  "_instructions": "Fill one entry per PDF file in the pdfs/ directory. Remove this _instructions field when done.",
  "_human_action": "REQUIRED — you must fill this for each PDF you provide",
  "documents": [
    {
      "filename": "eu_ai_act_2024.pdf",
      "country": "EU",
      "policy_name": "EU Artificial Intelligence Act",
      "enacted_date": "2024-03-13",
      "policy_type": "comprehensive",
      "language": "en",
      "legal_system": "civil_law",
      "sectors": ["all"],
      "tags": ["risk-based", "comprehensive", "rights-based"]
    },
    {
      "filename": "rwanda_ai_policy_2023.pdf",
      "country": "Rwanda",
      "policy_name": "Rwanda National AI Policy",
      "enacted_date": "2023-06-01",
      "policy_type": "national_strategy",
      "language": "en",
      "legal_system": "civil_law",
      "sectors": ["agriculture", "healthcare", "education"],
      "tags": ["development-focused", "capacity-building"]
    }
  ]
}
```

### 🚨 HUMAN ACTION REQUIRED
1. Place your policy PDF files in `backend/app/data/pdfs/`
2. Edit `pdf_manifest.json` — add one entry per PDF with the metadata above
3. Valid `policy_type` values: `comprehensive`, `sectoral`, `voluntary`, `sandbox`, `national_strategy`, `bill`
4. Valid `language` values: `en`, `fr`, `ar`
5. Valid `legal_system` values: `civil_law`, `common_law`, `hybrid`, `sharia_influenced`

### ✅ VERIFY
```bash
ls backend/app/data/pdfs/         # Should list your PDF files
python -c "import json; d=json.load(open('backend/app/data/pdf_manifest.json')); print(f'{len(d[\"documents\"])} PDFs registered')"
python -c "import json; d=json.load(open('backend/app/data/concepts.json')); print(f'{len(d)} concepts')"
python -c "import json; d=json.load(open('backend/app/data/case_studies.json')); print(f'{len(d)} case studies')"
```
Expected: N PDFs registered, 10 concepts, 8 case studies

---

## PHASE 2: PDF Pipeline + Core Services

### Objective
Build the entire backend service layer: PDF extraction, chunking, embedding, vector storage, hybrid search, and LLM.

### Build order (strict dependencies)
```
pdf_extractor.py    (no deps)
embedding_service.py (no deps)
vectordb_service.py  (depends on: embedding_service for dim)
search_service.py    (depends on: embedding_service, vectordb_service)
llm_service.py       (no deps)
```

---

### File: `backend/app/services/pdf_extractor.py`

Handles the complete pipeline: PDF → markdown → structured chunks with metadata.

```python
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PolicyChunk:
    """A single chunk of policy text with full metadata."""
    chunk_id: str
    text: str                           # The actual text content
    enriched_text: str                  # Context-prefixed text for embedding
    source_file: str                    # PDF filename
    country: str
    policy_name: str
    enacted_date: str
    policy_type: str
    language: str
    legal_system: str
    section_header: str = ""
    tags: list = field(default_factory=list)
    chunk_index: int = 0
    total_chunks: int = 0


def extract_pdf_to_markdown(pdf_path: str) -> str:
    """Extract a PDF to structured markdown using pymupdf4llm."""
    import pymupdf4llm
    logger.info(f"Extracting PDF: {pdf_path}")
    markdown = pymupdf4llm.to_markdown(pdf_path)
    logger.info(f"Extracted {len(markdown)} chars from {Path(pdf_path).name}")
    return markdown


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars for English, 3 for French."""
    return len(text) // 4


def chunk_policy_document(
    markdown: str,
    manifest_entry: dict,
    min_tokens: int = 128,
    max_tokens: int = 512
) -> list[PolicyChunk]:
    """
    Structure-aware chunking for policy documents.

    Strategy:
    1. Split on markdown headers (articles, sections, chapters)
    2. If a section exceeds max_tokens, split at paragraph boundaries
    3. Merge tiny sections with their neighbors
    4. Prepend jurisdictional context to each chunk for embedding
    5. Never split mid-paragraph
    """
    filename = manifest_entry["filename"]
    country = manifest_entry["country"]
    policy_name = manifest_entry["policy_name"]
    enacted_date = manifest_entry.get("enacted_date", "unknown")
    policy_type = manifest_entry.get("policy_type", "unknown")
    language = manifest_entry.get("language", "en")
    legal_system = manifest_entry.get("legal_system", "unknown")
    tags = manifest_entry.get("tags", [])

    # Step 1: Split on markdown headers (##, ###, ####) or numbered articles
    # This regex keeps the header with its content
    sections = re.split(r'(?=\n#{1,4}\s+)', markdown)
    # Also split on common policy patterns like "Article N" or "Section N"
    refined_sections = []
    for section in sections:
        # Split further on "Article X" patterns if section is very long
        if estimate_tokens(section) > max_tokens * 2:
            subsections = re.split(r'(?=\n(?:Article|Section|Chapter|ARTICLE|SECTION|CHAPTER)\s+\d+)', section)
            refined_sections.extend(subsections)
        else:
            refined_sections.append(section)

    # Step 2: Process each section into chunks
    raw_chunks = []
    for section in refined_sections:
        section = section.strip()
        if not section or estimate_tokens(section) < 20:
            continue

        # Extract header for metadata
        header_match = re.match(r'^(#{1,4}\s+.+?)$', section, re.MULTILINE)
        section_header = header_match.group(1).strip('# ').strip() if header_match else ""

        if estimate_tokens(section) <= max_tokens:
            raw_chunks.append((section, section_header))
        else:
            # Split at paragraph boundaries (double newline)
            paragraphs = section.split('\n\n')
            current = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if estimate_tokens(current + "\n\n" + para) > max_tokens and current:
                    raw_chunks.append((current.strip(), section_header))
                    current = para
                else:
                    current = (current + "\n\n" + para).strip()
            if current.strip():
                raw_chunks.append((current.strip(), section_header))

    # Step 3: Merge tiny chunks with neighbors
    merged_chunks = []
    buffer = ""
    buffer_header = ""
    for text, header in raw_chunks:
        if estimate_tokens(buffer + "\n\n" + text) <= max_tokens:
            buffer = (buffer + "\n\n" + text).strip()
            buffer_header = buffer_header or header
        else:
            if buffer:
                merged_chunks.append((buffer, buffer_header))
            buffer = text
            buffer_header = header
    if buffer:
        merged_chunks.append((buffer, buffer_header))

    # Step 4: Build PolicyChunk objects with enriched text
    context_prefix = f"[Country: {country}] [Policy: {policy_name}] [Enacted: {enacted_date}] [Type: {policy_type}] [Legal System: {legal_system}]"

    chunks = []
    for i, (text, header) in enumerate(merged_chunks):
        enriched = f"{context_prefix}\n[Section: {header}]\n\n{text}" if header else f"{context_prefix}\n\n{text}"

        chunks.append(PolicyChunk(
            chunk_id=f"{Path(filename).stem}_chunk_{i:04d}",
            text=text,
            enriched_text=enriched,
            source_file=filename,
            country=country,
            policy_name=policy_name,
            enacted_date=enacted_date,
            policy_type=policy_type,
            language=language,
            legal_system=legal_system,
            section_header=header,
            tags=tags,
            chunk_index=i,
            total_chunks=len(merged_chunks)  # updated after loop
        ))

    # Fix total_chunks
    for c in chunks:
        c.total_chunks = len(chunks)

    logger.info(f"Chunked {filename}: {len(chunks)} chunks from {len(markdown)} chars")
    return chunks


def process_all_pdfs(pdf_dir: str, manifest: dict) -> list[PolicyChunk]:
    """Process all PDFs listed in the manifest."""
    from app.config import CHUNK_MIN_TOKENS, CHUNK_MAX_TOKENS

    all_chunks = []
    pdf_path = Path(pdf_dir)

    for entry in manifest.get("documents", []):
        filepath = pdf_path / entry["filename"]
        if not filepath.exists():
            logger.warning(f"PDF not found: {filepath} — skipping")
            continue

        try:
            markdown = extract_pdf_to_markdown(str(filepath))
            chunks = chunk_policy_document(
                markdown, entry,
                min_tokens=CHUNK_MIN_TOKENS,
                max_tokens=CHUNK_MAX_TOKENS
            )
            all_chunks.extend(chunks)
            logger.info(f"✓ {entry['filename']}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"✗ Failed to process {entry['filename']}: {e}")

    logger.info(f"Total PDF chunks: {len(all_chunks)}")
    return all_chunks
```

---

### File: `backend/app/services/embedding_service.py`

BGE-M3 primary, MiniLM fallback. Singleton. Exposes model dimension.

```python
import logging
from typing import List

logger = logging.getLogger(__name__)


class EmbeddingService:
    _instance = None
    _model = None
    _model_name = None
    dim: int = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is not None:
            return
        self._try_load()

    def _try_load(self):
        from app.config import PRIMARY_EMBEDDING_MODEL, FALLBACK_EMBEDDING_MODEL

        # Try BGE-M3 first
        try:
            logger.info(f"Loading primary embedding model: {PRIMARY_EMBEDDING_MODEL}")
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(PRIMARY_EMBEDDING_MODEL, use_fp16=True)
            self._model_name = PRIMARY_EMBEDDING_MODEL
            self.dim = 1024
            logger.info(f"✓ Loaded BGE-M3 (dim={self.dim})")
            return
        except Exception as e:
            logger.warning(f"BGE-M3 failed to load: {e}")

        # Fallback to MiniLM
        try:
            logger.info(f"Loading fallback model: {FALLBACK_EMBEDDING_MODEL}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL)
            self._model_name = FALLBACK_EMBEDDING_MODEL
            self.dim = 384
            logger.info(f"✓ Loaded MiniLM fallback (dim={self.dim})")
            return
        except Exception as e:
            logger.error(f"Fallback model also failed: {e}")
            raise RuntimeError("No embedding model could be loaded. Check dependencies.")

    @property
    def is_bge_m3(self) -> bool:
        return "bge-m3" in (self._model_name or "")

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.is_bge_m3:
            output = self._model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
            # BGE-M3 returns dict with 'dense_vecs'
            return output["dense_vecs"].tolist() if hasattr(output["dense_vecs"], "tolist") else output["dense_vecs"]
        else:
            embeddings = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
            return embeddings.tolist()

    def get_collection_name(self, base_name: str) -> str:
        """Return collection name with dimension suffix: e.g., concepts_1024"""
        return f"{base_name}_{self.dim}"


# Singleton
embedding_service = EmbeddingService()
```

---

### File: `backend/app/services/vectordb_service.py`

ChromaDB wrapper. Uses the dimension from embedding_service for collection naming.

```python
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import logging
from app.config import CHROMADB_PATH

logger = logging.getLogger(__name__)


class VectorDBService:
    def __init__(self):
        logger.info(f"Initializing ChromaDB at: {CHROMADB_PATH}")
        self._client = chromadb.PersistentClient(
            path=CHROMADB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

    def get_or_create_collection(self, name: str):
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )

    def delete_collection(self, name: str):
        try:
            self._client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")
        except Exception:
            pass  # Collection didn't exist

    def add_documents(
        self,
        collection,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ):
        # ChromaDB requires metadata values to be str, int, float, or bool
        clean_metadatas = []
        for m in metadatas:
            clean = {}
            for k, v in m.items():
                if isinstance(v, list):
                    clean[k] = ", ".join(str(x) for x in v)
                elif isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            clean_metadatas.append(clean)

        # ChromaDB has batch limits — process in chunks of 500
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=clean_metadatas[i:end]
            )
        logger.info(f"Added {len(ids)} documents to collection '{collection.name}'")

    def search(
        self,
        collection,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        }

    def get_collection_count(self, collection) -> int:
        return collection.count()

    def get_by_id(self, collection, doc_id: str) -> Optional[Dict]:
        results = collection.get(ids=[doc_id], include=["documents", "metadatas"])
        if results["ids"]:
            return {
                "id": results["ids"][0],
                "document": results["documents"][0] if results["documents"] else None,
                "metadata": results["metadatas"][0] if results["metadatas"] else None,
            }
        return None


# Singleton
vectordb_service = VectorDBService()
```

---

### File: `backend/app/services/search_service.py`

Hybrid search: dense (ChromaDB) + BM25 (rank_bm25) + Reciprocal Rank Fusion.

```python
import logging
from typing import List, Dict, Optional, Any
from rank_bm25 import BM25Okapi
from app.config import RRF_K, SEARCH_TOP_K, DENSE_WEIGHT, BM25_WEIGHT

logger = logging.getLogger(__name__)


class SearchService:
    """
    Hybrid search combining dense vector search (ChromaDB) with
    BM25 keyword search, merged via Reciprocal Rank Fusion (RRF).
    """

    def __init__(self):
        # BM25 indices per collection — built during data loading
        self._bm25_indices: Dict[str, BM25Okapi] = {}
        self._bm25_docs: Dict[str, List[Dict]] = {}  # id, text, metadata per collection

    def build_bm25_index(self, collection_name: str, documents: List[Dict[str, Any]]):
        """
        Build a BM25 index for a collection.
        documents: list of {"id": str, "text": str, "metadata": dict}
        """
        tokenized = [doc["text"].lower().split() for doc in documents]
        self._bm25_indices[collection_name] = BM25Okapi(tokenized)
        self._bm25_docs[collection_name] = documents
        logger.info(f"Built BM25 index for '{collection_name}': {len(documents)} docs")

    def bm25_search(self, collection_name: str, query: str, top_k: int = 10) -> List[Dict]:
        """Keyword search using BM25."""
        if collection_name not in self._bm25_indices:
            return []

        bm25 = self._bm25_indices[collection_name]
        docs = self._bm25_docs[collection_name]
        tokenized_query = query.lower().split()

        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "id": docs[idx]["id"],
                    "document": docs[idx]["text"],
                    "metadata": docs[idx]["metadata"],
                    "bm25_score": float(scores[idx])
                })
        return results

    def hybrid_search(
        self,
        collection_name: str,
        query: str,
        query_embedding: List[float],
        chromadb_collection,
        top_k: int = None,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Hybrid search: dense + BM25 merged with RRF.

        1. Run dense search (ChromaDB) → 2x candidates
        2. Run BM25 search → 2x candidates
        3. Merge with Reciprocal Rank Fusion
        4. Return top_k results
        """
        if top_k is None:
            top_k = SEARCH_TOP_K

        candidate_count = top_k * 3

        # Dense search
        from app.services.vectordb_service import vectordb_service
        dense_results = vectordb_service.search(
            chromadb_collection,
            query_embedding,
            top_k=candidate_count,
            where=where
        )

        # BM25 search
        bm25_results = self.bm25_search(collection_name, query, top_k=candidate_count)

        # Build ranked lists
        dense_ranked = {}
        for rank, doc_id in enumerate(dense_results["ids"]):
            dense_ranked[doc_id] = {
                "rank": rank + 1,
                "document": dense_results["documents"][rank],
                "metadata": dense_results["metadatas"][rank],
                "distance": dense_results["distances"][rank]
            }

        bm25_ranked = {}
        for rank, result in enumerate(bm25_results):
            bm25_ranked[result["id"]] = {
                "rank": rank + 1,
                "document": result["document"],
                "metadata": result["metadata"],
                "bm25_score": result["bm25_score"]
            }

        # RRF fusion
        all_ids = set(dense_ranked.keys()) | set(bm25_ranked.keys())
        fused = []

        for doc_id in all_ids:
            rrf_score = 0.0
            if doc_id in dense_ranked:
                rrf_score += DENSE_WEIGHT * (1.0 / (RRF_K + dense_ranked[doc_id]["rank"]))
            if doc_id in bm25_ranked:
                rrf_score += BM25_WEIGHT * (1.0 / (RRF_K + bm25_ranked[doc_id]["rank"]))

            # Get document and metadata from whichever source has it
            doc_info = dense_ranked.get(doc_id) or bm25_ranked.get(doc_id)
            fused.append({
                "id": doc_id,
                "document": doc_info["document"],
                "metadata": doc_info["metadata"],
                "rrf_score": rrf_score,
                "in_dense": doc_id in dense_ranked,
                "in_bm25": doc_id in bm25_ranked,
            })

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused[:top_k]


# Singleton
search_service = SearchService()
```

---

### File: `backend/app/services/llm_service.py`

Connects to Ollama's OpenAI-compatible API. No API key needed.

```python
import logging
from openai import OpenAI
from app.config import OLLAMA_BASE_URL, LLM_MODEL, LLM_FALLBACK_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self._model = LLM_MODEL
        self._client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama"  # Ollama doesn't need a real key but the SDK requires the field
        )
        # Verify Ollama is reachable and model is available
        self._verify_connection()
        logger.info(f"LLM service initialized: {self._model} via Ollama")

    def _verify_connection(self):
        """Check that Ollama is running and the model is pulled."""
        try:
            # Quick test — list models
            models = self._client.models.list()
            available = [m.id for m in models.data]
            logger.info(f"Ollama models available: {available}")

            if self._model not in available and f"{self._model}:latest" not in available:
                # Try without :latest suffix matching
                matching = [m for m in available if m.startswith(self._model)]
                if not matching:
                    logger.warning(f"Model '{self._model}' not found. Available: {available}")
                    # Try fallback
                    fallback_match = [m for m in available if m.startswith(LLM_FALLBACK_MODEL)]
                    if fallback_match:
                        self._model = LLM_FALLBACK_MODEL
                        logger.info(f"Using fallback model: {self._model}")
                    else:
                        raise RuntimeError(
                            f"Neither '{LLM_MODEL}' nor '{LLM_FALLBACK_MODEL}' found in Ollama. "
                            f"Run: ollama pull {LLM_MODEL}"
                        )
        except Exception as e:
            if "Connection" in str(type(e).__name__) or "refused" in str(e).lower():
                raise RuntimeError(
                    "Ollama is not running. Start it with: ollama serve\n"
                    "Install Ollama: curl -fsSL https://ollama.com/install.sh | sh\n"
                    f"Then pull the model: ollama pull {LLM_MODEL}"
                ) from e
            raise

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature or LLM_TEMPERATURE,
                max_tokens=max_tokens or LLM_MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise


# Singleton
llm_service = LLMService()
```

> **Ollama tips for 7B models:**
> - Keep system prompts **under 500 words** — Mistral 7B has less capacity than GPT-4o-mini
> - Put retrieval context in the **user prompt**, not system prompt
> - Use temperature **0.3-0.4** — lower is better for factual policy analysis
> - Set max_tokens to **2048** for the impact simulator (long analysis output)
> - First inference is slow (~5-10s cold start), subsequent calls are fast (~1-3s)

### ✅ VERIFY (Phase 2)
```bash
cd policybridge/backend
python -c "from app.services.embedding_service import embedding_service; print(f'Model: {embedding_service._model_name}, dim: {embedding_service.dim}')"
python -c "from app.services.vectordb_service import vectordb_service; print('ChromaDB OK')"
python -c "from app.services.search_service import search_service; print('Search service OK')"
```
Expected: Model name printed, dim 1024 (or 384 fallback), all services initialize.

---

## PHASE 3: Data Loading Script

### Objective
Single script that ingests ALL data (JSON concepts + JSON case studies + PDF documents) into ChromaDB and builds BM25 indices. Run once. Idempotent (deletes and re-creates collections).

### File: `backend/scripts/load_data.py`

```python
"""
Data loader: Ingests concepts, case studies, and PDF documents into ChromaDB.
Run from backend/: python -m scripts.load_data
"""
import json
import sys
import time
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    CONCEPTS_FILE, CASE_STUDIES_FILE, PDF_MANIFEST_FILE, PDF_DIR,
    COLLECTION_CONCEPTS, COLLECTION_CASE_STUDIES, COLLECTION_POLICY_DOCS
)
from app.services.embedding_service import embedding_service
from app.services.vectordb_service import vectordb_service
from app.services.search_service import search_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("load_data")


def load_concepts():
    """Load curated concepts from JSON into ChromaDB."""
    collection_name = embedding_service.get_collection_name(COLLECTION_CONCEPTS)
    vectordb_service.delete_collection(collection_name)
    collection = vectordb_service.get_or_create_collection(collection_name)

    with open(CONCEPTS_FILE) as f:
        concepts = json.load(f)

    ids = []
    texts = []
    metadatas = []
    bm25_docs = []

    for c in concepts:
        doc_id = c["id"]
        searchable = c.get("searchable_text", f"{c['term']} {c['definition']} {c.get('morocco_context', '')}")

        ids.append(doc_id)
        texts.append(searchable)
        metadatas.append({
            "term": c["term"],
            "difficulty": c.get("metadata", {}).get("difficulty", "intermediate"),
            "categories": ", ".join(c.get("metadata", {}).get("categories", [])),
            "source_type": "concept"
        })
        bm25_docs.append({"id": doc_id, "text": searchable, "metadata": metadatas[-1]})

    logger.info(f"Embedding {len(ids)} concepts...")
    embeddings = embedding_service.embed_batch(texts)
    vectordb_service.add_documents(collection, ids, texts, embeddings, metadatas)
    search_service.build_bm25_index(collection_name, bm25_docs)
    logger.info(f"✓ Loaded {len(ids)} concepts into '{collection_name}'")


def load_case_studies():
    """Load case studies from JSON into ChromaDB."""
    collection_name = embedding_service.get_collection_name(COLLECTION_CASE_STUDIES)
    vectordb_service.delete_collection(collection_name)
    collection = vectordb_service.get_or_create_collection(collection_name)

    with open(CASE_STUDIES_FILE) as f:
        case_studies = json.load(f)

    ids = []
    texts = []
    metadatas = []
    bm25_docs = []

    for cs in case_studies:
        doc_id = cs["id"]
        policy = cs["policy"]

        # Build rich searchable text from structured data
        searchable = (
            f"{policy['name']} — {cs['country']}\n"
            f"{policy['description']}\n"
            f"Key provisions: {'; '.join(policy.get('key_provisions', []))}\n"
            f"Insights: {'; '.join(cs.get('outcomes', {}).get('qualitative_insights', []))}"
        )

        ids.append(doc_id)
        texts.append(searchable)
        metadatas.append({
            "country": cs["country"],
            "policy_name": policy["name"],
            "policy_type": policy.get("type", ""),
            "enacted_date": policy.get("enacted_date", ""),
            "data_quality": cs.get("outcomes", {}).get("data_quality", "medium"),
            "gdp_ratio_to_morocco": cs.get("metadata", {}).get("gdp_ratio_to_morocco", 1.0),
            "legal_similarity": cs.get("metadata", {}).get("legal_similarity", 0.5),
            "tags": ", ".join(cs.get("metadata", {}).get("tags", [])),
            "source_type": "case_study"
        })
        bm25_docs.append({"id": doc_id, "text": searchable, "metadata": metadatas[-1]})

    logger.info(f"Embedding {len(ids)} case studies...")
    embeddings = embedding_service.embed_batch(texts)
    vectordb_service.add_documents(collection, ids, texts, embeddings, metadatas)
    search_service.build_bm25_index(collection_name, bm25_docs)
    logger.info(f"✓ Loaded {len(ids)} case studies into '{collection_name}'")


def load_pdf_documents():
    """Extract, chunk, embed, and store all PDF documents."""
    if not PDF_MANIFEST_FILE.exists():
        logger.warning("No pdf_manifest.json found — skipping PDF loading")
        return 0

    with open(PDF_MANIFEST_FILE) as f:
        manifest = json.load(f)

    documents = manifest.get("documents", [])
    if not documents:
        logger.warning("pdf_manifest.json has no documents — skipping PDF loading")
        return 0

    from app.services.pdf_extractor import process_all_pdfs

    chunks = process_all_pdfs(str(PDF_DIR), manifest)
    if not chunks:
        logger.warning("No chunks produced from PDFs")
        return 0

    collection_name = embedding_service.get_collection_name(COLLECTION_POLICY_DOCS)
    vectordb_service.delete_collection(collection_name)
    collection = vectordb_service.get_or_create_collection(collection_name)

    ids = [c.chunk_id for c in chunks]
    texts = [c.enriched_text for c in chunks]
    metadatas = [{
        "source_file": c.source_file,
        "country": c.country,
        "policy_name": c.policy_name,
        "enacted_date": c.enacted_date,
        "policy_type": c.policy_type,
        "language": c.language,
        "legal_system": c.legal_system,
        "section_header": c.section_header,
        "tags": ", ".join(c.tags),
        "chunk_index": c.chunk_index,
        "total_chunks": c.total_chunks,
        "source_type": "pdf_chunk"
    } for c in chunks]

    bm25_docs = [{"id": c.chunk_id, "text": c.enriched_text, "metadata": m}
                 for c, m in zip(chunks, metadatas)]

    logger.info(f"Embedding {len(ids)} PDF chunks...")
    # Batch embed in groups to avoid OOM on large corpora
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        all_embeddings.extend(embedding_service.embed_batch(batch))
        logger.info(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)} chunks")

    vectordb_service.add_documents(collection, ids, texts, all_embeddings, metadatas)
    search_service.build_bm25_index(collection_name, bm25_docs)
    logger.info(f"✓ Loaded {len(ids)} PDF chunks into '{collection_name}'")
    return len(chunks)


def main():
    start = time.time()

    logger.info("=" * 60)
    logger.info("PolicyBridge Data Loader")
    logger.info(f"Embedding model: {embedding_service._model_name} (dim={embedding_service.dim})")
    logger.info("=" * 60)

    load_concepts()
    load_case_studies()
    pdf_count = load_pdf_documents()

    # Print summary
    elapsed = time.time() - start
    concepts_col = embedding_service.get_collection_name(COLLECTION_CONCEPTS)
    cases_col = embedding_service.get_collection_name(COLLECTION_CASE_STUDIES)
    docs_col = embedding_service.get_collection_name(COLLECTION_POLICY_DOCS)

    c1 = vectordb_service.get_or_create_collection(concepts_col)
    c2 = vectordb_service.get_or_create_collection(cases_col)

    logger.info("=" * 60)
    logger.info(f"DONE in {elapsed:.1f}s")
    logger.info(f"  Concepts:    {vectordb_service.get_collection_count(c1)}")
    logger.info(f"  Case Studies: {vectordb_service.get_collection_count(c2)}")

    if pdf_count > 0:
        c3 = vectordb_service.get_or_create_collection(docs_col)
        logger.info(f"  PDF Chunks:  {vectordb_service.get_collection_count(c3)}")
    else:
        logger.info(f"  PDF Chunks:  0 (no PDFs provided — this is OK for demo)")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
```

### 🚨 HUMAN ACTION REQUIRED
Before running `load_data.py`:
1. Ensure Ollama is running: `ollama list` (not needed for data loading, but good to verify)
2. Ensure `concepts.json` and `case_studies.json` exist with data
3. If you have PDFs: ensure they're in `data/pdfs/` and `pdf_manifest.json` is filled
4. If you have NO PDFs yet: that's fine — the loader skips PDFs gracefully

### ✅ VERIFY
```bash
cd policybridge/backend
python -m scripts.load_data
```
Expected output:
```
PolicyBridge Data Loader
Embedding model: BAAI/bge-m3 (dim=1024)
✓ Loaded 10 concepts into 'concepts_1024'
✓ Loaded 8 case studies into 'case_studies_1024'
✓ Loaded N PDF chunks into 'policy_docs_1024'   # if PDFs provided
DONE in Xs
```

Quick search test:
```bash
python -c "
from app.services.embedding_service import embedding_service
from app.services.vectordb_service import vectordb_service

col = vectordb_service.get_or_create_collection(embedding_service.get_collection_name('concepts'))
q = embedding_service.embed_text('AI discrimination and fairness')
results = vectordb_service.search(col, q, top_k=3)
for i, doc_id in enumerate(results['ids']):
    print(f'{i+1}. {doc_id} (distance: {results[\"distances\"][i]:.4f})')
"
```
Expected: `algorithmic-bias` or `fairness-in-ai` as top result.

---

## PHASE 4: API Layer — RAG, Routes, Schemas

### Objective
Build the complete API: Pydantic schemas, RAG service for concepts, case study service, impact simulator, and all FastAPI routes.

### File: `backend/app/models/schemas.py`

> **Claude Code: Use the EXACT schemas from the STEP-3-RAG-AND-CONCEPT-API.md file** (ConceptQuestion, ConceptAnswer, RelatedConcept, ConceptSummary, CaseStudySearchQuery, CaseStudySummary, CaseStudyDetail, PolicyProposal, ImpactDimension, ImpactPrediction, HealthResponse).
>
> If STEP-3 is not available, create Pydantic models matching these signatures:
>
> - `ConceptQuestion(question: str, difficulty: Optional[str] = None)`
> - `ConceptAnswer(question: str, answer: str, related_concepts: list, sources: list, processing_time_ms: int)`
> - `CaseStudySearchQuery(query: str, country: Optional[str], policy_type: Optional[str], top_k: int = 5)`
> - `CaseStudySummary(id: str, country: str, policy_name: str, policy_type: str, enacted_date: str, data_quality: str, tags: list, relevance_score: Optional[float])`
> - `CaseStudyDetail` — full case study with all outcome metrics
> - `PolicyProposal(policy_name: str, description: str, sectors: list[str], policy_type: str = "comprehensive")`
> - `ImpactPrediction(policy_name: str, executive_summary: str, full_analysis: str, similar_policies: list, impact_dimensions: list, recommendations: list, evidence_base_size: int, processing_time_ms: int)`

---

### File: `backend/app/services/rag_service.py`

> **Claude Code: Build this following the logic from STEP-3-RAG-AND-CONCEPT-API.md**, with these KEY CHANGES:
>
> 1. **Use hybrid search** instead of pure vector search:
>    ```python
>    from app.services.search_service import search_service
>    from app.services.embedding_service import embedding_service
>
>    # In answer_question():
>    collection_name = embedding_service.get_collection_name("concepts")
>    collection = vectordb_service.get_or_create_collection(collection_name)
>    query_embedding = embedding_service.embed_text(question)
>
>    results = search_service.hybrid_search(
>        collection_name=collection_name,
>        query=question,
>        query_embedding=query_embedding,
>        chromadb_collection=collection,
>        top_k=3
>    )
>    ```
>
> 2. **Also search policy_docs collection** if it exists (PDF chunks), and merge results:
>    ```python
>    # Also search PDF policy documents for supplementary context
>    pdf_collection_name = embedding_service.get_collection_name("policy_docs")
>    try:
>        pdf_collection = vectordb_service.get_or_create_collection(pdf_collection_name)
>        if vectordb_service.get_collection_count(pdf_collection) > 0:
>            pdf_results = search_service.hybrid_search(
>                collection_name=pdf_collection_name,
>                query=question,
>                query_embedding=query_embedding,
>                chromadb_collection=pdf_collection,
>                top_k=2
>            )
>            # Append PDF context to the LLM prompt
>    except Exception:
>        pass  # No PDF data available — that's fine
>    ```
>
> 3. **Load Morocco context** from `morocco_context.json` at startup and inject into system prompt
> 4. **System prompt must include**: Morocco demographics, economy, AI ecosystem, governance context
> 5. **answer_question()** returns: question, answer, related_concepts, sources, processing_time_ms

---

### File: `backend/app/services/case_study_service.py`

> **Claude Code: Build this following the logic from STEP-4-CASE-STUDY-LIBRARY.md**, with these KEY CHANGES:
>
> 1. **Use hybrid search** (same pattern as rag_service above)
> 2. **search()** also queries the PDF policy_docs collection for supplementary text
> 3. **find_similar()** is used by the Impact Simulator — keep this method
> 4. **compare()** takes a list of case study IDs and returns side-by-side metrics
> 5. Load `case_studies.json` at startup for get_all() and get_by_id() (fast lookups from JSON)

---

### File: `backend/app/services/impact_service.py`

> **Claude Code: Use the EXACT implementation from STEP-5-IMPACT-SIMULATOR.md.**
> This is the crown jewel. It:
> 1. Takes a PolicyProposal (name, description, sectors, type)
> 2. Finds similar international policies via case_study_service.find_similar()
> 3. Aggregates outcomes (trust, bias, cost, startups, timeline, compliance) with Morocco GDP adjustment
> 4. Generates analysis via LLM with evidence context
> 5. Returns structured ImpactPrediction with dimensions, recommendations, evidence base
>
> Key config:
> - 5 policy templates: bias_testing, transparency_requirement, ai_sandbox, data_governance, comprehensive_ai_act
> - LLM temperature: 0.3 (lower for analytical output — especially important with 7B models)
> - Max tokens: 2048 (analysis can be long)
> - **System prompts must be under 500 words** — Mistral 7B has limited instruction-following capacity compared to GPT-4o-mini. Put retrieval context in the user prompt, not the system prompt.

---

### Routes

> **Claude Code: Create these three route files following STEP-3, STEP-4, and STEP-5 prompts:**

### File: `backend/app/routes/concepts.py`
```
POST /api/concepts/ask          → ConceptAnswer (main Q&A)
GET  /api/concepts/list         → List[ConceptSummary]
GET  /api/concepts/{id}         → Full concept details
```

### File: `backend/app/routes/case_studies.py`
```
GET  /api/case-studies/          → List all summaries
POST /api/case-studies/search    → Semantic + keyword search with filters
POST /api/case-studies/find-similar → For impact simulator
POST /api/case-studies/compare   → Side-by-side metrics
GET  /api/case-studies/{id}      → Full case study details
```

### File: `backend/app/routes/simulator.py`
```
POST /api/simulate/predict       → ImpactPrediction (main simulation)
GET  /api/simulate/templates     → List of 5 policy templates
```

### Update `backend/app/main.py`

Register all routers and add the health check that reports collection counts:

```python
from app.routes.concepts import router as concepts_router
from app.routes.case_studies import router as case_studies_router
from app.routes.simulator import router as simulator_router

app.include_router(concepts_router)
app.include_router(case_studies_router)
app.include_router(simulator_router)

@app.get("/health")
async def health():
    from app.services.embedding_service import embedding_service
    from app.services.vectordb_service import vectordb_service

    counts = {}
    for base in ["concepts", "case_studies", "policy_docs"]:
        col_name = embedding_service.get_collection_name(base)
        try:
            col = vectordb_service.get_or_create_collection(col_name)
            counts[base] = vectordb_service.get_collection_count(col)
        except Exception:
            counts[base] = 0

    return {
        "status": "healthy",
        "embedding_model": embedding_service._model_name,
        "embedding_dim": embedding_service.dim,
        "collections": counts
    }
```

### ⚠️ IMPORTANT: BM25 Index Rebuild on Startup

The BM25 indices live in memory and are lost when the server restarts. Add this to `main.py`:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Rebuild BM25 indices on startup from ChromaDB data
    logger.info("Rebuilding BM25 indices...")
    from app.services.embedding_service import embedding_service
    from app.services.vectordb_service import vectordb_service
    from app.services.search_service import search_service

    for base_name in ["concepts", "case_studies", "policy_docs"]:
        col_name = embedding_service.get_collection_name(base_name)
        try:
            col = vectordb_service.get_or_create_collection(col_name)
            count = vectordb_service.get_collection_count(col)
            if count > 0:
                # Fetch all documents from ChromaDB to rebuild BM25
                all_data = col.get(include=["documents", "metadatas"])
                bm25_docs = [
                    {"id": all_data["ids"][i], "text": all_data["documents"][i], "metadata": all_data["metadatas"][i]}
                    for i in range(len(all_data["ids"]))
                ]
                search_service.build_bm25_index(col_name, bm25_docs)
        except Exception as e:
            logger.warning(f"Could not rebuild BM25 for {col_name}: {e}")

    logger.info("BM25 indices ready")
    yield
    logger.info("Shutting down")

# Replace app = FastAPI(...) with:
app = FastAPI(title="PolicyBridge API", version="1.0.0", lifespan=lifespan)
```

### ✅ VERIFY (Phase 4)
```bash
cd policybridge/backend
uvicorn app.main:app --reload --port 8000
```

Test each endpoint:
```bash
# Health
curl http://localhost:8000/health

# Concepts
curl http://localhost:8000/api/concepts/list
curl -X POST http://localhost:8000/api/concepts/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is algorithmic bias and why does it matter for Morocco?"}'

# Case Studies
curl http://localhost:8000/api/case-studies/
curl -X POST http://localhost:8000/api/case-studies/search \
  -H "Content-Type: application/json" \
  -d '{"query": "transparency regulation africa"}'

# Simulator
curl http://localhost:8000/api/simulate/templates
curl -X POST http://localhost:8000/api/simulate/predict \
  -H "Content-Type: application/json" \
  -d '{"policy_name": "Morocco AI Transparency Act", "description": "Mandatory disclosure requirements for AI systems used in public services", "sectors": ["public_services", "healthcare"], "policy_type": "sectoral"}'
```

Expected: All endpoints return valid JSON with data. Concept Q&A returns Morocco-contextualized answer. Simulator returns impact dimensions with evidence.

---

## PHASE 5: React Frontend

### Objective
Build the complete frontend with all 3 components: Concept Chat, Case Study Browser, Impact Simulator Dashboard.

> **Claude Code: Use the EXACT frontend implementation from STEP-6-REACT-FRONTEND.md.**
> It contains complete code for:
>
> 1. Project setup: `npm create vite@latest frontend -- --template react`, install dependencies
> 2. `src/api.js` — Centralized API client with all endpoints
> 3. `src/App.jsx` — Router with 4 routes: /, /concepts, /case-studies, /simulator
> 4. `src/components/Layout.jsx` — Header + navigation + main container
> 5. **Concept Simulator**: ConceptChat.jsx (chat interface), MessageBubble.jsx, ConceptCard.jsx
> 6. **Case Study Library**: CaseStudyBrowser.jsx (search + grid), CaseStudyCard.jsx, CaseStudyDetail.jsx (modal)
> 7. **Impact Simulator**: ImpactSimulator.jsx (dashboard), PolicyInput.jsx (form), ImpactReport.jsx (results)
> 8. Home page with 3 feature cards
>
> Dependencies: `axios react-router-dom lucide-react react-markdown`
> Styling: Tailwind CSS only
> API base URL: `http://localhost:8000`

### 🚨 HUMAN ACTION REQUIRED
```bash
cd policybridge/frontend
npm install
npm run dev
```

### ✅ VERIFY
1. Open http://localhost:5173 — home page with 3 feature cards
2. Click "Concept Simulator" — ask "What is algorithmic bias?" → get Morocco-contextualized answer
3. Click "Case Studies" — see 8 cards, search "transparency" → EU AI Act appears
4. Click "Simulator" — select a template, run prediction → see impact report with dimensions

---

## PHASE 6: Integration, Tests, Deploy

### File: `backend/tests/test_integration.py`

> **Claude Code: Use the test suite from STEP-7-INTEGRATION-AND-DEPLOY.md**, which covers:
> - Health check returns correct collection counts
> - Concept list returns 10 items
> - Concept Q&A returns answer with related concepts
> - Case study list returns 8 items
> - Case study search returns relevant results
> - Case study compare works
> - Simulator templates return 5 templates
> - Simulator predict returns impact dimensions

Run with: `cd backend && python -m pytest tests/ -v`

### File: `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY backend/app ./app
COPY backend/scripts ./scripts

# Create storage directory
RUN mkdir -p /app/storage/chromadb

# Note: Ollama must be running on the host or accessible via network
# Set OLLAMA_BASE_URL env var if Ollama is not on localhost
# ENV OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
# RUN python -m scripts.load_data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### File: `README.md`

> **Claude Code: Generate a README covering:**
> - What PolicyBridge does (3 components, Morocco-focused)
> - Tech stack (FastAPI, BGE-M3, ChromaDB, hybrid search, React)
> - Quick start (5 steps: clone, install, env, load data, run)
> - API endpoints table
> - How to add new policy PDFs (put in pdfs/, update manifest, re-run load_data)
> - Architecture diagram (ASCII)
> - Team credits

### Pre-Demo Checklist

```bash
# 0. Ensure Ollama is running with Mistral
ollama list                          # Should show 'mistral'
# If not running: ollama serve &

# 1. Fresh data load
cd backend && python -m scripts.load_data

# 2. Health check
curl http://localhost:8000/health
# Verify: concepts > 0, case_studies > 0

# 3. Backend running
uvicorn app.main:app --port 8000

# 4. Frontend running
cd frontend && npm run dev

# 5. Full demo flow
# → Home page → Concept Simulator → Ask "What is AI governance?"
# → Case Studies → Browse → Click EU AI Act → See full outcomes
# → Impact Simulator → Select template → Run → See prediction report

# Note: First LLM call may take 5-10s (Ollama cold start). Subsequent calls are fast.
```

---

## Quick Reference: File Dependency Graph

```
config.py ──────────────────────────────────────────────┐
                                                         │
pdf_extractor.py (standalone)                            │
                                                         │
embedding_service.py (standalone) ──────┐                │
                                        ├─→ search_service.py
vectordb_service.py (needs config) ─────┘                │
                                                         │
llm_service.py (needs config, Ollama running) ──────────┤
                                                         │
scripts/load_data.py (needs all services above) ─────────┤
                                                         │
rag_service.py (needs embedding, vectordb, search, llm) ─┤
case_study_service.py (needs embedding, vectordb, search)─┤
impact_service.py (needs case_study_service, llm) ────────┤
                                                         │
routes/concepts.py (needs rag_service) ──────────────────┤
routes/case_studies.py (needs case_study_service) ───────┤
routes/simulator.py (needs impact_service) ──────────────┤
                                                         │
main.py (registers all routes, lifespan, CORS) ──────────┘
```

---

## Summary: Human Actions (Complete List)

| # | When | Action | Time |
|---|------|--------|------|
| 1 | Phase 0 | Install Ollama: `curl -fsSL https://ollama.com/install.sh \| sh` | 1 min |
| 2 | Phase 0 | Pull model: `ollama pull mistral` | 2-5 min (4.1GB download) |
| 3 | Phase 0 | Run `pip install -r requirements.txt` | 2-5 min |
| 4 | Phase 1 | Place PDF files in `backend/app/data/pdfs/` | 1-5 min |
| 5 | Phase 1 | Fill `pdf_manifest.json` with metadata per PDF | 2-10 min |
| 6 | Phase 3 | Run `python -m scripts.load_data` | 1-5 min |
| 7 | Phase 5 | Run `npm install` in frontend/ | 1-2 min |
| 8 | Phase 5 | Run `npm run dev` to start frontend | 5s |

**Total human time: ~10-30 minutes.** Everything else is automated.
**Total cost: $0.** No API keys, no accounts, no billing.

> **Note for Claude Code**: If the STEP-1 through STEP-7 markdown files are available in the project, reference them for the detailed JSON data and frontend component code. If they are NOT available, generate all content following the schemas and patterns described in this guide. The schemas above are authoritative — do not deviate from them.
>
> **CRITICAL**: The original STEP files reference `gpt-4o-mini` and `OPENAI_API_KEY`. **Ignore those references.** This project uses **Ollama + Mistral 7B** exclusively. When adapting code from STEP files, replace all OpenAI references with the Ollama config from this guide. The `openai` Python SDK is used purely as a client library for Ollama's compatible API — no OpenAI account or key is needed.
