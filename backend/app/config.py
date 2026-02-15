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

# LLM (Groq API — OpenAI-compatible)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
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
