import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Data files
CONCEPTS_FILE = DATA_DIR / "concepts.json"
CASE_STUDIES_FILE = DATA_DIR / "case_studies.json"
MOROCCO_CONTEXT_FILE = DATA_DIR / "morocco_context.json"

# LLM (Groq API — OpenAI-compatible)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# Search
SEARCH_TOP_K = 5

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
