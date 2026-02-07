from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional, Literal, ClassVar
from pathlib import Path

class Settings(BaseSettings):
    LLM_PROVIDER: Literal["groq"] = Field(
        default="groq",
        description="LLM provider to use. Currently only 'groq' is supported."
    )
    GROQ_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for Groq. Required if LLM_PROVIDER is set to 'groq'."
    )
    LLM_MODEL: Optional[str] = Field(
        default=None,
        description="Model name for the LLM provider. Required if LLM_PROVIDER is set to 'groq'."
    )
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model name for generating embeddings. Default is 'sentence-transformers/all-MiniLM-L6-v2'."
    )
    EMBEDDING_DIMENSION: int = Field(
        default=384,
        description="Dimension of the embedding vectors. Default is 384 for 'sentence-transformers/all-MiniLM-L6-v2'."
    )
    
    VECTOR_DB_TYPE: Literal["chromadb"] = Field(
        default="chromadb",
        description="Type of vector database to use. Currently only 'chromadb' is supported."
    )
    CHROMADB_PATH: str = Field(
        default="./storage/chromadb",
        description="File path for storing ChromaDB data. Default is './storage/chromadb'."
    )
    
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="PostgreSQL database URL. Required if using a relational database for metadata storage."
    )
    
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    LOG_FILE: str = Field(
        default="./storage/logs/app.log",
        description="Path to log file"
    )
    
    CONCEPTS_FILE: str = Field(
        default="./data/concepts/concepts.json",
        description="Path to concepts JSON file"
    )
    
    CONCEPTS_SCHEMA_FILE: str = Field(
        default="./data/concepts/concepts_schema.json",
        description="Path to concepts JSON schema for validation"
    )
    
model_config = {
    "env_file": str(Path(__file__).parent.parent / ".env"),
    "env_file_encoding": "utf-8",
    "case_sensitive": True
}
 
@field_validator("GROQ_API_KEY")
@classmethod
def validate_groq_api_key(cls, value, info):
    if info.data.get("LLM_PROVIDER") == "groq" and not value:
        raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER is set to 'groq'")
    return value

@field_validator("EMBEDDING_DIMENSION")
@classmethod
def validate_embedding_dimension(cls, value, info):
    model = info.data.get("EMBEDDING_MODEL", "")
    model_dimensions = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
    }
    if model in model_dimensions:
        expected_dimension = model_dimensions[model]
        if value != expected_dimension:
            raise ValueError(f"EMBEDDING_DIMENSION must be {expected_dimension} for model {model}")
    return value

def get_llm_config(self) -> dict:
    return {
        "provider": self.LLM_PROVIDER,
        "model": self.LLM_MODEL,
        "api_key": self.GROQ_API_KEY if self.LLM_PROVIDER == "groq" else None
    }

def get_embedding_config(self) -> dict:
    """Get embedding configuration as a dictionary."""
    return {
        "model": self.EMBEDDING_MODEL,
        "dimension": self.EMBEDDING_DIMENSION,
    }
        
settings = Settings()
