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
            logger.info(f"Loaded BGE-M3 (dim={self.dim})")
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
            logger.info(f"Loaded MiniLM fallback (dim={self.dim})")
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
            return output["dense_vecs"].tolist() if hasattr(output["dense_vecs"], "tolist") else output["dense_vecs"]
        else:
            embeddings = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
            return embeddings.tolist()

    def get_collection_name(self, base_name: str) -> str:
        """Return collection name with dimension suffix: e.g., concepts_1024"""
        return f"{base_name}_{self.dim}"


# Singleton
embedding_service = EmbeddingService()
