from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np
from concept_translator.backend.app.core.config import settings
class EmbeddingService:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.embedding_dimension = settings.EMBEDDING_DIMENSION
        self.model = SentenceTransformer(self.model_name)
        
    def encode(self, text:str):
        if not text or not text.strip():
            raise ValueError("Can't Encode Empty Text")
        return self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    
    
    def encode_batch(self, texts: List[str]):
        if not texts:
            raise ValueError("Can't Encode Empty List of Texts")
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
    
    def get_similarity_from_text(self, text1, text2):
        vec1 = self.encode(text1)
        vec2 = self.encode(text2)
        return self.get_similarity_from_vecs(vec1, vec2)
    
    def get_similarity_from_vecs(self, vec1, vec2):
        if len(vec1) != self.embedding_dimension or len(vec2) != self.embedding_dimension:
            raise ValueError(f"Embedding vectors must be of dimension {self.embedding_dimension}")
        return float(np.dot(vec1, vec2))
    
    def get_similarities(self, query: str, texts: List[str]):
        query_vec = self.encode(query)
        text_vecs = self.encode_batch(texts)
        return [self.get_similarity_from_vecs(query_vec, vec) for vec in text_vecs]