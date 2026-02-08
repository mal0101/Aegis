
import chromadb
from concept_translator.backend.app.core.config import settings
from concept_translator.backend.app.services.embeddings import EmbeddingService

class VectirDBService:
    def __init__(self):
        self.client = chromadb.PersistentClient(settings.VECTOR_DB_PATH)
        self.collection = self.client.get_or_create_collection(name=settings.VECTOR_DB_COLLECTION)
        self.embedder = EmbeddingService()
        
    def add_concepts(self, concepts):
        for concept in concepts:
            text=f"{concept.get('term', '')}\n{concept.get('definition', '')}\n{concept.get('examples', '')}\n{concept.get('policy_relevance', '')}"
            embedding = self.embedder.encode(text)
            self.collection.upsert(
                ids=[concept['id']],
                embeddings=[embedding],
                documents=[text],
                metadatas=[concept]
            )
    
    def search(self, query, n_results=5):
        query_vec = self.embedder.encode(query)
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_results
        )
        return results
    
    
        