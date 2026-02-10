import json
import logging
from pathlib import Path
from typing import Dict, List

import chromadb

from concept_translator.backend.app.services.embeddings import EmbeddingService
from policy_simulation.backend.app.core.config import policy_settings

logger = logging.getLogger(__name__)


class PolicyVectorDBService:
    def __init__(self):
        self.client = chromadb.PersistentClient(policy_settings.POLICY_CHROMADB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=policy_settings.POLICY_CHROMADB_COLLECTION,
        )
        self.embedder = EmbeddingService()

    def load_policies(self) -> int:
        """Load all policy JSON files from POLICIES_DIR into ChromaDB. Returns count loaded."""
        policies_dir = Path(policy_settings.POLICIES_DIR)
        if not policies_dir.exists():
            raise FileNotFoundError(f"Policies directory not found: {policies_dir}")

        count = 0
        for policy_file in sorted(policies_dir.glob("*.json")):
            with open(policy_file, "r") as f:
                policy = json.load(f)

            embedding_text = policy.get("embedding_text", "")
            if not embedding_text:
                logger.warning(f"Skipping {policy_file.name}: no embedding_text")
                continue

            embedding = self.embedder.encode(embedding_text)
            metadata = {
                "name": policy["name"],
                "jurisdiction": policy["jurisdiction"],
                "jurisdiction_iso": policy["jurisdiction_iso"],
                "approach": policy["approach"],
                "status": policy["status"],
            }

            self.collection.upsert(
                ids=[policy["id"]],
                embeddings=[embedding.tolist()],
                documents=[embedding_text],
                metadatas=[metadata],
            )
            count += 1
            logger.info(f"Loaded policy: {policy['name']}")

        return count

    def search(self, query: str, n_results: int = 3) -> Dict:
        """Semantic search over policy cases. Returns ChromaDB query results."""
        query_vec = self.embedder.encode(query)
        results = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=n_results,
        )
        return results

    def get_collection_count(self) -> int:
        return self.collection.count()
