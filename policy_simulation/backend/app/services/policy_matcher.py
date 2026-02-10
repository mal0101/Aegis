import json
import logging
from pathlib import Path
from typing import List

from policy_simulation.backend.app.core.config import policy_settings
from policy_simulation.backend.app.schemas.policy_case import PolicyCase
from policy_simulation.backend.app.schemas.simulation_result import MatchedPolicy
from policy_simulation.backend.app.services.policy_vector_db import PolicyVectorDBService

logger = logging.getLogger(__name__)


class PolicyMatcher:
    def __init__(self, vector_db: PolicyVectorDBService):
        self.vector_db = vector_db
        self._policy_cache: dict = {}
        self._load_full_policies()

    def _load_full_policies(self):
        """Load all policy JSON files into memory for detail retrieval."""
        policies_dir = Path(policy_settings.POLICIES_DIR)
        for policy_file in policies_dir.glob("*.json"):
            with open(policy_file, "r") as f:
                data = json.load(f)
            self._policy_cache[data["id"]] = data

    def match(self, policy_description: str, n_results: int = 3) -> List[MatchedPolicy]:
        """Find the most similar reference policies to the given description."""
        results = self.vector_db.search(policy_description, n_results=n_results)

        matched = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for i, policy_id in enumerate(ids):
            # ChromaDB returns L2 distances by default; convert to similarity
            # For normalized vectors, similarity = 1 - (distance^2 / 2)
            distance = distances[i]
            similarity = max(0.0, 1.0 - (distance**2 / 2))

            meta = metadatas[i]
            matched.append(
                MatchedPolicy(
                    id=policy_id,
                    name=meta.get("name", ""),
                    jurisdiction=meta.get("jurisdiction", ""),
                    approach=meta.get("approach", ""),
                    similarity_score=round(similarity, 4),
                )
            )

        return matched

    def get_full_policy(self, policy_id: str) -> PolicyCase:
        """Get the full PolicyCase data for a matched policy."""
        data = self._policy_cache.get(policy_id)
        if data is None:
            raise KeyError(f"Policy not found: {policy_id}")
        return PolicyCase(**data)
