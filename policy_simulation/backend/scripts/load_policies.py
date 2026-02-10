#!/usr/bin/env python3
"""Load policy JSON files into ChromaDB for semantic search."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from policy_simulation.backend.app.services.policy_vector_db import PolicyVectorDBService


def main():
    print("Loading policies into ChromaDB...")
    db = PolicyVectorDBService()
    count = db.load_policies()
    print(f"Loaded {count} policies into ChromaDB.")
    print(f"Collection count: {db.get_collection_count()}")


if __name__ == "__main__":
    main()
