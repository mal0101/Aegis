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
    logger.info(f"Loaded {len(ids)} concepts into '{collection_name}'")


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
    logger.info(f"Loaded {len(ids)} case studies into '{collection_name}'")


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
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        all_embeddings.extend(embedding_service.embed_batch(batch))
        logger.info(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)} chunks")

    vectordb_service.add_documents(collection, ids, texts, all_embeddings, metadatas)
    search_service.build_bm25_index(collection_name, bm25_docs)
    logger.info(f"Loaded {len(ids)} PDF chunks into '{collection_name}'")
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
