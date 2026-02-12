import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Rebuild BM25 indices on startup from ChromaDB data
    logger.info("Rebuilding BM25 indices...")
    from app.services.embedding_service import embedding_service
    from app.services.vectordb_service import vectordb_service
    from app.services.search_service import search_service

    for base_name in ["concepts", "case_studies", "policy_docs"]:
        col_name = embedding_service.get_collection_name(base_name)
        try:
            col = vectordb_service.get_or_create_collection(col_name)
            count = vectordb_service.get_collection_count(col)
            if count > 0:
                all_data = col.get(include=["documents", "metadatas"])
                bm25_docs = [
                    {"id": all_data["ids"][i], "text": all_data["documents"][i], "metadata": all_data["metadatas"][i]}
                    for i in range(len(all_data["ids"]))
                ]
                search_service.build_bm25_index(col_name, bm25_docs)
        except Exception as e:
            logger.warning(f"Could not rebuild BM25 for {col_name}: {e}")

    logger.info("BM25 indices ready")
    yield
    logger.info("Shutting down")


app = FastAPI(title="PolicyBridge API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
from app.routes.concepts import router as concepts_router
from app.routes.case_studies import router as case_studies_router
from app.routes.simulator import router as simulator_router

app.include_router(concepts_router)
app.include_router(case_studies_router)
app.include_router(simulator_router)


@app.get("/health")
async def health():
    from app.services.embedding_service import embedding_service
    from app.services.vectordb_service import vectordb_service

    counts = {}
    for base in ["concepts", "case_studies", "policy_docs"]:
        col_name = embedding_service.get_collection_name(base)
        try:
            col = vectordb_service.get_or_create_collection(col_name)
            counts[base] = vectordb_service.get_collection_count(col)
        except Exception:
            counts[base] = 0

    return {
        "status": "healthy",
        "embedding_model": embedding_service._model_name,
        "embedding_dim": embedding_service.dim,
        "collections": counts
    }
