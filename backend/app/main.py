import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import CORS_ORIGINS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing search indices...")
    from app.services.search_service import search_service
    search_service.initialize()
    logger.info("Search indices ready")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Aegis API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
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
    from app.services.search_service import search_service

    return {
        "status": "healthy",
        "search_engine": "bm25",
        "collections": {
            "concepts": search_service.get_document_count("concepts"),
            "case_studies": search_service.get_document_count("case_studies"),
        },
    }
