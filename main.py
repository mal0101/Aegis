from fastapi import FastAPI

from policy_simulation.backend.app.api.routes.simulate import router as simulate_router

app = FastAPI(
    title="The Bridge",
    description="AI Policy Intelligence Platform for Morocco",
    version="0.1.0",
)


@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "The Bridge"}


# Policy simulation routes
app.include_router(simulate_router, prefix="/api/v1", tags=["policy-simulation"])

# Concept translator routes (mounted when teammate implements them)
# from concept_translator.backend.app.api.routes.concepts import router as concepts_router
# from concept_translator.backend.app.api.routes.chat import router as chat_router
# app.include_router(concepts_router, prefix="/api/v1/concepts", tags=["concept-translator"])
# app.include_router(chat_router, prefix="/api/v1/chat", tags=["concept-translator"])
