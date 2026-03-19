from fastapi import FastAPI
from cinerag.api.router import router
from cinerag.logging_config import setup_logging

setup_logging()

app = FastAPI(title="CineRAG", description="Movie knowledge RAG chat API")
app.include_router(router)
