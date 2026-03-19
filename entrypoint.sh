#!/bin/bash
set -e

echo "Loading BM25 documents..."
python -c "from cinerag.retrieval.bm25_retriever import build_bm25_index; build_bm25_index()"

echo "Starting API server..."
exec uvicorn cinerag.api.app:app --host 0.0.0.0 --port 8000
