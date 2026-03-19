import os

DEFAULT_S3_BUCKET = "cinerag-data"
S3_RAW_DATA_FOLDER = "raw/"
S3_PROCESSED_DATA_FOLDER = "processed/"
S3_EMBEDDINGS_FOLDER = "embeddings/"
DEFAULT_FILE_NAME = "movie_data"

EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 16))
EMBEDDING_DIMENSION = 768
VECTOR_COLLECTION_NAME = "cinerag_stage"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_THRESHOLD = 0.4

RETRIEVAL_K = 10

CHAT_MODEL_ID="amazon.nova-lite-v1:0"
QUERY_ENRICHMENT_MODEL_ID="gemini-2.5-flash-lite"
RAG_RETRIEVAL_TYPE="vector"