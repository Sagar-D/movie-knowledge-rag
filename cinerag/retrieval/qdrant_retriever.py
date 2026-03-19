from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue

from cinerag.embeddings import embedder
from cinerag import config
from typing import Dict


class QdrantRetriever:

    def __init__(self):

        self.embedder = embedder
        self.vector_store = QdrantVectorStore.from_existing_collection(
            collection_name=config.VECTOR_COLLECTION_NAME,
            embedding=self.embedder.embedding_fn,
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
        )
        pass

    def retrieve_docs(self, query: str, metadata_filters: Dict = None, k: int = 5):
        if metadata_filters is not None:
            metadata_filters = self._generate_metadata_filter(metadata_filters)
            return self.vector_store.similarity_search(
                query, k=k, filter=metadata_filters
            )
        return self.vector_store.similarity_search(query, k=k)

    def _generate_metadata_filter(self, filters: Dict) -> Filter:

        if filters.get("title", None) is not None:
            filters["title"] = str(filters["title"]).lower().strip()
        if filters.get("genre", None) is not None:
            filters["genre"] = str(filters["genre"]).lower().strip()

        return Filter(
            must=[
                FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                for key, value in filters.items()
                if value is not None
            ]
        )
