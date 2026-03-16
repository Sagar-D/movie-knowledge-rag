from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, Distance
from typing import List, Dict
from cinerag import config
import hashlib
from uuid import UUID

client = AsyncQdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


class VectorStore:

    def __init__(self):
        self.client = client

        if not self.client.collection_exists(
            collection_name=config.VECTOR_COLLECTION_NAME
        ):
            self.client.create_collection(
                collection_name=config.VECTOR_COLLECTION_NAME,
                vectors_config={"size": 768, "distance": Distance.COSINE},
            )

    async def store_embeddings(self, docs: List[Dict], embeddings: List):

        points = []
        for doc, embedding in zip(docs, embeddings):
            points.append(
                PointStruct(
                    id=movie_id(doc["title"], doc["year"], doc["director"]),
                    vector=embedding,
                    payload=doc,
                )
            )
        await client.upsert(collection_name=config.VECTOR_COLLECTION_NAME, points=points)


def movie_id(title: str, year: int, director: list[str] | str) -> str:

    if type(director) == list:
        director = ",".join(director)
    key = f"{title}_{year}_{director}"
    hash_bytes = hashlib.sha256(key.encode()).digest()
    return str(UUID(bytes=hash_bytes[:16]))
