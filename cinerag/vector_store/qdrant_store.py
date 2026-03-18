from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, Distance
from typing import List, Dict
from cinerag import config
from cinerag.documents.helper import generate_movie_doc_id


from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, Distance
from typing import List, Dict
from cinerag import config



class VectorStore:

    def __init__(self):
        self.client = AsyncQdrantClient(
            host=config.QDRANT_HOST, port=config.QDRANT_PORT
        )

    async def initialize(self):

        exists = await self.client.collection_exists(
            collection_name=config.VECTOR_COLLECTION_NAME
        )

        if not exists:

            await self.client.create_collection(
                collection_name=config.VECTOR_COLLECTION_NAME,
                vectors_config={
                    "size": config.EMBEDDING_DIMENSION,
                    "distance": Distance.COSINE,
                },
            )

    async def store_embeddings(self, docs: List[Dict], embeddings: List):

        points = [
            PointStruct(
                id=generate_movie_doc_id(
                    doc["metadata"]["title"],
                    doc["metadata"]["year"],
                    doc["metadata"]["director"],
                ),
                vector=embedding,
                payload=doc,
            )
            for doc, embedding in zip(docs, embeddings)
        ]

        await self.client.upsert(
            collection_name=config.VECTOR_COLLECTION_NAME, points=points
        )
