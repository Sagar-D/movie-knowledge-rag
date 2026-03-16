from cinerag.storage import s3_client
from cinerag.embeddings import embedder
from cinerag import config
from cinerag.vector_store.qdrant_store import VectorStore
import json


async def stream_document_embeddings():

    s3_data_obj = s3_client.get_processed_data_stream()
    batch_docs = []

    for line in s3_data_obj["Body"].iter_lines():

        doc = json.loads(line.decode("utf-8"))
        batch_docs.append(doc)
        if len(batch_docs) == config.EMBEDDING_BATCH_SIZE:
            embeddings = await embedder.generate_embeddings(
                [data["text"] for data in batch_docs]
            )
            yield batch_docs, embeddings
            batch_docs = []

    if batch_docs:
        embeddings = await embedder.generate_embeddings(
            [data["text"] for data in batch_docs]
        )
        yield batch_docs, embeddings


async def build_rag_index():

    vector_store = VectorStore()
    async for docs, embeddings in stream_document_embeddings():
        docs = [{"text": record["text"], **record["metadata"]} for record in docs]
        await vector_store.store_embeddings(docs=docs, embeddings=embeddings)
