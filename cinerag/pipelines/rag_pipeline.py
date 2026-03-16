from cinerag.data.preperation import build_movie_rag_documents
from cinerag.storage import s3_client
from cinerag.embeddings import embedder
from cinerag import config
import pandas as pd
import json
import time
import asyncio


async def generate_embeddings_stream():

    print("Downloading data from s3)")
    s3_data_obj = s3_client.get_processed_data_stream()

    batch_data = []

    for line in s3_data_obj["Body"].iter_lines():

        print(f"Current batch size : {len(batch_data)}" )

        doc = json.loads(line.decode("utf-8"))
        batch_data.append(doc)

        if len(batch_data) == config.EMBEDDING_BATCH_SIZE:
            embeddings = await embedder.generate_embeddings([data["text"] for data in batch_data])
            yield batch_data, embeddings
            batch_data = []
            time.sleep(2)

    if batch_data:
        embeddings = await embedder.generate_embeddings([data["text"] for data in batch_data])
        yield batch_data, embeddings

async def runner():
    async for data, embeddings in generate_embeddings_stream():
        print(embeddings[0])

if __name__ == "__main__":
    asyncio.run(runner())
    


