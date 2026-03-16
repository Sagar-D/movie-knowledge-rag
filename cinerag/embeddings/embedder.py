from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from typing import List
from cinerag import config

embedder = HuggingFaceEmbeddings()


async def generate_embeddings(documents: List[str]) -> List:
    if len(documents) > config.EMBEDDING_BATCH_SIZE:
        raise ValueError(
            f"Number of documents exceeds the batch size limit of {config.EMBEDDING_BATCH_SIZE}"
        )

    embeddings = await embedder.aembed_documents(texts=documents)
    return embeddings
