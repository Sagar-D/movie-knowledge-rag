from langchain_qdrant import QdrantVectorStore
from cinerag.embeddings import embedder
from cinerag import config

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

    def retrieve_docs(self, query: str, k:int = 5) :
        return self.vector_store.similarity_search(query, k=k)



