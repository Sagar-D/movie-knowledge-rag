from cinerag.retrieval.bm25_retriever import BM25Retriever
from cinerag.retrieval.qdrant_retriever import QdrantRetriever
from sentence_transformers import CrossEncoder
from cinerag import config

reranker = CrossEncoder(config.RERANKER_MODEL)


class HybridRetriever:

    def __init__(
        self, bm25_retriever: BM25Retriever, vector_retriever: QdrantRetriever
    ):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever

    def retrieve_docs(self, query: str, k: int = 5):

        candidate_docs = self.vector.retrieve_docs(query, max(20, 3 * k))
        unique_doc_ids = {
            [doc.metadata.get("id", doc.page_content) for doc in candidate_docs]
        }

        for doc in self.bm25.retrieve_docs(query, max(20, 2 * k)):
            doc_id = doc.metadata.get("id", doc.page_content)
            if doc_id not in unique_doc_ids:
                candidate_docs.append(doc)
                unique_doc_ids.add(doc_id)

        pairs = [(query, doc.page_content) for doc in candidate_docs[:50]]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
        filtered_documents = [
            doc for doc, score in ranked if score > config.RERANKER_THRESHOLD
        ]
        return filtered_documents[:k]
