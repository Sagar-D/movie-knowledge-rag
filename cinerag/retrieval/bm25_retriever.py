from langchain_community.retrievers import BM25Retriever as LC_BM25Retriever
from langchain_core.documents import Document
from cinerag.storage import s3_client
from cinerag.documents.helper import generate_movie_doc_id
import json
import logging


is_bm25_index_built = False
bm25_documents = []


def build_bm25_idnde():
    global is_bm25_index_built
    global bm25_documents

    if is_bm25_index_built:
        return

    logging.info("Loading documents for BM25 Indexing")
    s3_data_obj = s3_client.get_processed_data_stream()
    for line in s3_data_obj["Body"].iter_lines():
        record = json.loads(line.decode("utf-8"))
        bm25_documents.append(
            Document(
                page_content=(str(record["text"]).lower().strip()),
                metadata={
                    "id": generate_movie_doc_id(
                        record["metadata"]["title"],
                        record["metadata"]["year"],
                        record["metadata"]["director"],
                    ),
                    "metadata": record["metadata"],
                },
            )
        )
    logging.info("Loaded %s documents for BM25 Indexing", len(bm25_documents))


class BM25Retriever:

    _instance = None

    def __new__(cls):

        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):

        global bm25_documents
        self.retriever = LC_BM25Retriever.from_documents(bm25_documents)
        logging.info("Created BM25 Index for %s documents", len(bm25_documents))

    def retrieve_docs(self, query: str, k: int = 5):

        self.retriever.k = k
        query = query.strip().lower()
        return self.retriever.invoke(query)
