from langchain_community.retrievers import BM25Retriever as LC_BM25Retriever
from langchain_core.documents import Document
from cinerag.storage import s3_client
from cinerag.documents.helper import generate_movie_doc_id
import json
import logging


class BM25Retriever:

    _instance = None

    def __new__(cls):

        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):

        self.documents = self._load_documents()
        self.retriever = LC_BM25Retriever.from_documents(self.documents)
        logging.info("Created BM25 Index for %s documents", len(self.documents))

    def _load_documents(self):

        logging.info("Loading documents for BM25 Indexing")
        s3_data_obj = s3_client.get_processed_data_stream()
        documents = []

        for line in s3_data_obj["Body"].iter_lines():
            record = json.loads(line.decode("utf-8"))
            documents.append(
                Document(
                    page_content=record["text"],
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
        logging.info("Loaded %s documents for BM25 Indexing", len(documents))
        return documents

    def retrieve_docs(self, query: str, k: int = 5):

        self.retriever.k = k
        return self.retriever.invoke(query)
