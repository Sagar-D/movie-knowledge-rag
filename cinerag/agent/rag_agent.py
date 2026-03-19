from langgraph.graph import StateGraph, START, END, add_messages
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import List, Dict, Annotated, Optional
from cinerag.retrieval.hybrid_retriever import HybridRetriever
from cinerag.retrieval.qdrant_retriever import QdrantRetriever
from cinerag.llm.model_handler import get_chat_model, get_query_enrichment_model
from cinerag.agent.rag_agent_prompts import (
    RAG_CHAT_TEMPLATE,
    RAG_QUERY_ENRICHMENT_PROMPT,
)
from cinerag import config
from typing import Optional, Literal
from pydantic import BaseModel, Field
import logging


class RAGAgentState(BaseModel):
    query: str
    retrieved_docs: List[Document]
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    context: str
    has_context: bool
    enriched_query: str
    rag_filters: Dict


class RetrievalMetadataFilters(BaseModel):
    year: Optional[int] = Field(
        None,
        description="If the query is related to a specific movie, extract the movie release year and provide it here. Otherwise, leave it empty.",
    )
    genre: Optional[Literal["drama", "comedy", "horror", "action"]] = Field(
        None,
        description="If user has requested for a specific genre, pass the value here. If not leave it empty.",
    )


class RetrievalConfig(BaseModel):
    """JSON fomrated filters and enriched query for retrieving context based on enriched query and filters"""

    enriched_query: str = Field(
        ...,
        description="The original user query enriched with any additional information extracted from the query, such as movie titles, years, directors, etc. This is the query that will be used for retrieval.",
    )
    filters: RetrievalMetadataFilters = Field(
        description="Metadata filters to be applied during retrieval. These filters help narrow down the search results based on specific criteria like movie title, year, or genre.",
        default_factory=dict,
    )


class RAGAgent:

    def __init__(self):
        self.build_graph()
        self.chat_model = get_chat_model()
        self.query_enrichment_model = get_query_enrichment_model()
        if config.RAG_RETRIEVAL_TYPE == "vector":
            self.retriever = QdrantRetriever()
        else:
            self.retriever = HybridRetriever()

    def enrich_rag_filter(self, state: RAGAgentState) -> RAGAgentState:

        enrichment_chain = (
            RAG_QUERY_ENRICHMENT_PROMPT
            | self.query_enrichment_model.with_structured_output(RetrievalConfig)
        )
        response: RetrievalConfig = enrichment_chain.invoke(
            {"messages": state.messages}
        )
        logging.info(f"Retrieval Config : {type(response)} - {response}")

        enriched_query = (
            response.enriched_query
            if response.enriched_query is not None
            and response.enriched_query.strip() != ""
            else state.query
        )

        filters = {
            k: v for k, v in response.filters.model_dump().items() if v is not None
        }
        return {"rag_filters": filters, "enriched_query": enriched_query}

    def fetch_context(self, state: RAGAgentState) -> RAGAgentState:

        docs = self.retriever.retrieve_docs(
            state.enriched_query,
            metadata_filters=state.rag_filters,
            k=config.RETRIEVAL_K,
        )
        has_context = len(docs) > 0
        context = "\n\n".join([doc.page_content for doc in docs]) if has_context else ""
        logging.info(f"Retrieved {len(docs)} documents for context building")
        return {"retrieved_docs": docs, "context": context, "has_context": has_context}

    def should_initiate_llm(self, state: RAGAgentState) -> bool:
        if state.has_context:
            return True
        return False

    def no_context_handler(self, state: RAGAgentState) -> RAGAgentState:
        logging.info(
            "Aborting execution as context is not available. User query - %s :: Enriched query - %s :: Metadata filters : %s",
            state.query,
            state.enriched_query,
            state.rag_filters,
        )
        response = (
            "Sorry, I couldn't find any relevant information to answer your query."
        )
        return {"messages": [AIMessage(content=response)]}

    def rag_chat(self, state: RAGAgentState) -> RAGAgentState:

        logging.info("Initiating LLM chat pipeline")
        llm_chain = RAG_CHAT_TEMPLATE | self.chat_model
        response = llm_chain.invoke(
            {
                "input": state.query,
                "context": state.context,
                "messages": state.messages[:-1],
            }
        )
        logging.info("LLM Response : %s", response.content)
        return {"messages": [response]}

    def build_graph(self):

        graph = StateGraph(RAGAgentState)
        graph.add_node("enrich_rag_filter", self.enrich_rag_filter)
        graph.add_node("fetch_context", self.fetch_context)
        graph.add_node("rag_chat", self.rag_chat)
        graph.add_node("no_context_handler", self.no_context_handler)

        graph.add_edge(START, "enrich_rag_filter")
        graph.add_edge("enrich_rag_filter", "fetch_context")
        graph.add_conditional_edges(
            "fetch_context",
            self.should_initiate_llm,
            {True: "rag_chat", False: "no_context_handler"},
        )
        graph.add_edge("rag_chat", END)
        graph.add_edge("no_context_handler", END)

        self.agent = graph.compile()

    def invoke(self, state: Dict):

        if str(state.get("query", "")).strip() == "":
            raise ValueError("Required field 'query' missing in the state")

        state["messages"] = state.get("messages", [])
        state["context"] = state.get("context", "")
        state["retrieved_docs"] = state.get("retrieved_docs", [])
        state["has_context"] = state.get("has_context", False)
        state["enriched_query"] = state.get("enriched_query", state.get("query"))
        state["rag_filters"] = state.get("rag_filters", {})

        agent_state = RAGAgentState(**state)
        agent_state.messages.append(HumanMessage(content=agent_state.query))
        return self.agent.invoke(agent_state)
