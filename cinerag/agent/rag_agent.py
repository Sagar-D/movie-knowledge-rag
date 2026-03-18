from langgraph.graph import StateGraph, START, END, add_messages
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool
from typing import List, Dict, Annotated, Optional
from cinerag.retrieval.hybrid_retriever import HybridRetriever
from cinerag.retrieval.qdrant_retriever import QdrantRetriever
from cinerag.llm.model_handler import get_chat_model
from cinerag.agent.rag_agent_prompts import (
    RAG_CHAT_TEMPLATE,
    RAG_QUERY_ENRICHMENT_PROMPT,
)
from cinerag import config
from typing import Optional, Literal
from pydantic import BaseModel, Field



class RAGAgentState(BaseModel):
    query: str
    retrieved_docs: List[Document]
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    context: str
    has_context: bool


class RetrievalToolInput(BaseModel):
    enriched_query: str = Field(
        ...,
        description="The original user query enriched with any additional information extracted from the query, such as movie titles, years, directors, etc. This is the query that will be used for retrieval.",
    )
    title: Optional[str] = Field(
        None,
        description="If the query is related to a specific movie, extract the movie title and provide it here. Otherwise, leave it empty.",
    )
    year: Optional[int] = Field(
        None,
        description="If the query is related to a specific movie, extract the movie release year and provide it here. Otherwise, leave it empty.",
    )
    genre: Optional[Literal["drama", "comedy", "horror", "action"]] = Field(
        None,
        description="If user has requested for a specific genre, pass the value here. If not leave it empty.",
    )


@tool("context_retrieval_tool", args_schema=RetrievalToolInput)
def retrieval_tool(
    enriched_query: str,
    title: Optional[str] = None,
    year: Optional[int] = None,
    genre: Optional[str] = None,
) -> dict:
    """This tool is responsible for retrieving relevant context documents based on the enriched user query and any extracted metadata filters. \
It uses a hybrid retrieval approach combining BM25 and vector-based retrieval to fetch the most relevant documents from the knowledge base."""

    retriever = QdrantRetriever()
    filters = {
        "title": title,
        "year": year,
        "genre": genre,
    }
    filters = {k: v for k, v in filters.items() if v is not None}
    docs = retriever.retrieve_docs(
        enriched_query, metadata_filters=filters, k=config.RETRIEVAL_K
    )
    has_context = len(docs) > 0
    context = "\n\n".join([doc.page_content for doc in docs]) if has_context else ""

    return {
        "context": context,
        "retrieved_docs": docs,
        "has_context": has_context,
    }


class RAGAgent:

    def __init__(self):
        self.build_graph()
        self.chat_model = get_chat_model()

    def enrich_rag_filter(self, state: RAGAgentState) -> RAGAgentState:

        enrichment_chain = RAG_QUERY_ENRICHMENT_PROMPT | self.chat_model.bind_tools(
            [retrieval_tool]
        )
        response = enrichment_chain.invoke({"messages": state.messages})
        print(f"Enrichment Response : {response.tool_calls}")
        return {"messages": [response]}

    def rag_tool_node(self, state: RAGAgentState) -> RAGAgentState:

        tool_calls = state.messages[-1].tool_calls
        for tool_call in tool_calls:
            if tool_call["name"] == "context_retrieval_tool":
                rag_tool_response = retrieval_tool.invoke(tool_call["args"])
                tool_message = ToolMessage(
                    content=rag_tool_response["context"], tool_call_id=tool_call["id"]
                )
                return {"messages": [tool_message], **rag_tool_response}

    def should_initiate_llm(self, state: RAGAgentState) -> bool:
        if state.has_context:
            return True
        return False

    def no_context_handler(self, state: RAGAgentState) -> RAGAgentState:
        response = (
            "Sorry, I couldn't find any relevant information to answer your query."
        )
        return {"messages": [AIMessage(content=response)]}

    def rag_chat(self, state: RAGAgentState) -> RAGAgentState:

        llm_chain = RAG_CHAT_TEMPLATE | self.chat_model
        response = llm_chain.invoke(
            {
                "input": state.query,
                "context": state.context,
                "messages": state.messages[:-1],
            }
        )

        return {"messages": [response]}

    def build_graph(self):

        graph = StateGraph(RAGAgentState)
        graph.add_node("enrich_rag_filter", self.enrich_rag_filter)
        graph.add_node("rag_tool_node", self.rag_tool_node)
        graph.add_node("rag_chat", self.rag_chat)
        graph.add_node("no_context_handler", self.no_context_handler)

        graph.add_edge(START, "enrich_rag_filter")
        graph.add_edge("enrich_rag_filter", "rag_tool_node")
        graph.add_conditional_edges(
            "rag_tool_node",
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

        agent_state = RAGAgentState(**state)
        agent_state.messages.append(HumanMessage(content=agent_state.query))
        return self.agent.invoke(agent_state)
