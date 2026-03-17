from langgraph.graph import StateGraph, START, END, add_messages
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import List, Dict, Annotated
from cinerag.retrieval.hybrid_retriever import HybridRetriever
from cinerag.llm.model_handler import get_chat_model
from cinerag.llm.prompt_templates import RAG_CHAT_TEMPLATE


class RAGAgentState(BaseModel):
    query: str
    retrieved_docs: List[Document]
    messages: Annotated[List[BaseMessage], add_messages]
    context: str


class RAGAgent:

    def __init__(self):
        self.retriever = HybridRetriever()
        self.build_graph()

    def fetch_context(self, state: RAGAgentState) -> RAGAgentState:
        docs = self.retriever.retrieve_docs(state.query)
        if len(docs) == 0:
            context = "No relevant information found."
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"context": context, "retrieved_docs": docs}

    def rag_chat(self, state: RAGAgentState) -> RAGAgentState:
        model = get_chat_model()

        llm_chain = RAG_CHAT_TEMPLATE | model
        response = llm_chain.invoke(
            {"input": state.query, "context": state.context, "messages": state.messages}
        )

        return {"messages": [HumanMessage(content=state.query), response]}

    def build_graph(self):

        graph = StateGraph(RAGAgentState)
        graph.add_node("fetch_context", self.fetch_context)
        graph.add_node("rag_chat", self.rag_chat)
        
        graph.add_edge(START, "fetch_context")
        graph.add_edge("fetch_context", "rag_chat")
        graph.add_edge("llm_infrag_chaterence", END)

        self.agent = graph.compile()

    def invoke(self, state: Dict):

        if str(state.get("query", "")).strip() == "":
            raise ValueError("Required field 'query' missing in the state")

        agent_state = RAGAgentState(**state)
        return self.agent.invoke(agent_state)
