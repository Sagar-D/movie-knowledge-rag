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
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    context: str
    has_context: bool


class RAGAgent:

    def __init__(self):
        self.retriever = HybridRetriever()
        self.build_graph()
        self.chat_model = get_chat_model()

    def fetch_context(self, state: RAGAgentState) -> RAGAgentState:
        docs = self.retriever.retrieve_docs(state.query)
        has_context = len(docs) > 0
        context = "\n\n".join([doc.page_content for doc in docs]) if has_context else ""
        return {"context": context, "retrieved_docs": docs, "has_context": has_context}
    
    def should_initiate_llm(self, state: RAGAgentState) -> bool :
        if state.has_context:
            return True
        return False
    
    def no_context_handler(self, state: RAGAgentState) -> RAGAgentState :
        response = "Sorry, I couldn't find any relevant information to answer your query."
        return {"messages": [AIMessage(content=response)]}

    def rag_chat(self, state: RAGAgentState) -> RAGAgentState:

        llm_chain = RAG_CHAT_TEMPLATE | self.chat_model
        response = llm_chain.invoke(
            {"input": state.query, "context": state.context, "messages": state.messages[:-1]}
        )

        return {"messages": [response]}

    def build_graph(self):

        graph = StateGraph(RAGAgentState)
        graph.add_node("fetch_context", self.fetch_context)
        graph.add_node("rag_chat", self.rag_chat)
        graph.add_node("no_context_handler", self.no_context_handler)
        
        graph.add_edge(START, "fetch_context")
        graph.add_conditional_edges(
            "fetch_context",
            self.should_initiate_llm,
            {
                True: "rag_chat",
                False: "no_context_handler"
            }
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
