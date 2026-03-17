from langgraph.graph import StateGraph, START, END, add_messages
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict, Annotated
from cinerag.retrieval.hybrid_retriever import HybridRetriever
from cinerag.llm.model_handler import get_chat_model


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

    def llm_inference(self, state: RAGAgentState) -> RAGAgentState:
        model = get_chat_model()

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder("messages"),
                (
                    "system",
                    """
You are a strictly context-grounded assistant.

Your task is to respond to the user's message ONLY if it can be fully supported by the provided context.

MANDATORY RULES:
1. You MUST NOT use any internal, external, or prior knowledge.
2. Every part of your response MUST be directly supported by the provided context.
3. If the context does not clearly support a complete response, you MUST refuse.
4. Do NOT infer, assume, generalize, or fill in missing information.
5. Evidence must be taken verbatim or near-verbatim from the context.
""",
                ),
                ("user", "Message: {input}\n\nContext:\n{context}"),
            ]
        )
        llm_chain = chat_prompt_template | model
        response = llm_chain.invoke(
            {"input": state.query, "context": state.context, "messages": state.messages}
        )

        return {"messages": [HumanMessage(content=state.query), response]}

    def build_graph(self):

        graph = StateGraph(RAGAgentState)
        graph.add_node("rag_context", self.fetch_context)
        graph.add_node("llm_inference", self.llm_inference)
        graph.add_edge(START, "rag_context")
        graph.add_edge("rag_context", "llm_inference")
        graph.add_edge("llm_inference", END)

        self.agent = graph.compile()

    def invoke(self, state: Dict):

        if str(state.get("query", "")).strip() == "":
            raise ValueError("Required field 'query' missing in the state")

        agent_state = RAGAgentState(**state)
        return self.agent.invoke(agent_state)
