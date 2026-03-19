from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, AIMessage
from cinerag.api.schemas import ChatRequest, ChatResponse
from cinerag.agent.rag_agent import RAGAgent

router = APIRouter(prefix="/chat", tags=["chat"])
agent = RAGAgent()

_role_map = {"human": HumanMessage, "ai": AIMessage}


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        messages = [_role_map[m.role](content=m.content) for m in request.history]
        state = {
            "query": request.query,
            "messages": messages,
        }
        result = agent.invoke(state)
        return ChatResponse(
            answer=result["messages"][-1].content,
            enriched_query=result.get("enriched_query"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
