from pydantic import BaseModel
from typing import Optional, Literal


class Message(BaseModel):
    role: Literal["human", "ai"]
    content: str


class ChatRequest(BaseModel):
    query: str
    history: list[Message] = []


class ChatResponse(BaseModel):
    answer: str
    enriched_query: Optional[str] = None
