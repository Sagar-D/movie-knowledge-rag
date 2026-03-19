export type Role = "human" | "ai";

export interface Message {
  role: Role;
  content: string;
}

export interface ChatRequest {
  query: string;
  history: Message[];
}

export interface ChatResponse {
  answer: string;
  enriched_query?: string;
}
