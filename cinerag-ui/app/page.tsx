"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Film, Loader2 } from "lucide-react";
import { Message, ChatResponse } from "./types";
import ReactMarkdown from "react-markdown";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [enrichedQuery, setEnrichedQuery] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const canSend = input.trim().length > 0 && !loading;

  async function sendMessage() {
    const query = input.trim();
    if (!canSend) return;

    const userMessage: Message = { role: "human", content: query };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput("");
    setLoading(true);
    setEnrichedQuery(null);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, history: messages }),
      });
      const data: ChatResponse = await res.json();
      setMessages([...updatedMessages, { role: "ai", content: data.answer }]);
      if (data.enriched_query) setEnrichedQuery(data.enriched_query);
    } catch {
      setMessages([...updatedMessages, { role: "ai", content: "Something went wrong. Please try again." }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-screen" style={{ backgroundColor: "var(--bg-primary)" }}>

      {/* Header */}
      <header className="flex items-center gap-3 px-6 py-4 border-b" style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-secondary)" }}>
        <Film size={22} style={{ color: "var(--gold)" }} />
        <span className="text-lg font-semibold tracking-wide" style={{ color: "var(--text-primary)" }}>CineRAG</span>
        <span className="text-xs px-2 py-0.5 rounded-full ml-1" style={{ backgroundColor: "#1e1e2e", color: "var(--text-muted)", border: "1px solid var(--border)" }}>
          Movie Knowledge Assistant
        </span>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto scrollbar-thin px-4 py-6 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
            <Film size={40} style={{ color: "var(--gold-muted)" }} />
            <p className="text-base font-medium" style={{ color: "var(--text-primary)" }}>Ask me anything about movies</p>
            <p className="text-sm" style={{ color: "var(--text-muted)" }}>Directors, cast, genres, release years — I've got you covered.</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "human" ? "justify-end" : "justify-start"}`}>
            <div
              className="max-w-[70%] px-4 py-3 rounded-xl text-sm leading-relaxed"
              style={
                msg.role === "human"
                  ? { backgroundColor: "var(--gold-muted)", color: "#fff" }
                  : { backgroundColor: "var(--bg-card)", color: "var(--text-primary)", border: "1px solid var(--border)" }
              }
            >
              {msg.role === "ai" ? (
                <ReactMarkdown
                  components={{
                    p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                    strong: ({ children }) => <strong className="font-semibold" style={{ color: "var(--gold)" }}>{children}</strong>,
                    ol: ({ children }) => <ol className="list-decimal list-inside space-y-2">{children}</ol>,
                    ul: ({ children }) => <ul className="list-disc list-inside space-y-1">{children}</ul>,
                    li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                  }}
                >
                  {msg.content}
                </ReactMarkdown>
              ) : (
                msg.content
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="px-4 py-3 rounded-xl text-sm flex items-center gap-2" style={{ backgroundColor: "var(--bg-card)", border: "1px solid var(--border)", color: "var(--text-muted)" }}>
              <Loader2 size={14} className="animate-spin" />
              Thinking...
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Enriched query hint */}
      {enrichedQuery && (
        <div className="px-6 py-1.5 text-xs" style={{ color: "var(--text-muted)", backgroundColor: "var(--bg-secondary)", borderTop: "1px solid var(--border)" }}>
          Searched for: <span style={{ color: "var(--gold)" }}>{enrichedQuery}</span>
        </div>
      )}

      {/* Input */}
      <div className="px-4 py-4" style={{ backgroundColor: "var(--bg-secondary)", borderTop: "1px solid var(--border)" }}>
        <div className="flex items-center gap-3 max-w-3xl mx-auto rounded-xl px-4 py-2" style={{ backgroundColor: "var(--bg-card)", border: "1px solid var(--border)" }}>
          <input
            className="flex-1 bg-transparent text-sm outline-none placeholder-shown:text-gray-500"
            style={{ color: "var(--text-primary)" }}
            placeholder="Ask about a movie, director, genre..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage()}
            disabled={loading}
          />
          <button
            onClick={sendMessage}
            disabled={!canSend}
            className="p-1.5 rounded-lg transition-opacity"
            style={{ backgroundColor: "var(--gold)", color: "#000", opacity: canSend ? 1 : 0.3, cursor: canSend ? "pointer" : "not-allowed" }}
          >
            <Send size={15} />
          </button>
        </div>
      </div>
    </div>
  );
}
