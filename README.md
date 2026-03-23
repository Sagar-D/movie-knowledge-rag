# CineRAG 🎬

A movie knowledge RAG (Retrieval-Augmented Generation) chatbot that answers queries about movies using a hybrid retrieval pipeline and LLM-powered responses.

## Architecture

```
User → Nginx (port 80)
         ├── / → Next.js Frontend (port 3000)
         └── /api → FastAPI Backend (port 8000)
                        ├── Query Enrichment  → Gemini 2.5 Flash
                        ├── Vector Retrieval  → Qdrant
                        └── Chat Response     → Amazon Nova Lite (Bedrock)
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16, Tailwind CSS, TypeScript |
| Backend | FastAPI, LangGraph, LangChain |
| LLM (Chat) | Amazon Bedrock — `amazon.nova-lite-v1:0` |
| LLM (Query Enrichment) | Google Gemini — `gemini-2.5-flash-lite` |
| Embeddings | HuggingFace — `sentence-transformers/all-mpnet-base-v2` |
| Vector Store | Qdrant |
| Retrieval | Vector search / Hybrid (BM25 + Vector + Cross-encoder reranking) |
| Data Storage | AWS S3 |
| Deployment | EC2 t3.small + Docker Compose + Nginx |

---

## Project Structure

```
movie-knowledge-rag/
├── cinerag/
│   ├── agent/              # LangGraph RAG agent + prompts
│   ├── api/                # FastAPI app, router, schemas
│   ├── data/               # Data preparation pipeline
│   ├── embeddings/         # HuggingFace embedding generation
│   ├── llm/                # Bedrock + Gemini model handlers
│   ├── pipelines/          # Index building pipeline
│   ├── retrieval/          # BM25, Qdrant, Hybrid retrievers
│   ├── storage/            # S3 client
│   ├── vector_store/       # Qdrant vector store
│   └── config.py           # Centralized configuration
├── cinerag-ui/             # Next.js frontend
├── nginx/                  # Nginx reverse proxy config
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
└── entrypoint.sh
```

---

## RAG Pipeline

1. **Query Enrichment** — Gemini extracts metadata filters (title, year, genre) and rewrites the query for better retrieval
2. **Retrieval** — Vector search via Qdrant (or hybrid BM25 + vector with cross-encoder reranking)
3. **Response Generation** — Amazon Nova Lite answers strictly based on retrieved context via Bedrock

---

## Local Development

### Prerequisites
- Python 3.12+
- Node.js 20+
- Docker + Docker Compose
- AWS credentials with Bedrock + S3 access
- Google API key

### Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start Qdrant locally
docker run -p 6333:6333 qdrant/qdrant

# Build the vector index
python -m cinerag.pipelines.build_index

# Start the API server
uvicorn cinerag.api.app:app --reload
```

### Frontend

```bash
cd cinerag-ui
npm install
cp .env.local.example .env.local  # set NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

---

## Environment Variables

### Backend
| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Google Gemini API key |
| `QDRANT_HOST` | Qdrant host (default: `localhost`) |
| `QDRANT_PORT` | Qdrant port (default: `6333`) |
| `EMBEDDING_BATCH_SIZE` | Batch size for indexing (default: `16`) |

AWS credentials are sourced from the EC2 IAM role — no hardcoded keys needed.

### Frontend
| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_API_URL` | Backend API URL |

---

## Deployment (AWS EC2)

### Prerequisites
- EC2 t3.small, Amazon Linux 2023, 20GB gp3
- IAM role with `AmazonBedrockFullAccess` + `AmazonS3ReadOnlyAccess`
- Security group: port `80` open, port `6333` restricted to your IP

### Steps

**1. Install dependencies on EC2**
```bash
sudo yum update -y && sudo yum install docker git -y
sudo service docker start && sudo usermod -aG docker ec2-user
sudo systemctl enable docker
sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

**2. Clone and deploy**
```bash
git clone https://github.com/Sagar-D/movie-knowledge-rag.git && cd movie-knowledge-rag
echo "GOOGLE_API_KEY=<your_key>" > .env
docker-compose up -d --build
```

**3. Build vector index (run once from local machine)**
```bash
export QDRANT_HOST=<ec2-public-dns>
export QDRANT_PORT=6333
export EMBEDDING_BATCH_SIZE=16
python -m cinerag.pipelines.build_index
```

> ⚠️ Open port `6333` in the security group only during indexing, then close it.

**4. Access the app**
```
http://<ec2-public-dns>
```

---

## API Reference

### `POST /chat`
Standard request-response chat.

```json
{
  "query": "What are some movies by Christopher Nolan?",
  "history": [
    {"role": "human", "content": "..."},
    {"role": "ai", "content": "..."}
  ]
}
```

### `POST /chat/stream`
Server-sent events streaming response. Each event:
```
data: {"token": "..."}
data: [DONE]
```

Interactive API docs available at `http://<host>/api/docs`.
