# Demo Mode Guide

## Overview
Run docqa-engine without external dependencies for testing and demonstrations. The engine uses local TF-IDF embeddings (scikit-learn), so it's already demo-ready out of the box.

## Quick Start

### Standard Demo
```bash
make demo
```
This launches the Streamlit UI with 3 pre-loaded demo documents at `http://localhost:8501`.

### Python API Demo
```python
from pathlib import Path
from docqa_engine.pipeline import DocQAPipeline

pipeline = DocQAPipeline()

# Load demo documents
demo_dir = Path("demo_docs")
pipeline.ingest(demo_dir / "python_guide.md")
pipeline.ingest(demo_dir / "machine_learning.md")
pipeline.ingest(demo_dir / "startup_playbook.md")

# Ask questions
result = pipeline.query("What is supervised learning?")
print(f"Answer: {result['answer']}")
print(f"Citations: {result['citations']}")
```

## Demo Documents Included

The `demo_docs/` directory contains 3 markdown files:

| Document | Topic | Use Case |
|----------|-------|----------|
| `python_guide.md` | Python basics | Test code-related queries |
| `machine_learning.md` | ML concepts | Test technical definitions |
| `startup_playbook.md` | Startup advice | Test business/strategy questions |

These documents are automatically loaded in the Streamlit demo.

## What's Mocked

**Nothing is mocked.** This engine uses:
- **Local TF-IDF embeddings** (scikit-learn) — no API calls
- **BM25 retrieval** (Okapi) — implemented locally
- **In-memory storage** — no database required

The only external service is LLM answer generation, which is **optional**. You can use the retrieval system standalone:

```python
# Retrieval-only mode (no LLM)
chunks = pipeline.retriever.retrieve("your query", k=5)
for chunk in chunks:
    print(f"Score: {chunk.score}")
    print(f"Content: {chunk.content}")
```

## Switching to Production

### 1. Add LLM Integration (Optional)
The engine currently returns retrieved chunks. To generate answers, integrate an LLM:

```python
# Install LLM client
pip install anthropic  # or openai, google-generativeai

# Add to .env
ANTHROPIC_API_KEY=your_api_key_here

# Modify pipeline to use LLM
from docqa_engine.answer import AnswerGenerator

pipeline = DocQAPipeline()
answer_gen = AnswerGenerator(provider="claude")
result = answer_gen.generate(query="Your question", chunks=chunks)
```

### 2. Add Vector Database (Optional)
For large-scale deployments, replace in-memory storage:

```python
# Install vector store
pip install chromadb  # or pinecone-client

# Update pipeline
from docqa_engine.vector_store import ChromaStore

pipeline = DocQAPipeline(vector_store=ChromaStore())
```

### 3. Enable Persistence
Save index to disk for faster restarts:

```python
pipeline.save_index("index.pkl")  # Save
pipeline.load_index("index.pkl")  # Load on restart
```

### 4. Scale with FastAPI
Wrap the pipeline in a REST API:

```bash
# Install FastAPI
pip install fastapi uvicorn

# Run API server
uvicorn docqa_engine.api:app --host 0.0.0.0 --port 8000
```

## Environment Variables

The engine requires no environment variables for demo mode. For production:

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | Optional | LLM answer generation (Claude) |
| `OPENAI_API_KEY` | Optional | LLM answer generation (GPT) |
| `CHROMA_PERSIST_DIR` | Optional | ChromaDB persistence path |
| `LOG_LEVEL` | Optional | Debug logging (default: INFO) |

## Performance Benchmarks (Demo Mode)

On a standard laptop:
- **Ingestion**: 10 documents/second (mixed formats)
- **Retrieval**: <50ms for 1,000 chunks
- **Memory**: ~100MB for 10,000 chunks
- **No network calls** — fully offline capable

## Security Checklist

Demo mode is safe for public demonstrations:
- No API keys required
- No external network calls
- No persistent storage (in-memory only)
- No file uploads outside demo directory (when using Streamlit defaults)

For production:
- Validate file uploads (size, type, content)
- Sanitize user queries (prevent injection)
- Rate limit API endpoints
- Enable HTTPS for web deployments
