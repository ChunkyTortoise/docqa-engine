# Customization Guide

## Quick Start (5 minutes)

### Environment Setup
```bash
git clone https://github.com/ChunkyTortoise/docqa-engine.git
cd docqa-engine
pip install -r requirements.txt
make demo
```

No API keys required. The engine uses local TF-IDF embeddings via scikit-learn, so you can start immediately without external dependencies.

### First Run Verification
```bash
make test  # Run all 157 tests
streamlit run app.py  # Launch UI at http://localhost:8501
```

Upload any document (PDF, DOCX, TXT, MD, CSV), ask a question, and verify citation accuracy scores appear.

## Common Customizations

### 1. Branding & UI
**Streamlit Configuration** (`app.py`, lines 1-20):
- Page title: `st.set_page_config(page_title="Your Company DocQA")`
- Logo: Add `page_icon="🏢"` or path to image file
- Theme colors: Create `.streamlit/config.toml` with `[theme]` section

**Session Timeout** (`app.py`, line 15):
- Default: 30 minutes of inactivity
- Change: `SESSION_TIMEOUT_SECONDS = 60 * 60  # 1 hour`

### 2. Chunking Strategies
**Chunker Configuration** (`docqa_engine/chunking.py`):
- Fixed-size: `Chunker(strategy="fixed", chunk_size=500, overlap=50)`
- Sentence-boundary: `Chunker(strategy="sentence", chunk_size=800)`
- Semantic: `Chunker(strategy="semantic", similarity_threshold=0.75)`

**Custom Chunking** (`chunking.py`, line 120):
Implement your own chunking logic by subclassing `Chunker` and overriding `_chunk_text()`.

### 3. Retrieval Configuration
**Hybrid Search Weights** (`docqa_engine/retriever.py`, line 85):
- BM25 vs Dense balance: `alpha=0.5` (0.0=pure BM25, 1.0=pure dense)
- Top-K results: `retriever.retrieve(query, k=5)` (default: 3)

**TF-IDF Features** (`docqa_engine/embedder.py`, line 25):
- Vocabulary size: `TfidfVectorizer(max_features=5000)` (default: 5000)
- N-grams: `ngram_range=(1, 2)` for bigrams

### 4. Citation Scoring
**Scoring Thresholds** (`docqa_engine/citation_scorer.py`, lines 40-50):
- Faithfulness: 0.8+ = high accuracy
- Coverage: 0.7+ = comprehensive
- Adjust thresholds in `score_citations()` method

**Custom Metrics**: Add new scoring dimensions by extending `CitationScorer` class.

## Advanced Features

### REST API Integration
The engine includes a FastAPI wrapper (`docqa_engine/api.py` if you add it):
```python
from fastapi import FastAPI
from docqa_engine.pipeline import DocQAPipeline

app = FastAPI()
pipeline = DocQAPipeline()

@app.post("/ingest")
async def ingest(file: UploadFile):
    content = await file.read()
    pipeline.ingest(content, filename=file.filename)
    return {"status": "success"}

@app.post("/query")
async def query(question: str):
    return pipeline.query(question)
```

### Batch Processing
**Parallel Ingestion** (`docqa_engine/batch.py`):
```python
from docqa_engine.batch import batch_ingest

files = [Path("doc1.pdf"), Path("doc2.docx")]
results = batch_ingest(files, max_workers=4)
```

**Batch Queries** (`batch.py`, line 80):
```python
from docqa_engine.batch import batch_query

questions = ["What is X?", "How does Y work?"]
answers = batch_query(pipeline, questions, parallel=True)
```

### Cost & Performance Tracking
**Enable Tracking** (`docqa_engine/pipeline.py`, line 45):
```python
pipeline = DocQAPipeline(enable_cost_tracking=True)
pipeline.query("Your question")
stats = pipeline.cost_tracker.get_total_cost()
```

**Export Metrics** (`docqa_engine/exporter.py`):
```python
from docqa_engine.exporter import export_metrics

export_metrics(pipeline, "metrics.csv", format="csv")
```

## Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Point to `app.py`
4. No environment variables needed (local embeddings)

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8080"]
```

```bash
docker build -t docqa-engine .
docker run -p 8080:8080 docqa-engine
```

## Troubleshooting

### Common Errors

**ImportError: No module named 'docqa_engine'**
- Fix: Run `pip install -e .` or `pip install -r requirements.txt`

**PDF parsing fails**
- Fix: Install `pdfplumber` for better PDF support: `pip install pdfplumber`
- Alternative: Use `PyMuPDF` for scanned documents: `pip install PyMuPDF`

**Low retrieval accuracy**
- Increase chunk overlap: `Chunker(overlap=100)` (default: 50)
- Switch to semantic chunking: `Chunker(strategy="semantic")`
- Increase TF-IDF features: `max_features=10000`

### Debug Mode
**Enable Detailed Logging** (`app.py`, line 1):
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Retrieval Diagnostics** (`docqa_engine/retriever.py`, line 120):
Set `verbose=True` in `retrieve()` to see BM25/dense scores for each chunk.

## Support Resources

- **GitHub Issues**: [docqa-engine/issues](https://github.com/ChunkyTortoise/docqa-engine/issues)
- **Documentation**: See module docstrings in `docqa_engine/` directory
- **Live Demo**: [ct-document-engine.streamlit.app](https://ct-document-engine.streamlit.app)
- **Portfolio**: [chunkytortoise.github.io](https://chunkytortoise.github.io)
