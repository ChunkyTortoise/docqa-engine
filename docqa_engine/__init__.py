"""docqa-engine: RAG document Q&A with prompt engineering lab."""

__version__ = "0.1.0"

from docqa_engine.batch import BatchProcessor, BatchResult, QueryResult
from docqa_engine.chunking import Chunk, Chunker, ChunkingComparison, ChunkingResult
from docqa_engine.citation_scorer import CitationReport, CitationScore, CitationScorer
from docqa_engine.evaluator import Evaluator
from docqa_engine.exporter import Exporter
from docqa_engine.vector_store import (
    ChromaVectorStore,
    PineconeVectorStore,
    VectorStore,
    create_vector_store,
)

__all__ = [
    "BatchProcessor",
    "BatchResult",
    "Chunk",
    "Chunker",
    "ChunkingComparison",
    "ChunkingResult",
    "CitationReport",
    "CitationScore",
    "CitationScorer",
    "ChromaVectorStore",
    "Evaluator",
    "Exporter",
    "PineconeVectorStore",
    "QueryResult",
    "VectorStore",
    "create_vector_store",
]
