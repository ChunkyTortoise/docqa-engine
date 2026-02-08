"""docqa-engine: RAG document Q&A with prompt engineering lab."""

__version__ = "0.1.0"

from docqa_engine.batch import BatchProcessor, BatchResult, QueryResult
from docqa_engine.chunking import Chunk, Chunker, ChunkingComparison, ChunkingResult
from docqa_engine.citation_scorer import CitationReport, CitationScore, CitationScorer
from docqa_engine.evaluator import Evaluator
from docqa_engine.exporter import Exporter

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
    "Evaluator",
    "Exporter",
    "QueryResult",
]
