"""docqa-engine: RAG document Q&A with prompt engineering lab."""

__version__ = "0.1.0"

from docqa_engine.answer_quality import AnswerComparison, AnswerQualityScorer, QualityReport
from docqa_engine.batch import BatchProcessor, BatchResult, QueryResult
from docqa_engine.chunking import Chunk, Chunker, ChunkingComparison, ChunkingResult
from docqa_engine.citation_scorer import CitationReport, CitationScore, CitationScorer
from docqa_engine.conversation_manager import (
    ContextAwareExpander,
    ConversationContext,
    ConversationHistory,
    ConversationTurn,
)
from docqa_engine.document_graph import (
    DocumentNode,
    GraphBuilder,
    GraphRetriever,
    Relationship,
    RelationshipGraph,
)
from docqa_engine.evaluator import Evaluator
from docqa_engine.exporter import Exporter
from docqa_engine.multi_hop import MultiHopResult, MultiHopRetriever, RetrievedPassage
from docqa_engine.query_expansion import ExpandedQuery, QueryExpander
from docqa_engine.reranker import CrossEncoderReranker, RerankReport, RerankResult
from docqa_engine.summarizer import DocumentSummarizer, KeyPhrase, SummaryResult
from docqa_engine.benchmark_runner import BenchmarkRegistry, BenchmarkResult, BenchmarkSuite
from docqa_engine.context_compressor import (
    Budget,
    CompressedContext,
    ContextCompressor,
    TokenBudgetManager,
)
from docqa_engine.vector_store import (
    ChromaVectorStore,
    InMemoryVectorStore,
    PineconeVectorStore,
    VectorStore,
    create_vector_store,
)

__all__ = [
    "AnswerComparison",
    "AnswerQualityScorer",
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
    "CrossEncoderReranker",
    "Evaluator",
    "ExpandedQuery",
    "Exporter",
    "InMemoryVectorStore",
    "PineconeVectorStore",
    "QualityReport",
    "QueryExpander",
    "QueryResult",
    "RerankReport",
    "RerankResult",
    "VectorStore",
    "DocumentSummarizer",
    "KeyPhrase",
    "MultiHopResult",
    "MultiHopRetriever",
    "RetrievedPassage",
    "SummaryResult",
    "ContextAwareExpander",
    "ConversationContext",
    "ConversationHistory",
    "ConversationTurn",
    "DocumentNode",
    "GraphBuilder",
    "GraphRetriever",
    "Relationship",
    "RelationshipGraph",
    "create_vector_store",
    "BenchmarkRegistry",
    "BenchmarkResult",
    "BenchmarkSuite",
    "Budget",
    "CompressedContext",
    "ContextCompressor",
    "TokenBudgetManager",
]
