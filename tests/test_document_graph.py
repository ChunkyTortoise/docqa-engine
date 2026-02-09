"""Tests for document relationship graph: nodes, edges, traversal, building."""

from __future__ import annotations

import pytest

from docqa_engine.document_graph import (
    DocumentNode,
    GraphBuilder,
    GraphRetriever,
    Relationship,
    RelationshipGraph,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph() -> RelationshipGraph:
    return RelationshipGraph()


@pytest.fixture()
def populated_graph() -> RelationshipGraph:
    """Graph with 4 nodes and directed edges: A->B->C, A->D."""
    g = RelationshipGraph()
    g.add_node("A", "Python is a programming language", summary="Python intro")
    g.add_node("B", "Data science uses Python extensively", summary="Data science")
    g.add_node("C", "Machine learning builds on data science", summary="ML")
    g.add_node("D", "Web development with Python Flask", summary="Web dev")
    g.add_relationship("A", "B", "related", weight=0.9)
    g.add_relationship("B", "C", "expands", weight=0.8)
    g.add_relationship("A", "D", "related", weight=0.5)
    return g


@pytest.fixture()
def sample_chunks() -> list[dict]:
    return [
        {"chunk_id": "c1", "content": "Python is a versatile programming language used in data science"},
        {"chunk_id": "c2", "content": "Data science and machine learning use Python extensively"},
        {"chunk_id": "c3", "content": "Cooking recipes require fresh ingredients and careful preparation"},
        {"chunk_id": "c4", "content": "Python libraries like pandas help with data analysis"},
    ]


# ---------------------------------------------------------------------------
# DocumentNode and Relationship dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_document_node_fields(self) -> None:
        node = DocumentNode(chunk_id="x", content="hello", summary="s", metadata={"k": "v"})
        assert node.chunk_id == "x"
        assert node.content == "hello"
        assert node.summary == "s"
        assert node.metadata == {"k": "v"}
        assert node.embedding is None

    def test_relationship_fields(self) -> None:
        rel = Relationship(source_id="a", target_id="b", rel_type="cites", weight=0.7)
        assert rel.source_id == "a"
        assert rel.target_id == "b"
        assert rel.rel_type == "cites"
        assert rel.weight == 0.7

    def test_document_node_default_metadata(self) -> None:
        node = DocumentNode(chunk_id="x", content="c", summary="s")
        assert node.metadata == {}


# ---------------------------------------------------------------------------
# RelationshipGraph — node/edge CRUD
# ---------------------------------------------------------------------------


class TestGraphCRUD:
    def test_add_node(self, graph: RelationshipGraph) -> None:
        node = graph.add_node("n1", "content", summary="sum")
        assert isinstance(node, DocumentNode)
        assert graph.node_count == 1

    def test_add_multiple_nodes(self, graph: RelationshipGraph) -> None:
        graph.add_node("n1", "c1")
        graph.add_node("n2", "c2")
        assert graph.node_count == 2

    def test_add_relationship(self, graph: RelationshipGraph) -> None:
        graph.add_node("n1", "c1")
        graph.add_node("n2", "c2")
        rel = graph.add_relationship("n1", "n2", "cites", weight=0.8)
        assert isinstance(rel, Relationship)
        assert graph.edge_count == 1

    def test_node_count_empty(self, graph: RelationshipGraph) -> None:
        assert graph.node_count == 0

    def test_edge_count_empty(self, graph: RelationshipGraph) -> None:
        assert graph.edge_count == 0

    def test_add_node_with_metadata(self, graph: RelationshipGraph) -> None:
        node = graph.add_node("n1", "content", metadata={"source": "paper.pdf"})
        assert node.metadata["source"] == "paper.pdf"


# ---------------------------------------------------------------------------
# Neighbor traversal
# ---------------------------------------------------------------------------


class TestNeighborTraversal:
    def test_get_neighbors_depth_1(self, populated_graph: RelationshipGraph) -> None:
        neighbors = populated_graph.get_neighbors("A")
        ids = [n.chunk_id for n in neighbors]
        assert "B" in ids
        assert "D" in ids
        assert "C" not in ids  # C is depth 2 from A

    def test_get_neighbors_depth_2(self, populated_graph: RelationshipGraph) -> None:
        neighbors = populated_graph.get_neighbors("A", max_depth=2)
        ids = [n.chunk_id for n in neighbors]
        assert "B" in ids
        assert "C" in ids
        assert "D" in ids

    def test_get_neighbors_filtered_by_type(self, populated_graph: RelationshipGraph) -> None:
        neighbors = populated_graph.get_neighbors("A", rel_type="related")
        ids = [n.chunk_id for n in neighbors]
        assert "B" in ids
        assert "D" in ids

    def test_get_neighbors_nonexistent_node(self, populated_graph: RelationshipGraph) -> None:
        assert populated_graph.get_neighbors("MISSING") == []

    def test_get_neighbors_no_outgoing(self, populated_graph: RelationshipGraph) -> None:
        # C has no outgoing edges
        assert populated_graph.get_neighbors("C") == []

    def test_get_related_sorted_by_weight(self, populated_graph: RelationshipGraph) -> None:
        related = populated_graph.get_related("A", top_k=5)
        assert len(related) == 2
        assert related[0][1] >= related[1][1]  # Sorted descending

    def test_get_related_top_k(self, populated_graph: RelationshipGraph) -> None:
        related = populated_graph.get_related("A", top_k=1)
        assert len(related) == 1

    def test_get_related_nonexistent(self, populated_graph: RelationshipGraph) -> None:
        assert populated_graph.get_related("MISSING") == []


# ---------------------------------------------------------------------------
# Path finding
# ---------------------------------------------------------------------------


class TestPathFinding:
    def test_find_path_direct(self, populated_graph: RelationshipGraph) -> None:
        path = populated_graph.find_path("A", "B")
        assert path == ["A", "B"]

    def test_find_path_multi_hop(self, populated_graph: RelationshipGraph) -> None:
        path = populated_graph.find_path("A", "C")
        assert path == ["A", "B", "C"]

    def test_find_path_same_node(self, populated_graph: RelationshipGraph) -> None:
        path = populated_graph.find_path("A", "A")
        assert path == ["A"]

    def test_find_path_no_path(self, populated_graph: RelationshipGraph) -> None:
        # D has no outgoing edges, so no path from D to B
        path = populated_graph.find_path("D", "B")
        assert path is None

    def test_find_path_nonexistent_source(self, populated_graph: RelationshipGraph) -> None:
        assert populated_graph.find_path("MISSING", "A") is None

    def test_find_path_nonexistent_target(self, populated_graph: RelationshipGraph) -> None:
        assert populated_graph.find_path("A", "MISSING") is None


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


class TestCycleDetection:
    def test_no_cycles_in_dag(self, populated_graph: RelationshipGraph) -> None:
        cycles = populated_graph.detect_cycles()
        assert cycles == []

    def test_detect_simple_cycle(self) -> None:
        g = RelationshipGraph()
        g.add_node("X", "x")
        g.add_node("Y", "y")
        g.add_relationship("X", "Y", "related")
        g.add_relationship("Y", "X", "related")
        cycles = g.detect_cycles()
        assert len(cycles) >= 1
        # The cycle should contain both X and Y
        flat = [node for cycle in cycles for node in cycle]
        assert "X" in flat
        assert "Y" in flat

    def test_detect_three_node_cycle(self) -> None:
        g = RelationshipGraph()
        g.add_node("A", "a")
        g.add_node("B", "b")
        g.add_node("C", "c")
        g.add_relationship("A", "B", "related")
        g.add_relationship("B", "C", "related")
        g.add_relationship("C", "A", "related")
        cycles = g.detect_cycles()
        assert len(cycles) >= 1


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------


class TestGraphBuilder:
    def test_build_from_chunks(self, sample_chunks: list[dict]) -> None:
        builder = GraphBuilder(similarity_threshold=0.1)
        graph = builder.build_from_chunks(sample_chunks)
        assert graph.node_count == 4
        # Python/data science chunks should be linked
        assert graph.edge_count > 0

    def test_build_empty_chunks(self) -> None:
        builder = GraphBuilder()
        graph = builder.build_from_chunks([])
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_build_single_chunk(self) -> None:
        builder = GraphBuilder()
        graph = builder.build_from_chunks([{"chunk_id": "c1", "content": "hello"}])
        assert graph.node_count == 1
        assert graph.edge_count == 0

    def test_high_threshold_fewer_edges(self, sample_chunks: list[dict]) -> None:
        low = GraphBuilder(similarity_threshold=0.1).build_from_chunks(sample_chunks)
        high = GraphBuilder(similarity_threshold=0.8).build_from_chunks(sample_chunks)
        assert high.edge_count <= low.edge_count

    def test_build_preserves_metadata(self) -> None:
        chunks = [
            {"chunk_id": "c1", "content": "hello world", "metadata": {"page": 1}},
            {"chunk_id": "c2", "content": "hello earth", "metadata": {"page": 2}},
        ]
        builder = GraphBuilder(similarity_threshold=0.01)
        graph = builder.build_from_chunks(chunks)
        assert graph.node_count == 2


# ---------------------------------------------------------------------------
# GraphRetriever
# ---------------------------------------------------------------------------


class TestGraphRetriever:
    def test_expand_results(self, populated_graph: RelationshipGraph) -> None:
        retriever = GraphRetriever()
        expanded = retriever.expand_results(["A"], populated_graph, max_expansion=3)
        assert "A" in expanded
        assert "B" in expanded or "D" in expanded

    def test_expand_no_duplicates(self, populated_graph: RelationshipGraph) -> None:
        retriever = GraphRetriever()
        expanded = retriever.expand_results(["A", "B"], populated_graph)
        assert len(expanded) == len(set(expanded))

    def test_expand_empty_initial(self, populated_graph: RelationshipGraph) -> None:
        retriever = GraphRetriever()
        expanded = retriever.expand_results([], populated_graph)
        assert expanded == []

    def test_expand_max_expansion_limit(self, populated_graph: RelationshipGraph) -> None:
        retriever = GraphRetriever()
        expanded = retriever.expand_results(["A"], populated_graph, max_expansion=1)
        # A + at most 1 expansion
        assert len(expanded) <= 2
