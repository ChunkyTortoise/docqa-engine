"""Document Relationship Graph: build and query document relationship DAGs."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DocumentNode:
    chunk_id: str
    content: str
    summary: str
    metadata: dict = field(default_factory=dict)
    embedding: np.ndarray | None = None


@dataclass
class Relationship:
    source_id: str
    target_id: str
    rel_type: str  # "cites" | "expands" | "related" | "contradicts"
    weight: float = 1.0


class RelationshipGraph:
    """Build and query a document relationship graph."""

    def __init__(self) -> None:
        self._nodes: dict[str, DocumentNode] = {}
        self._edges: list[Relationship] = []
        self._adj: dict[str, list[tuple[str, Relationship]]] = defaultdict(list)

    def add_node(
        self,
        chunk_id: str,
        content: str,
        summary: str = "",
        metadata: dict | None = None,
    ) -> DocumentNode:
        """Add a document node to the graph."""
        node = DocumentNode(
            chunk_id=chunk_id,
            content=content,
            summary=summary,
            metadata=metadata if metadata is not None else {},
        )
        self._nodes[chunk_id] = node
        return node

    def add_relationship(
        self,
        source: str,
        target: str,
        rel_type: str = "related",
        weight: float = 1.0,
    ) -> Relationship:
        """Add a directed relationship between two nodes."""
        rel = Relationship(
            source_id=source,
            target_id=target,
            rel_type=rel_type,
            weight=weight,
        )
        self._edges.append(rel)
        self._adj[source].append((target, rel))
        return rel

    def get_neighbors(
        self,
        chunk_id: str,
        rel_type: str | None = None,
        max_depth: int = 1,
    ) -> list[DocumentNode]:
        """Get neighboring nodes up to max_depth hops, optionally filtered by rel_type."""
        if chunk_id not in self._nodes:
            return []

        visited: set[str] = {chunk_id}
        queue: deque[tuple[str, int]] = deque([(chunk_id, 0)])
        result: list[DocumentNode] = []

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor_id, rel in self._adj.get(current, []):
                if neighbor_id in visited:
                    continue
                if rel_type is not None and rel.rel_type != rel_type:
                    continue
                if neighbor_id in self._nodes:
                    visited.add(neighbor_id)
                    result.append(self._nodes[neighbor_id])
                    queue.append((neighbor_id, depth + 1))

        return result

    def get_related(self, chunk_id: str, top_k: int = 5) -> list[tuple[DocumentNode, float]]:
        """Get related nodes sorted by relationship weight (descending)."""
        if chunk_id not in self._nodes:
            return []

        neighbors_with_weight: list[tuple[DocumentNode, float]] = []
        for neighbor_id, rel in self._adj.get(chunk_id, []):
            if neighbor_id in self._nodes:
                neighbors_with_weight.append((self._nodes[neighbor_id], rel.weight))

        neighbors_with_weight.sort(key=lambda x: x[1], reverse=True)
        return neighbors_with_weight[:top_k]

    def find_path(self, source: str, target: str) -> list[str] | None:
        """BFS shortest path between two nodes. Returns node IDs or None."""
        if source not in self._nodes or target not in self._nodes:
            return None
        if source == target:
            return [source]

        visited: set[str] = {source}
        queue: deque[list[str]] = deque([[source]])

        while queue:
            path = queue.popleft()
            current = path[-1]
            for neighbor_id, _rel in self._adj.get(current, []):
                if neighbor_id in visited:
                    continue
                new_path = path + [neighbor_id]
                if neighbor_id == target:
                    return new_path
                visited.add(neighbor_id)
                queue.append(new_path)

        return None

    def detect_cycles(self) -> list[list[str]]:
        """Detect all cycles in the graph using DFS."""
        cycles: list[list[str]] = []
        visited: set[str] = set()

        for start_id in self._nodes:
            if start_id in visited:
                continue
            self._dfs_cycles(start_id, [], set(), visited, cycles)

        return cycles

    def _dfs_cycles(
        self,
        node: str,
        path: list[str],
        path_set: set[str],
        global_visited: set[str],
        cycles: list[list[str]],
    ) -> None:
        """DFS helper for cycle detection."""
        if node in path_set:
            # Found a cycle — extract it
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            cycles.append(cycle)
            return

        if node in global_visited:
            return

        path.append(node)
        path_set.add(node)

        for neighbor_id, _rel in self._adj.get(node, []):
            self._dfs_cycles(neighbor_id, path, path_set, global_visited, cycles)

        path.pop()
        path_set.remove(node)
        global_visited.add(node)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)


class GraphBuilder:
    """Auto-detect relationships from chunk content using TF-IDF similarity."""

    def __init__(self, similarity_threshold: float = 0.3) -> None:
        self.similarity_threshold = similarity_threshold

    def build_from_chunks(self, chunks: list[dict]) -> RelationshipGraph:
        """Build a RelationshipGraph from a list of chunk dicts.

        Each dict should have: chunk_id, content, and optionally summary, metadata.
        """
        graph = RelationshipGraph()

        if not chunks:
            return graph

        for chunk in chunks:
            graph.add_node(
                chunk_id=chunk["chunk_id"],
                content=chunk["content"],
                summary=chunk.get("summary", ""),
                metadata=chunk.get("metadata", {}),
            )

        contents = [c["content"] for c in chunks]
        if len(contents) < 2:
            return graph

        # Filter out empty content
        non_empty_indices = [i for i, c in enumerate(contents) if c and c.strip()]
        if len(non_empty_indices) < 2:
            return graph

        non_empty_contents = [contents[i] for i in non_empty_indices]

        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(non_empty_contents)
            sim_matrix = cosine_similarity(tfidf_matrix)
        except ValueError:
            return graph

        for i_idx, i in enumerate(non_empty_indices):
            for j_idx, j in enumerate(non_empty_indices):
                if i >= j:
                    continue
                sim = float(sim_matrix[i_idx][j_idx])
                if sim >= self.similarity_threshold:
                    graph.add_relationship(
                        source=chunks[i]["chunk_id"],
                        target=chunks[j]["chunk_id"],
                        rel_type="related",
                        weight=round(sim, 4),
                    )

        return graph


class GraphRetriever:
    """Expand retrieval results using graph neighbors."""

    def expand_results(
        self,
        initial_results: list[str],
        graph: RelationshipGraph,
        max_expansion: int = 3,
    ) -> list[str]:
        """Expand initial chunk IDs by adding graph neighbors.

        Returns a deduplicated list of chunk IDs including originals and expansions.
        """
        seen: set[str] = set(initial_results)
        expanded: list[str] = list(initial_results)

        for chunk_id in initial_results:
            neighbors = graph.get_neighbors(chunk_id)
            added = 0
            for neighbor in neighbors:
                if neighbor.chunk_id not in seen and added < max_expansion:
                    seen.add(neighbor.chunk_id)
                    expanded.append(neighbor.chunk_id)
                    added += 1

        return expanded
