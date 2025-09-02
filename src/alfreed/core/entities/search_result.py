"""Search result domain entities."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .sequence import Sequence


@dataclass
class SearchResult:
    """Represents a single search result match."""

    query_sequence_id: str
    matched_sequence_id: str
    matched_index: int
    similarity_score: float
    taxon: Optional[str] = None
    taxonomy: Optional[Dict[str, str]] = None
    matched_sequence: Optional[Sequence] = None
    alignment_score: Optional[float] = None
    identity_percent: Optional[float] = None

    def __post_init__(self):
        """Validate search result data."""
        if not self.query_sequence_id:
            raise ValueError("Query sequence ID cannot be empty")

        if not self.matched_sequence_id:
            raise ValueError("Matched sequence ID cannot be empty")

        # Allow small floating point precision errors (±1e-6)
        if not (-1.000001 <= self.similarity_score <= 1.000001):
            raise ValueError(
                "Similarity score must be between -1 and 1 (allowing small floating point errors)"
            )

        if self.identity_percent is not None and not 0 <= self.identity_percent <= 100:
            raise ValueError("Identity percent must be between 0 and 100")


@dataclass
class SearchResultCollection:
    """Collection of search results for multiple queries."""

    results: List[SearchResult] = field(default_factory=list)
    total_queries: int = 0
    search_parameters: Optional[Dict[str, Any]] = None
    execution_time_seconds: Optional[float] = None

    def add_result(self, result: SearchResult) -> None:
        """Add a search result to the collection."""
        self.results.append(result)

    def get_results_for_query(self, query_id: str) -> List[SearchResult]:
        """Get all results for a specific query."""
        return [r for r in self.results if r.query_sequence_id == query_id]

    def get_top_results(self, n: int = 10) -> List[SearchResult]:
        """Get top N results across all queries by similarity score."""
        return sorted(self.results, key=lambda r: r.similarity_score, reverse=True)[:n]

    def get_unique_query_ids(self) -> List[str]:
        """Get list of unique query sequence IDs."""
        return list(set(r.query_sequence_id for r in self.results))

    def filter_by_similarity_threshold(
        self, threshold: float
    ) -> "SearchResultCollection":
        """Return new collection with results above similarity threshold."""
        filtered_results = [r for r in self.results if r.similarity_score >= threshold]

        return SearchResultCollection(
            results=filtered_results,
            total_queries=self.total_queries,
            search_parameters=self.search_parameters,
            execution_time_seconds=self.execution_time_seconds,
        )

    @property
    def result_count(self) -> int:
        """Total number of results in the collection."""
        return len(self.results)

    @property
    def average_similarity_score(self) -> float:
        """Average similarity score across all results."""
        if not self.results:
            return 0.0
        return sum(r.similarity_score for r in self.results) / len(self.results)
