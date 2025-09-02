"""Core vector search algorithms."""

from typing import List, Optional

import numpy as np

from ..entities.embedding import Embedding
from ..entities.search_result import SearchResult, SearchResultCollection


class VectorSearchEngine:
    """Core algorithm for vector similarity search."""

    def __init__(self, metric: str = "cosine", vector_store_repository=None):
        """Initialize search engine with similarity metric."""
        self.metric = metric
        self._index = None
        self._index_metadata = None
        self._vector_store_repository = vector_store_repository

    def set_index(self, index: any, metadata: Optional[dict] = None) -> None:
        """Inject the vector index dependency."""
        self._index = index
        self._index_metadata = metadata or {}

    def search_similar_vectors(
        self,
        query_embeddings: List[Embedding],
        k: int = 10,
        similarity_threshold: Optional[float] = None,
    ) -> SearchResultCollection:
        """Search for similar vectors using the configured index."""
        if not self._index:
            raise ValueError("Vector index must be set before searching")

        # Convert embeddings to numpy array
        query_matrix = np.vstack([emb.vector for emb in query_embeddings])

        # Perform the actual search (implementation depends on injected index)
        distances, indices = self._vector_store_repository.search_similar_vectors(
            self._index, query_matrix, k
        )

        # Convert results to domain objects
        results = self._convert_to_search_results(
            query_embeddings, distances, indices, similarity_threshold
        )

        return SearchResultCollection(
            results=results,
            total_queries=len(query_embeddings),
            search_parameters={
                "k": k,
                "metric": self.metric,
                "similarity_threshold": similarity_threshold,
            },
        )

    def _convert_to_search_results(
        self,
        query_embeddings: List[Embedding],
        distances: np.ndarray,
        indices: np.ndarray,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Convert raw search results to domain objects."""
        results = []

        for i, query_embedding in enumerate(query_embeddings):
            query_distances = distances[i]
            query_indices = indices[i]

            for rank, (distance, matched_index) in enumerate(
                zip(query_distances, query_indices)
            ):
                # Convert distance to similarity score and clamp to valid range
                # Handle floating point precision issues by clamping to [-1, 1]
                raw_score = float(distance)
                similarity_score = max(-1.0, min(1.0, raw_score))

                # Apply similarity threshold if specified
                if similarity_threshold and similarity_score < similarity_threshold:
                    continue

                # Get matched sequence ID and taxonomy from metadata
                matched_sequence_id = self._get_sequence_id_from_index(matched_index)
                taxon, taxonomy = self._get_taxonomy_info_from_index(matched_index)

                result = SearchResult(
                    query_sequence_id=query_embedding.sequence_id,
                    matched_sequence_id=matched_sequence_id,
                    matched_index=int(matched_index),
                    similarity_score=similarity_score,
                    taxon=taxon,
                    taxonomy=taxonomy,
                )
                results.append(result)

        return results

    def _get_sequence_id_from_index(self, index: int) -> str:
        """Get sequence ID from index position."""
        if self._index_metadata and "sequence_ids" in self._index_metadata:
            sequence_ids = self._index_metadata["sequence_ids"]
            return sequence_ids[index]

        # Fallback to generic ID
        return f"seq_{index}"

    def _get_taxonomy_info_from_index(
        self, index: int
    ) -> tuple[Optional[str], Optional[dict]]:
        """Get taxonomy information from index position."""
        if self._index_metadata and "taxonomy_data" in self._index_metadata:
            taxonomy_data = self._index_metadata["taxonomy_data"]
            taxon_info = taxonomy_data[index]
            return taxon_info.get("taxon"), taxon_info.get("taxonomy")

        return None, None
