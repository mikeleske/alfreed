"""Search service implementation."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.algorithms.alignment import SequenceAligner
from ..core.algorithms.vector_search import VectorSearchEngine
from ..core.entities.embedding import Embedding
from ..core.entities.search_result import SearchResultCollection
from ..core.entities.sequence import Sequence
from ..core.interfaces.repositories import (
    ResultRepositoryInterface,
    VectorStoreRepositoryInterface,
)
from ..core.interfaces.services import SearchServiceInterface
from .consensus_service import ConsensusService
from .metadata_service import MetadataService


class SearchService(SearchServiceInterface):
    """Service for search orchestration business logic."""

    def __init__(
        self,
        vector_store_repository: VectorStoreRepositoryInterface,
        result_repository: Optional[ResultRepositoryInterface] = None,
        vector_search_engine: Optional[VectorSearchEngine] = None,
        consensus_service: Optional[ConsensusService] = None,
        sequence_aligner: Optional[SequenceAligner] = None,
        metadata_service: Optional[MetadataService] = None,
    ):
        self._vector_store_repository = vector_store_repository
        self._result_repository = result_repository
        self._consensus_service = consensus_service
        # Inject the vector store repository into the search engine for dependency injection
        self._vector_search_engine = vector_search_engine or VectorSearchEngine(
            vector_store_repository=vector_store_repository
        )
        self._sequence_aligner = sequence_aligner or SequenceAligner()
        self._metadata_service = metadata_service or MetadataService()
        self._logger = logging.getLogger(__name__)

    def build_search_index(
        self,
        embeddings: np.ndarray,
        index_type: str = "flat",
        metric: str = "cosine",
        **kwargs,
    ) -> Any:
        """Build search index from embeddings."""
        self._logger.debug(
            f"Building {index_type} index for {len(embeddings)} embeddings"
        )

        start_time = time.time()

        try:
            # Build index based on type
            if index_type == "flat":
                index = self._vector_store_repository.create_index(
                    embeddings=embeddings, metric=metric
                )
            elif index_type == "ivf":
                n_clusters = kwargs.get("n_clusters", min(100, len(embeddings) // 10))
                index = self._vector_store_repository.create_ivf_index(
                    embeddings=embeddings, n_clusters=n_clusters, metric=metric
                )
            else:
                raise ValueError(f"Unsupported index type: {index_type}")

            sequence_ids = self._metadata_service.get_sequence_ids()
            taxons = self._metadata_service.get_taxons()
            parsed_taxons = self._metadata_service.get_parsed_taxons()

            taxonomy_data = []
            for seq_id, taxon, parsed_taxon in zip(sequence_ids, taxons, parsed_taxons):
                taxonomy_data.append({"taxon": taxon, "taxonomy": parsed_taxon})

            # Set up search engine with index and metadata including taxonomy
            self._vector_search_engine.set_index(
                index=index,
                metadata={"sequence_ids": sequence_ids, "taxonomy_data": taxonomy_data},
            )

            elapsed_time = time.time() - start_time
            self._logger.debug(f"Built search index in {elapsed_time:.2f} seconds")

            return index

        except Exception as e:
            self._logger.error(f"Failed to build search index: {e}")
            raise

    def search_similar_sequences(
        self,
        query_embeddings: List[Embedding],
        database_embeddings: List[Embedding],
        k: int = 10,
        similarity_threshold: Optional[float] = None,
        exclude_self_matches: bool = True,
    ) -> SearchResultCollection:
        """Search for similar sequences."""
        self._logger.debug(
            f"Searching for similar sequences: {len(query_embeddings)} queries "
            f"against {len(database_embeddings)} database sequences"
        )

        self._database_embeddings = database_embeddings

        start_time = time.time()

        try:
            # Build index if not already built
            if not self._vector_search_engine._index:
                self.build_search_index(database_embeddings)

            # Perform search
            search_results = self._vector_search_engine.search_similar_vectors(
                query_embeddings=query_embeddings,
                k=k,
                similarity_threshold=similarity_threshold,
            )

            # Filter self-matches if requested
            if exclude_self_matches:
                search_results = self._filter_self_matches(search_results)

            elapsed_time = time.time() - start_time
            search_results.execution_time_seconds = elapsed_time

            self._logger.info(
                f"Vector Search found {search_results.result_count} matches in {elapsed_time:.2f}s"
            )

            return search_results

        except Exception as e:
            self._logger.error(f"Failed to perform search: {e}")
            raise

    def _filter_self_matches(
        self, search_results: SearchResultCollection
    ) -> SearchResultCollection:
        """Filter out self-matches from search results."""
        filtered_results = []
        for result in search_results.results:
            if result.matched_sequence_id != result.query_sequence_id:
                filtered_results.append(result)

        search_results.results = filtered_results
        return search_results

    def perform_sequence_alignment(
        self,
        results: SearchResultCollection,
        query_sequences: List[Sequence],
        database_sequences: List[Sequence],
    ) -> SearchResultCollection:
        """Add alignment information to search results."""
        self._logger.info(
            f"Performing sequence alignment for {results.result_count} results"
        )

        start_time = time.time()

        try:
            # Create sequence dictionaries for quick lookup
            query_seq_dict = {seq.id: seq for seq in query_sequences}
            db_seq_dict = {seq.id: seq for seq in database_sequences}

            # Add alignment information to results
            updated_results = self._sequence_aligner.add_alignment_to_search_results(
                results=results.results,
                query_sequences=query_seq_dict,
                target_sequences=db_seq_dict,
            )

            # Update the collection
            results.results = updated_results

            elapsed_time = time.time() - start_time
            self._logger.info(f"Alignment completed in {elapsed_time:.2f} seconds")

            return results

        except Exception as e:
            self._logger.error(f"Failed to perform alignment: {e}")
            raise

    def export_results(
        self,
        results: SearchResultCollection,
        output_path: Path,
        format_type: str = "json",
        include_alignments: bool = False,
        alignment_output_path: Optional[Path] = None,
    ) -> None:
        """Export search results to file."""
        self._logger.debug(f"Exporting {results.result_count} results to {output_path}")

        try:
            if not self._result_repository:
                raise ValueError("Result repository not configured")

            # Export main results
            self._result_repository.save_results(
                results=results.results,
                consensus_data=self.get_consensus_data(),
                output_path=output_path,
                format_type=format_type,
            )

            # Export alignments if requested
            if include_alignments and alignment_output_path:
                alignment_results = [
                    r for r in results.results if r.alignment_score is not None
                ]

                if alignment_results:
                    self._result_repository.save_alignments(
                        results=alignment_results, output_path=alignment_output_path
                    )
                    self._logger.info(f"Exported alignments to {alignment_output_path}")

            self._logger.debug(f"Successfully exported results to {output_path}")

        except Exception as e:
            self._logger.error(f"Failed to export results: {e}")
            raise

    def get_top_hits_per_query(
        self, results: SearchResultCollection, n_hits: int = 5
    ) -> SearchResultCollection:
        """
        Get top N unique similarity scores for each query.

        Returns all results with similarity scores >= the nth unique score.
        For example, if n_hits=5 and unique scores are [0.95, 0.90, 0.85, 0.80, 0.75],
        all results with scores >= 0.75 will be returned.
        """
        top_results = []

        for query_id in results.get_unique_query_ids():
            query_results = results.get_results_for_query(query_id)

            if not query_results:
                continue

            unique_scores = sorted(
                list(set([result.similarity_score for result in query_results])),
                reverse=True,
            )

            if len(unique_scores) <= n_hits:
                threshold_score = min(unique_scores)
            else:
                threshold_score = unique_scores[n_hits - 1]

            # Keep all results with scores >= threshold
            filtered_results = [
                result
                for result in query_results
                if result.similarity_score >= threshold_score
            ]

            top_results.extend(filtered_results)

            self._logger.debug(
                f"Query {query_id}: {len(unique_scores)} unique scores, "
                f"threshold={threshold_score:.3f}, kept {len(filtered_results)}/{len(query_results)} results"
            )

        return SearchResultCollection(
            results=top_results,
            total_queries=results.total_queries,
            search_parameters=results.search_parameters,
            execution_time_seconds=results.execution_time_seconds,
        )

    def calculate_consensus(self, results: SearchResultCollection) -> None:
        """Calculate consensus for search results."""

        self._logger.info(f"Calculating consensus for {results.result_count} results")
        for query_id in results.get_unique_query_ids():
            query_results = results.get_results_for_query(query_id)

            self._consensus_service.calculate_consensus_for_query(
                query_id,
                query_results,
                strategy_name=self._consensus_service._default_strategy,
            )

    def get_candidate_embeddings(self) -> Dict[str, Any]:
        """Get the indices of sequences that match the given taxonomy level and value."""

        self._logger.info("Getting candidate embeddings to improve consensus")
        consensus_results = (
            self._consensus_service._consensus_repository.get_all_consensus()
        )

        candidate_data = {}

        for query_id, consensus_info in consensus_results.items():
            candidate_data[query_id] = {}
            if consensus_info.get("consensus_level") != "species":
                options = consensus_info.get("next_level_options").items()
                for option, count in options:
                    if len(option) > 3:
                        indices = self._metadata_service.get_taxonomy_indices(option)
                        # candidate_data[query_id][option] = {
                        #    'indices': indices,
                        #    'embeddings': self._database_embeddings[indices]
                        # }
        self._logger.info("Getting candidate embeddings to improve consensus - Done")
        return candidate_data

    def filter_results_by_identity_threshold(
        self, results: SearchResultCollection, identity_threshold: float
    ) -> SearchResultCollection:
        """Filter results by sequence identity threshold."""
        filtered_results = [
            r
            for r in results.results
            if r.identity_percent is not None
            and r.identity_percent >= identity_threshold
        ]

        self._logger.info(
            f"Filtered to {len(filtered_results)} results above "
            f"identity threshold {identity_threshold}%"
        )

        return SearchResultCollection(
            results=filtered_results,
            total_queries=results.total_queries,
            search_parameters=results.search_parameters,
            execution_time_seconds=results.execution_time_seconds,
        )

    def calculate_search_statistics(
        self, results: SearchResultCollection
    ) -> Dict[str, Any]:
        """Calculate statistics for search results."""
        if not results.results:
            return {}

        similarity_scores = [r.similarity_score for r in results.results]

        stats = {
            "total_results": results.result_count,
            "unique_queries": len(results.get_unique_query_ids()),
            "mean_similarity_score": np.mean(similarity_scores),
            "std_similarity_score": np.std(similarity_scores),
            "min_similarity_score": min(similarity_scores),
            "max_similarity_score": max(similarity_scores),
            "execution_time_seconds": results.execution_time_seconds,
        }

        # Add alignment statistics if available
        identity_scores = [
            r.identity_percent
            for r in results.results
            if r.identity_percent is not None
        ]

        if identity_scores:
            stats.update(
                {
                    "results_with_alignment": len(identity_scores),
                    "mean_identity_percent": np.mean(identity_scores),
                    "std_identity_percent": np.std(identity_scores),
                    "min_identity_percent": min(identity_scores),
                    "max_identity_percent": max(identity_scores),
                }
            )

        return stats

    def get_consensus_data(
        self,
    ) -> Dict[str, Any]:
        """Get consensus data."""
        return self._consensus_service._consensus_repository.get_all_consensus()

    def get_concensus_stats(
        self,
    ) -> Dict[str, Any]:
        """Get consensus statistics."""
        return self._consensus_service._consensus_repository.get_summary_stats()

    def save_search_index(self, index: Any, file_path: Path) -> None:
        """Save search index to file for later use."""
        self._logger.info(f"Saving search index to {file_path}")

        try:
            self._vector_store_repository.save_index(index, file_path)
            self._logger.info(f"Successfully saved index to {file_path}")

        except Exception as e:
            self._logger.error(f"Failed to save index: {e}")
            raise

    def load_search_index(self, file_path: Path) -> Any:
        """Load a pre-built search index from file."""
        self._logger.info(f"Loading search index from {file_path}")

        try:
            index = self._vector_store_repository.load_index(file_path)
            self._vector_search_engine.set_index(index)

            self._logger.info(f"Successfully loaded index from {file_path}")
            return index

        except Exception as e:
            self._logger.error(f"Failed to load index: {e}")
            raise

    def load_metadata(self, metadata_file: Path) -> None:
        """Load sequence metadata including taxonomy information."""
        self._logger.info(f"Loading sequence metadata from {metadata_file}")
        try:
            self._metadata_service.load_metadata_file(metadata_file)
            self._logger.info("Successfully loaded sequence metadata")
        except Exception as e:
            self._logger.error(f"Failed to load metadata: {e}")
            raise

    def get_taxonomy_summary(self) -> Dict[str, Any]:
        """Get a summary of available taxonomy information."""
        return self._metadata_service.get_taxonomy_summary()
