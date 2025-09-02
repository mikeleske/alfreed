"""Result repository implementation for saving search results."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from ..core.entities.search_result import SearchResult
from ..core.interfaces.repositories import ResultRepositoryInterface


class ResultRepository(ResultRepositoryInterface):
    """Repository for saving and loading search results."""

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def save_results(
        self,
        results: List[SearchResult],
        consensus_data: Dict[str, Any],
        output_path: Path,
        format_type: str = "json",
    ) -> None:
        """Save search results to file."""
        self._logger.debug(
            f"Saving {len(results)} results to {output_path} in {format_type} format"
        )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format_type.lower() == "csv":
                self._save_to_csv(results, output_path)
            elif format_type.lower() == "json":
                self._save_to_json(results, consensus_data, output_path)
            elif format_type.lower() == "parquet":
                self._save_to_parquet(results, output_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

            self._logger.debug(f"Successfully saved results to {output_path}")

        except Exception as e:
            self._logger.error(f"Failed to save results: {e}")
            raise

    def _save_to_csv(self, results: List[SearchResult], output_path: Path) -> None:
        """Save results to CSV format."""
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "query_sequence_id",
                "matched_sequence_id",
                "matched_index",
                "similarity_score",
                "taxon",
                "domain",
                "phylum",
                "class",
                "order",
                "family",
                "genus",
                "species",
                "alignment_score",
                "identity_percent",
                "query_start",
                "query_end",
                "match_start",
                "match_end",
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    "query_sequence_id": result.query_sequence_id,
                    "matched_sequence_id": result.matched_sequence_id,
                    "matched_index": result.matched_index,
                    "similarity_score": result.similarity_score,
                    "taxon": getattr(result, "taxon", None),
                    "domain": (
                        getattr(result, "taxonomy", {}).get("Domain")
                        if getattr(result, "taxonomy", None)
                        else None
                    ),
                    "phylum": (
                        getattr(result, "taxonomy", {}).get("Phylum")
                        if getattr(result, "taxonomy", None)
                        else None
                    ),
                    "class": (
                        getattr(result, "taxonomy", {}).get("Class")
                        if getattr(result, "taxonomy", None)
                        else None
                    ),
                    "order": (
                        getattr(result, "taxonomy", {}).get("Order")
                        if getattr(result, "taxonomy", None)
                        else None
                    ),
                    "family": (
                        getattr(result, "taxonomy", {}).get("Family")
                        if getattr(result, "taxonomy", None)
                        else None
                    ),
                    "genus": (
                        getattr(result, "taxonomy", {}).get("Genus")
                        if getattr(result, "taxonomy", None)
                        else None
                    ),
                    "species": (
                        getattr(result, "taxonomy", {}).get("Species")
                        if getattr(result, "taxonomy", None)
                        else None
                    ),
                    "alignment_score": getattr(result, "alignment_score", None),
                    "identity_percent": getattr(result, "identity_percent", None),
                    "query_start": getattr(result, "query_start", None),
                    "query_end": getattr(result, "query_end", None),
                    "match_start": getattr(result, "match_start", None),
                    "match_end": getattr(result, "match_end", None),
                }
                writer.writerow(row)

    def _save_to_json(
        self,
        results: List[SearchResult],
        consensus_data: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """Save results to JSON format organized by enumerated query indices."""
        # Group results by query sequence ID
        query_results = {}
        for result in results:
            query_id = result.query_sequence_id
            if query_id not in query_results:
                query_results[query_id] = []

            result_dict = {
                "matched_sequence_id": result.matched_sequence_id,
                "matched_index": result.matched_index,
                "similarity_score": result.similarity_score,
                "taxon": getattr(result, "taxon", None),
                "taxonomy": getattr(result, "taxonomy", None),
                "alignment_score": getattr(result, "alignment_score", None),
                "identity_percent": getattr(result, "identity_percent", None),
                "query_start": getattr(result, "query_start", None),
                "query_end": getattr(result, "query_end", None),
                "match_start": getattr(result, "match_start", None),
                "match_end": getattr(result, "match_end", None),
            }
            query_results[query_id].append(result_dict)

        # Get unique query IDs in the order they appear
        unique_queries = []
        seen = set()
        for result in results:
            if result.query_sequence_id not in seen:
                unique_queries.append(result.query_sequence_id)
                seen.add(result.query_sequence_id)

        # Create enumerated structure with consensus
        output_data = {}
        for i, query_id in enumerate(unique_queries):

            # Build consensus element (only include level and taxonomy for cleaner output)
            consensus_element = {
                "consensus_level": consensus_data[query_id].get("consensus_level"),
                "proposed_taxonomy": consensus_data[query_id].get(
                    "proposed_taxonomy", ""
                ),
                "next_level": consensus_data[query_id].get("next_level"),
                "next_level_options": consensus_data[query_id].get(
                    "next_level_options", []
                ),
            }

            output_data[str(i)] = {
                "query_sequence_id": query_id,
                "consensus": consensus_element,
                "results": query_results[query_id],
            }

        with open(output_path, "w", encoding="utf-8") as jsonfile:
            json.dump(output_data, jsonfile, indent=2, ensure_ascii=False)

    def save_alignments(self, results: List[SearchResult], output_path: Path) -> None:
        """Save alignment information to file."""
        self._logger.info(
            f"Saving alignment data for {len(results)} results to {output_path}"
        )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Filter results that have alignment information
            aligned_results = [r for r in results if r.alignment_score is not None]

            if not aligned_results:
                self._logger.warning("No alignment data found in results")
                return

            # Save as CSV with alignment-specific fields
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "query_sequence_id",
                    "matched_sequence_id",
                    "alignment_score",
                    "identity_percent",
                    "query_start",
                    "query_end",
                    "match_start",
                    "match_end",
                    "similarity_score",
                    "taxon",
                    "taxonomy",
                    "metadata",
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in aligned_results:
                    row = {
                        "query_sequence_id": result.query_sequence_id,
                        "matched_sequence_id": result.matched_sequence_id,
                        "alignment_score": getattr(result, "alignment_score", None),
                        "identity_percent": getattr(result, "identity_percent", None),
                        "query_start": getattr(result, "query_start", None),
                        "query_end": getattr(result, "query_end", None),
                        "match_start": getattr(result, "match_start", None),
                        "match_end": getattr(result, "match_end", None),
                        "similarity_score": result.similarity_score,
                        "taxon": getattr(result, "taxon", None),
                        "taxonomy": (
                            json.dumps(getattr(result, "taxonomy", None))
                            if getattr(result, "taxonomy", None)
                            else None
                        ),
                        "metadata": (
                            json.dumps(result.metadata) if result.metadata else None
                        ),
                    }
                    writer.writerow(row)

            self._logger.info(f"Successfully saved alignment data to {output_path}")

        except Exception as e:
            self._logger.error(f"Failed to save alignment data: {e}")
            raise

    def get_statistics(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Calculate statistics for search results."""
        if not results:
            return {}

        similarity_scores = [r.similarity_score for r in results]

        stats = {
            "total_results": len(results),
            "unique_queries": len(set(r.query_sequence_id for r in results)),
            "unique_matches": len(set(r.matched_sequence_id for r in results)),
            "mean_similarity": sum(similarity_scores) / len(similarity_scores),
            "min_similarity": min(similarity_scores),
            "max_similarity": max(similarity_scores),
        }

        # Add alignment statistics if available
        aligned_results = [r for r in results if r.alignment_score is not None]
        if aligned_results:
            identity_scores = [
                r.identity_percent
                for r in aligned_results
                if r.identity_percent is not None
            ]
            alignment_scores = [r.alignment_score for r in aligned_results]

            stats.update(
                {
                    "results_with_alignment": len(aligned_results),
                    "mean_alignment_score": (
                        sum(alignment_scores) / len(alignment_scores)
                        if alignment_scores
                        else None
                    ),
                    "mean_identity_percent": (
                        sum(identity_scores) / len(identity_scores)
                        if identity_scores
                        else None
                    ),
                    "min_identity_percent": (
                        min(identity_scores) if identity_scores else None
                    ),
                    "max_identity_percent": (
                        max(identity_scores) if identity_scores else None
                    ),
                }
            )

        return stats
