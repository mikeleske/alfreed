"""Consensus repository implementation for storing and querying consensus data."""

import logging
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np

from ..core.interfaces.repositories import ConsensusRepositoryInterface


class ConsensusRepository(ConsensusRepositoryInterface):
    """Simple implementation of consensus repository with in-memory storage."""

    def __init__(self):
        """Initialize the consensus repository."""
        self._logger = logging.getLogger(__name__)
        self._consensus_store: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    def store_consensus(
        self,
        query_id: str,
        consensus_data: Dict[str, Any],
        strategy_name: str = "basic",
    ) -> None:
        """Store consensus data for a query."""
        if not query_id:
            raise ValueError("Query ID cannot be empty")

        if not consensus_data:
            raise ValueError("Consensus data cannot be empty")

        self._consensus_store[query_id][strategy_name] = consensus_data

        self._logger.debug(
            f"Stored consensus for query '{query_id}' using strategy '{strategy_name}'"
        )

    def get_consensus(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve consensus data for a query (returns most recent strategy result)."""
        if not query_id or query_id not in self._consensus_store:
            return None

        strategies_data = self._consensus_store[query_id]

        # Return the first available strategy result (could be enhanced to prefer specific strategies)
        if strategies_data:
            return next(iter(strategies_data.values()))

        return None

    def get_all_consensus(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored consensus data."""
        # Flatten the nested structure for simpler access
        flattened = {}

        for query_id, strategies_data in self._consensus_store.items():
            # For simplicity, return the first strategy result per query
            if strategies_data:
                flattened[query_id] = next(iter(strategies_data.values()))

        return flattened

    def has_consensus(self, query_id: str) -> bool:
        """Check if consensus data exists for a query."""
        return query_id in self._consensus_store and bool(
            self._consensus_store[query_id]
        )

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about stored consensus data."""
        total_queries = len(self._consensus_store)

        if not total_queries:
            return {
                "total_queries": 0,
                "consensus_level_distribution": {},
                "queries_with_consensus": 0,
                "queries_without_consensus": 0,
                "average_consensus_depth": 0.0,
            }

        # Calculate consensus for each query
        consensus_levels = []
        consensus_level_counts = {}
        taxonomic_levels = [
            "domain",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ]

        # Initialize counts
        for level in taxonomic_levels:
            consensus_level_counts[level] = 0
        consensus_level_counts["no_consensus"] = 0

        for query_id, strategies_data in self._consensus_store.items():
            for strategy_name, consensus_data in strategies_data.items():
                level = consensus_data.get("consensus_level")
                if level:
                    consensus_levels.append(level)
                    consensus_level_counts[level] += 1

        queries_with_consensus = total_queries - consensus_level_counts["no_consensus"]

        # Calculate average consensus depth (numerical representation)
        level_depth_map = {
            "domain": 1,
            "phylum": 2,
            "class": 3,
            "order": 4,
            "family": 5,
            "genus": 6,
            "species": 7,
        }

        if consensus_levels:
            consensus_depths = [
                level_depth_map.get(level, 0) for level in consensus_levels
            ]
            average_consensus_depth = np.mean(consensus_depths)
        else:
            average_consensus_depth = 0.0

        # Calculate percentage distribution
        consensus_level_percentages = {}
        for level, count in consensus_level_counts.items():
            consensus_level_percentages[level] = (
                (count / total_queries) * 100 if total_queries > 0 else 0.0
            )

        stats = {
            "total_queries": total_queries,
            "consensus_level_distribution": consensus_level_counts,
            "consensus_level_percentages": consensus_level_percentages,
            "queries_with_consensus": queries_with_consensus,
            "queries_without_consensus": consensus_level_counts["no_consensus"],
            "average_consensus_depth": average_consensus_depth,
            "consensus_coverage": (
                (queries_with_consensus / total_queries) * 100
                if total_queries > 0
                else 0.0
            ),
        }

        return stats
