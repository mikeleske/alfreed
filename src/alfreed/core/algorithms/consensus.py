"""Consensus calculation algorithms."""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from ..entities.search_result import SearchResult
from ..entities.taxonomy import parse_taxon_string
from ..interfaces.consensus import TaxonomyConsensusStrategyInterface


class BasicConsensusStrategy(TaxonomyConsensusStrategyInterface):
    """
    Basic consensus strategy that finds the deepest taxonomic level
    where all results agree (unanimous consensus).
    """

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        # Standard taxonomic levels in order from broad to specific
        self._taxonomic_levels = [
            "domain",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ]
        # Prefixes for taxonomy string formatting
        self._level_prefixes = {
            "domain": "d__",
            "phylum": "p__",
            "class": "c__",
            "order": "o__",
            "family": "f__",
            "genus": "g__",
            "species": "s__",
        }

    @property
    def strategy_name(self) -> str:
        return "basic"

    @property
    def description(self) -> str:
        return "Unanimous consensus: finds deepest level where all results agree"

    def calculate_consensus(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Calculate unanimous consensus from search results."""
        if not results:
            return {
                "consensus_level": None,
                "proposed_taxonomy": "",
                "confidence_score": 0.0,
                "strategy_metadata": {
                    "strategy": self.strategy_name,
                    "total_results": 0,
                    "results_with_taxonomy": 0,
                },
            }

        self._logger.debug(f"Calculating consensus for {len(results)} results")

        # Extract and parse taxonomies
        parsed_taxonomies = []
        results_with_taxonomy = 0

        for result in results:
            if result.taxonomy:
                parsed_taxonomies.append(result.taxonomy)
                results_with_taxonomy += 1
            elif result.taxon:
                # Parse taxon string if taxonomy dict not available
                parsed_taxonomy = parse_taxon_string(result.taxon)
                if parsed_taxonomy:
                    parsed_taxonomies.append(parsed_taxonomy)
                    results_with_taxonomy += 1

        if not parsed_taxonomies:
            return {
                "consensus_level": None,
                "proposed_taxonomy": "",
                "confidence_score": 0.0,
                "strategy_metadata": {
                    "strategy": self.strategy_name,
                    "total_results": len(results),
                    "results_with_taxonomy": 0,
                },
            }

        # Find consensus level and build proposed taxonomy
        consensus_level, proposed_taxonomy = self._find_unanimous_consensus(
            parsed_taxonomies
        )

        # Calculate confidence score based on proportion of results with taxonomy
        confidence_score = results_with_taxonomy / len(results)

        return {
            "consensus_level": consensus_level,
            "proposed_taxonomy": proposed_taxonomy,
            "confidence_score": confidence_score,
            "strategy_metadata": {
                "strategy": self.strategy_name,
                "total_results": len(results),
                "results_with_taxonomy": results_with_taxonomy,
                "agreement_levels": self._get_agreement_analysis(parsed_taxonomies),
            },
        }

    def _find_unanimous_consensus(
        self, taxonomies: List[Dict[str, str]]
    ) -> tuple[Optional[str], str]:
        """Find the deepest level where all taxonomies agree."""
        if not taxonomies:
            return None, ""

        consensus_taxonomy = {}
        consensus_level = None

        # Check each taxonomic level from domain to species
        for level in self._taxonomic_levels:
            level_values = []

            # Collect values for this level from all taxonomies
            for taxonomy in taxonomies:
                value = taxonomy.get(
                    level.title()
                )  # Level names are capitalized in taxonomy dict
                if (
                    value and value.strip() and not value.strip().startswith("__")
                ):  # Skip empty or prefix-only values
                    level_values.append(value.strip())

            if not level_values:
                # No values at this level, stop here
                break

            # Check if all values agree (unanimous)
            unique_values = set(level_values)
            if len(unique_values) == 1 and len(level_values) == len(taxonomies):
                # Unanimous agreement at this level
                consensus_taxonomy[level] = list(unique_values)[0]
                consensus_level = level
            else:
                # Disagreement at this level, stop here
                break

        proposed_taxonomy = self._build_taxonomy_string(consensus_taxonomy)
        return consensus_level, proposed_taxonomy

    def _build_taxonomy_string(self, consensus_taxonomy: Dict[str, str]) -> str:
        """Build taxonomy string with proper prefixes."""
        taxonomy_parts = []

        for level in self._taxonomic_levels:
            if level in consensus_taxonomy:
                prefix = ""  # self._level_prefixes[level]
                value = consensus_taxonomy[level]
                taxonomy_parts.append(f"{prefix}{value}")
            else:
                break  # Stop at first missing level to maintain hierarchy

        return "; ".join(taxonomy_parts)

    def _get_agreement_analysis(
        self, taxonomies: List[Dict[str, str]]
    ) -> Dict[str, Dict[str, int]]:
        """Analyze agreement levels for debugging/metadata."""
        agreement_analysis = {}

        for level in self._taxonomic_levels:
            level_values = []
            for taxonomy in taxonomies:
                value = taxonomy.get(level.title())
                if value and value.strip() and not value.strip().startswith("__"):
                    level_values.append(value.strip())

            if level_values:
                value_counts = Counter(level_values)
                agreement_analysis[level] = dict(value_counts)

        return agreement_analysis


class MajorityConsensusStrategy(TaxonomyConsensusStrategyInterface):
    """
    Majority consensus strategy that accepts the most common taxonomy
    at each level (useful when unanimous consensus is too strict).
    """

    def __init__(self, minimum_agreement_ratio: float = 0.5):
        """
        Initialize majority consensus strategy.

        Args:
            minimum_agreement_ratio: Minimum fraction of results that must agree (0.0-1.0)
        """
        self._logger = logging.getLogger(__name__)
        self._minimum_agreement_ratio = min(max(minimum_agreement_ratio, 0.0), 1.0)
        self._taxonomic_levels = [
            "domain",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ]
        self._level_prefixes = {
            "domain": "d__",
            "phylum": "p__",
            "class": "c__",
            "order": "o__",
            "family": "f__",
            "genus": "g__",
            "species": "s__",
        }

    @property
    def strategy_name(self) -> str:
        return "majority"

    @property
    def description(self) -> str:
        return f"Majority consensus: accepts most common taxonomy (>{self._minimum_agreement_ratio:.0%} agreement)"

    def calculate_consensus(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Calculate majority consensus from search results."""
        if not results:
            return {
                "consensus_level": None,
                "proposed_taxonomy": "",
                "confidence_score": 0.0,
                "strategy_metadata": {
                    "strategy": self.strategy_name,
                    "minimum_agreement_ratio": self._minimum_agreement_ratio,
                    "total_results": 0,
                    "results_with_taxonomy": 0,
                },
            }

        # Extract taxonomies (similar to basic strategy)
        parsed_taxonomies = []
        results_with_taxonomy = 0

        for result in results:
            if result.taxonomy:
                parsed_taxonomies.append(result.taxonomy)
                results_with_taxonomy += 1
            elif result.taxon:
                parsed_taxonomy = parse_taxon_string(result.taxon)
                if parsed_taxonomy:
                    parsed_taxonomies.append(parsed_taxonomy)
                    results_with_taxonomy += 1

        if not parsed_taxonomies:
            return {
                "consensus_level": None,
                "proposed_taxonomy": "",
                "confidence_score": 0.0,
                "strategy_metadata": {
                    "strategy": self.strategy_name,
                    "minimum_agreement_ratio": self._minimum_agreement_ratio,
                    "total_results": len(results),
                    "results_with_taxonomy": 0,
                },
            }

        # Find majority consensus
        consensus_level, proposed_taxonomy, agreement_scores = (
            self._find_majority_consensus(parsed_taxonomies)
        )

        # Calculate overall confidence
        confidence_score = results_with_taxonomy / len(results)
        if consensus_level and agreement_scores:
            # Weight by agreement at consensus level
            level_agreement = agreement_scores.get(consensus_level, 0.0)
            confidence_score *= level_agreement

        return {
            "consensus_level": consensus_level,
            "proposed_taxonomy": proposed_taxonomy,
            "confidence_score": confidence_score,
            "strategy_metadata": {
                "strategy": self.strategy_name,
                "minimum_agreement_ratio": self._minimum_agreement_ratio,
                "total_results": len(results),
                "results_with_taxonomy": results_with_taxonomy,
                "agreement_scores": agreement_scores,
            },
        }

    def _find_majority_consensus(
        self, taxonomies: List[Dict[str, str]]
    ) -> tuple[Optional[str], str, Dict[str, float]]:
        """Find consensus based on majority agreement."""
        consensus_taxonomy = {}
        consensus_level = None
        agreement_scores = {}

        total_taxonomies = len(taxonomies)
        minimum_votes = max(1, int(total_taxonomies * self._minimum_agreement_ratio))

        for level in self._taxonomic_levels:
            level_values = []

            # Collect non-empty values for this level
            for taxonomy in taxonomies:
                value = taxonomy.get(level.title())
                if value and value.strip() and not value.strip().startswith("__"):
                    level_values.append(value.strip())

            if not level_values:
                break

            # Find most common value
            value_counts = Counter(level_values)
            most_common_value, vote_count = value_counts.most_common(1)[0]

            # Check if it meets minimum agreement threshold
            agreement_ratio = vote_count / len(level_values)
            agreement_scores[level] = agreement_ratio

            if (
                vote_count >= minimum_votes
                and agreement_ratio >= self._minimum_agreement_ratio
            ):
                consensus_taxonomy[level] = most_common_value
                consensus_level = level
            else:
                break  # Stop at first level without sufficient agreement

        proposed_taxonomy = self._build_taxonomy_string(consensus_taxonomy)
        return consensus_level, proposed_taxonomy, agreement_scores

    def _build_taxonomy_string(self, consensus_taxonomy: Dict[str, str]) -> str:
        """Build taxonomy string with proper prefixes."""
        taxonomy_parts = []

        for level in self._taxonomic_levels:
            if level in consensus_taxonomy:
                prefix = ""  # self._level_prefixes[level]
                value = consensus_taxonomy[level]
                taxonomy_parts.append(f"{prefix}{value}")
            else:
                break

        return "; ".join(taxonomy_parts)


class MLConsensusStrategy(TaxonomyConsensusStrategyInterface):
    """
    Placeholder for future ML-based consensus strategy.
    This can be implemented later with machine learning models.
    """

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    @property
    def strategy_name(self) -> str:
        return "ml"

    @property
    def description(self) -> str:
        return "Machine Learning consensus: uses ML models to resolve taxonomy conflicts (not implemented)"

    def calculate_consensus(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Placeholder for ML-based consensus calculation."""
        raise NotImplementedError(
            "ML consensus strategy is not yet implemented. "
            "This is a placeholder for future machine learning integration."
        )
