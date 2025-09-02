"""Consensus calculation service."""

import logging
from typing import Any, Dict, List, Optional

from ..core.algorithms.consensus import (
    BasicConsensusStrategy,
    MajorityConsensusStrategy,
)
from ..core.entities.search_result import SearchResult
from ..core.interfaces.consensus import (
    ConsensusCalculatorInterface,
    TaxonomyConsensusStrategyInterface,
)
from ..core.interfaces.repositories import ConsensusRepositoryInterface


class ConsensusService(ConsensusCalculatorInterface):
    """Service for calculating taxonomy consensus from search results."""

    # Class constant for taxonomic levels
    TAXONOMIC_LEVELS = [
        "domain",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]

    def __init__(
        self,
        consensus_repository: ConsensusRepositoryInterface,
        default_strategy: str = "basic",
    ):

        self._logger = logging.getLogger(__name__)
        self._strategies: Dict[str, TaxonomyConsensusStrategyInterface] = {}
        self._current_strategy: Optional[TaxonomyConsensusStrategyInterface] = None
        self._consensus_repository = consensus_repository
        self._default_strategy = default_strategy

        # Register built-in strategies
        self._register_builtin_strategies()

        # Set default strategy
        if default_strategy in self._strategies:
            self.set_strategy(self._strategies[default_strategy])
        else:
            self._logger.warning(
                f"Unknown default strategy '{default_strategy}', using 'basic'"
            )
            self.set_strategy(self._strategies["basic"])

    def _register_builtin_strategies(self) -> None:
        """Register all built-in consensus strategies."""
        strategies = [
            BasicConsensusStrategy(),
            MajorityConsensusStrategy(),
            MajorityConsensusStrategy(minimum_agreement_ratio=0.7),  # Stricter majority
            # MLConsensusStrategy(),  # Don't register until implemented
        ]

        for strategy in strategies:
            strategy_name = strategy.strategy_name
            if strategy_name == "majority" and hasattr(
                strategy, "_minimum_agreement_ratio"
            ):
                # Handle multiple majority strategies with different thresholds
                if strategy._minimum_agreement_ratio != 0.5:
                    strategy_name = (
                        f"majority_{int(strategy._minimum_agreement_ratio * 100)}"
                    )

            self._strategies[strategy_name] = strategy
            self._logger.debug(f"Registered consensus strategy: {strategy_name}")

    def set_strategy(self, strategy: TaxonomyConsensusStrategyInterface) -> None:
        """Set the current consensus calculation strategy."""
        self._current_strategy = strategy
        self._logger.debug(f"Set consensus strategy to: {strategy.strategy_name}")

    def set_strategy_by_name(self, strategy_name: str) -> None:
        """Set strategy by name."""
        if strategy_name not in self._strategies:
            available = list(self._strategies.keys())
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. Available: {available}"
            )

        self.set_strategy(self._strategies[strategy_name])

    def calculate_consensus_for_query(
        self,
        query_id: str,
        results: List[SearchResult],
        strategy_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate consensus for a single query's results.

        Args:
            results: List of search results for one query
            strategy_name: Optional strategy to use (overrides current strategy)

        Returns:
            Consensus information dictionary
        """
        if not results:
            return {
                "consensus_level": None,
                "proposed_taxonomy": "",
                "confidence_score": 0.0,
                "strategy_metadata": {
                    "strategy": strategy_name
                    or (
                        self._current_strategy.strategy_name
                        if self._current_strategy
                        else "none"
                    ),
                    "total_results": 0,
                },
            }

        # Use specified strategy or current strategy
        strategy = self._current_strategy
        if strategy_name:
            if strategy_name not in self._strategies:
                raise ValueError(
                    f"Unknown strategy '{strategy_name}'. Available: {list(self._strategies.keys())}"
                )
            strategy = self._strategies[strategy_name]

        if not strategy:
            raise ValueError("No consensus strategy set")

        self._logger.debug(
            f"Calculating consensus for {len(results)} results using {strategy.strategy_name}"
        )

        try:
            consensus_info = strategy.calculate_consensus(results)

            # Add next level information
            self._add_next_level_info(consensus_info)

            # Store consensus and return the result
            self._consensus_repository.store_consensus(query_id, consensus_info)
            return consensus_info

        except Exception as e:
            self._logger.error(f"Failed to calculate consensus: {e}")
            # Return empty consensus on error
            return {
                "consensus_level": None,
                "proposed_taxonomy": "",
                "confidence_score": 0.0,
                "strategy_metadata": {
                    "strategy": strategy.strategy_name,
                    "error": str(e),
                    "total_results": len(results),
                },
            }

    def _add_next_level_info(self, consensus_info: Dict[str, Any]) -> None:
        """Add next level information to consensus result."""
        consensus_level = consensus_info.get("consensus_level")

        # Find next level if current level is valid and not the last level
        next_level = "domain"
        if consensus_level is not None and consensus_level != "species":
            try:
                current_index = self.TAXONOMIC_LEVELS.index(consensus_level)
                next_level = self.TAXONOMIC_LEVELS[current_index + 1]
            except (ValueError, IndexError):
                # Level not found or is the last level
                pass

        # Set next level information
        consensus_info["next_level"] = next_level
        if next_level:
            # Safely get options with fallback to empty list
            strategy_metadata = consensus_info.get("strategy_metadata", {})
            agreement_levels = strategy_metadata.get("agreement_levels", {})
            consensus_info["next_level_options"] = agreement_levels.get(next_level, [])
        else:
            consensus_info["next_level_options"] = []

    def list_available_strategies(self) -> Dict[str, str]:
        """List all available consensus strategies."""
        return {
            name: strategy.description for name, strategy in self._strategies.items()
        }

    def register_custom_strategy(
        self, strategy: TaxonomyConsensusStrategyInterface
    ) -> None:
        """Register a custom consensus strategy."""
        strategy_name = strategy.strategy_name

        if strategy_name in self._strategies:
            self._logger.warning(f"Overriding existing strategy: {strategy_name}")

        self._strategies[strategy_name] = strategy
        self._logger.info(f"Registered custom consensus strategy: {strategy_name}")

    def get_current_strategy_info(self) -> Dict[str, str]:
        """Get information about the current strategy."""
        if not self._current_strategy:
            return {"name": "none", "description": "No strategy set"}

        return {
            "name": self._current_strategy.strategy_name,
            "description": self._current_strategy.description,
        }
