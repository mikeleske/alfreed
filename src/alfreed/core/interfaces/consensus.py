"""Consensus calculation interfaces."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..entities.search_result import SearchResult


class TaxonomyConsensusStrategyInterface(ABC):
    """Interface for taxonomy consensus calculation strategies."""

    @abstractmethod
    def calculate_consensus(self, results: List[SearchResult]) -> Dict[str, Any]:
        """
        Calculate taxonomy consensus from search results.

        Args:
            results: List of search results for a single query

        Returns:
            Dictionary containing:
            - consensus_level: Most specific taxonomic rank where all results agree
            - proposed_taxonomy: Full taxonomy string with proper prefixes
            - confidence_score: Optional confidence metric (0.0-1.0)
            - strategy_metadata: Optional strategy-specific information
        """
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Name of the consensus strategy."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the consensus strategy."""
        pass


class ConsensusCalculatorInterface(ABC):
    """Interface for consensus calculation orchestration."""

    @abstractmethod
    def calculate_consensus_for_query(
        self, results: List[SearchResult], strategy_name: str = "basic"
    ) -> Dict[str, Any]:
        """Calculate consensus for a single query's results."""
        pass

    @abstractmethod
    def set_strategy(self, strategy: TaxonomyConsensusStrategyInterface) -> None:
        """Set the consensus calculation strategy."""
        pass

    @abstractmethod
    def list_available_strategies(self) -> Dict[str, str]:
        """List all available consensus strategies."""
        pass
