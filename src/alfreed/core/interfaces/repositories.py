"""Repository interfaces for data access abstraction."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..entities.embedding import Embedding
from ..entities.search_result import SearchResult
from ..entities.sequence import Sequence


class SequenceRepositoryInterface(ABC):
    """Interface for sequence data access operations."""

    @abstractmethod
    def load_from_fasta(self, file_path: Path) -> List[Sequence]:
        """Load sequences from a FASTA file."""
        pass

    @abstractmethod
    def load_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Load sequence metadata from a file."""
        pass

    @abstractmethod
    def save_sequences(self, sequences: List[Sequence], file_path: Path) -> None:
        """Save sequences to a file."""
        pass

    @abstractmethod
    def get_sequence_by_id(self, sequence_id: str) -> Optional[Sequence]:
        """Retrieve a sequence by its ID."""
        pass


class EmbeddingRepositoryInterface(ABC):
    """Interface for embedding data access operations."""

    @abstractmethod
    def load_embeddings(self, file_path: Path) -> np.ndarray:
        """Load embedding matrix from file."""
        pass

    @abstractmethod
    def save_embeddings(self, embeddings: np.ndarray, file_path: Path) -> None:
        """Save embedding matrix to file."""
        pass

    @abstractmethod
    def get_embeddings_for_sequences(self, sequence_ids: List[str]) -> List[Embedding]:
        """Get embeddings for specific sequence IDs."""
        pass


class VectorStoreRepositoryInterface(ABC):
    """Interface for vector store operations (FAISS)."""

    @abstractmethod
    def create_index(self, embeddings: np.ndarray, metric: str = "cosine") -> Any:
        """Create a vector index from embeddings."""
        pass

    @abstractmethod
    def search_similar_vectors(
        self, index: Any, query_vectors: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors in the index."""
        pass

    @abstractmethod
    def add_vectors_to_index(self, index: Any, vectors: np.ndarray) -> None:
        """Add new vectors to an existing index."""
        pass


class ResultRepositoryInterface(ABC):
    """Interface for search result storage operations."""

    @abstractmethod
    def save_results(
        self, results: List[SearchResult], output_path: Path, format_type: str = "csv"
    ) -> None:
        """Save search results to file."""
        pass

    @abstractmethod
    def save_alignments(self, results: List[SearchResult], output_path: Path) -> None:
        """Save alignment information to file."""
        pass


class ConsensusRepositoryInterface(ABC):
    """Interface for consensus data storage and retrieval operations."""

    @abstractmethod
    def store_consensus(
        self,
        query_id: str,
        consensus_data: Dict[str, Any],
        strategy_name: str = "basic",
    ) -> None:
        """Store consensus data for a query."""
        pass

    @abstractmethod
    def get_consensus(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve consensus data for a query."""
        pass

    @abstractmethod
    def get_all_consensus(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored consensus data."""
        pass

    @abstractmethod
    def has_consensus(self, query_id: str) -> bool:
        """Check if consensus data exists for a query."""
        pass

    @abstractmethod
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about stored consensus data."""
        pass
