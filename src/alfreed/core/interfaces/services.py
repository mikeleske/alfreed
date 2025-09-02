"""Service interfaces for business logic orchestration."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional

from ..entities.embedding import Embedding
from ..entities.search_result import SearchResultCollection
from ..entities.sequence import Sequence


class SequenceServiceInterface(ABC):
    """Interface for sequence processing business logic."""

    @abstractmethod
    def load_sequences_from_fasta(self, file_path: Path) -> List[Sequence]:
        """Load and validate sequences from FASTA file."""
        pass

    @abstractmethod
    def validate_sequences(self, sequences: List[Sequence]) -> List[Sequence]:
        """Validate sequence data and filter invalid sequences."""
        pass

    @abstractmethod
    def preprocess_sequences(
        self, sequences: List[Sequence], max_length: Optional[int] = None
    ) -> List[Sequence]:
        """Preprocess sequences (trimming, filtering, etc.)."""
        pass


class EmbeddingServiceInterface(ABC):
    """Interface for embedding generation business logic."""

    @abstractmethod
    def generate_embeddings(
        self, sequences: List[Sequence], model_name: str, batch_size: int = 32
    ) -> List[Embedding]:
        """Generate embeddings for sequences."""
        pass

    @abstractmethod
    def load_precomputed_embeddings(
        self, file_path: Path
    ) -> List[Embedding]:
        """Load pre-computed embeddings from file."""
        pass

    @abstractmethod
    def save_embeddings(self, embeddings: List[Embedding], file_path: Path) -> None:
        """Save embeddings to file."""
        pass


class SearchServiceInterface(ABC):
    """Interface for search orchestration business logic."""

    @abstractmethod
    def build_search_index(
        self, embeddings: List[Embedding], index_type: str = "flat"
    ) -> Any:
        """Build search index from embeddings."""
        pass

    @abstractmethod
    def search_similar_sequences(
        self,
        query_embeddings: List[Embedding],
        database_embeddings: List[Embedding],
        k: int = 10,
        similarity_threshold: Optional[float] = None,
    ) -> SearchResultCollection:
        """Search for similar sequences."""
        pass

    @abstractmethod
    def perform_sequence_alignment(
        self,
        results: SearchResultCollection,
        query_sequences: List[Sequence],
        database_sequences: List[Sequence],
    ) -> SearchResultCollection:
        """Add alignment information to search results."""
        pass

    @abstractmethod
    def export_results(
        self,
        results: SearchResultCollection,
        output_path: Path,
        format_type: str = "csv",
        include_alignments: bool = False,
    ) -> None:
        """Export search results to file."""
        pass
