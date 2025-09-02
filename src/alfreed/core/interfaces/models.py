"""Model interfaces for embedding generation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class EmbeddingModelInterface(ABC):
    """Abstract interface for embedding models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model identifier."""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        pass

    @property
    @abstractmethod
    def max_sequence_length(self) -> int:
        """Get the maximum sequence length supported."""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and prepare for inference."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        pass

    @abstractmethod
    def embed_sequence(self, sequence: str) -> np.ndarray:
        """
        Generate embedding for a single sequence.

        Args:
            sequence: DNA sequence string

        Returns:
            Embedding vector as numpy array
        """
        pass

    @abstractmethod
    def embed_sequences_batch(self, sequences: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple sequences efficiently.

        Args:
            sequences: List of DNA sequence strings

        Returns:
            Embedding matrix as numpy array
        """
        pass

    def validate_sequence(self, sequence: str) -> bool:
        """
        Validate if a sequence is suitable for this model.
        Default implementation checks length.
        """
        return len(sequence) <= self.max_sequence_length

    def preprocess_sequence(self, sequence: str) -> str:
        """
        Preprocess sequence before embedding (override if needed).
        Default implementation just converts to uppercase.
        """
        return sequence.upper()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model_name,
            "dimension": self.embedding_dimension,
            "max_length": self.max_sequence_length,
            "loaded": self.is_loaded(),
        }
