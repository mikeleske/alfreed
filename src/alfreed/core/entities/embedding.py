"""Embedding domain entity."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Embedding:
    """Represents a sequence embedding vector with metadata."""

    sequence_id: str
    vector: np.ndarray
    model_name: str
    embedding_dimension: int
    metadata: Optional[dict] = None
    normalized_vector: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate embedding data after initialization."""
        if not self.sequence_id:
            raise ValueError("Sequence ID cannot be empty")

        if self.vector is None or len(self.vector) == 0:
            raise ValueError("Embedding vector cannot be empty")

        if not self.model_name:
            raise ValueError("Model name cannot be empty")

        if self.embedding_dimension != len(self.vector):
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(self.vector)}"
            )

        # Ensure vector is float32 for FAISS compatibility
        if self.vector.dtype != np.float32:
            object.__setattr__(self, "vector", self.vector.astype(np.float32))

        # Calculate normalized vector if not provided
        if self.normalized_vector is None:
            object.__setattr__(self, "normalized_vector", self.l2_normalize())

    def cosine_similarity(self, other: "Embedding") -> float:
        """Calculate cosine similarity with another embedding."""
        if self.embedding_dimension != other.embedding_dimension:
            raise ValueError("Cannot compare embeddings of different dimensions")

        # Normalize vectors
        norm_a = np.linalg.norm(self.vector)
        norm_b = np.linalg.norm(other.vector)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return np.dot(self.vector, other.vector) / (norm_a * norm_b)

    def l2_normalize(self) -> np.ndarray:
        """Return L2-normalized version of this embedding."""
        norm = np.linalg.norm(self.vector)
        normalized_vector = self.vector / norm
        return normalized_vector
