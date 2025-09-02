"""Embedding repository implementation."""

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..core.entities.embedding import Embedding
from ..core.interfaces.repositories import EmbeddingRepositoryInterface


class EmbeddingRepository(EmbeddingRepositoryInterface):
    """Implementation of embedding repository for file-based storage."""

    def __init__(self):
        self._embedding_cache: Dict[str, Embedding] = {}
        self._embeddings: np.ndarray = None

    def load_embeddings(self, file_path: Path) -> np.ndarray:
        """Load embedding matrix from NumPy file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {file_path}")

        try:
            embeddings = np.load(file_path)

            if embeddings.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")

            # Ensure float32 for FAISS compatibility
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)

            return embeddings

        except Exception as e:
            raise RuntimeError(f"Failed to load embeddings: {e}")

    def save_embeddings(self, embeddings: np.ndarray, file_path: Path) -> None:
        """Save embedding matrix to NumPy file."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Ensure float32 format
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)

            np.save(file_path, embeddings)

        except Exception as e:
            raise RuntimeError(f"Failed to save embeddings: {e}")

    def get_embeddings_for_sequences(self, sequence_ids: List[str]) -> np.ndarray:
        """Get embeddings for specific sequence IDs from cache."""
        return self._embeddings[sequence_ids]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
