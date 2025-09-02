"""Example custom model implementation showing how users can add their own models."""

import logging
from typing import Callable, List, Optional

import numpy as np

from ...core.interfaces.models import EmbeddingModelInterface


class CustomEmbeddingModel(EmbeddingModelInterface):
    """
    Example custom model implementation.

    This shows how users can implement their own embedding models
    by providing a custom embedding function.
    """

    def __init__(
        self,
        model_name: str,
        embedding_function: Callable[[str], np.ndarray],
        embedding_dimension: int,
        max_sequence_length: int = 1000,
        batch_function: Optional[Callable[[List[str]], np.ndarray]] = None,
    ):
        """
        Initialize custom model.

        Args:
            model_name: Name identifier for this model
            embedding_function: Function that takes a sequence string and returns embedding
            embedding_dimension: Dimension of the embeddings
            max_sequence_length: Maximum supported sequence length
            batch_function: Optional batch processing function
        """
        self._model_name = model_name
        self._embedding_function = embedding_function
        self._embedding_dimension = embedding_dimension
        self._max_sequence_length = max_sequence_length
        self._batch_function = batch_function
        self._is_loaded = False
        self._logger = logging.getLogger(__name__)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    def load_model(self) -> None:
        """Load the custom model (if needed)."""
        self._logger.info(f"Loading custom model: {self._model_name}")
        self._is_loaded = True

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Generate embedding using the custom function."""
        if not self.is_loaded():
            self.load_model()

        try:
            embedding = self._embedding_function(sequence)

            # Ensure correct format
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)

            if embedding.dtype != np.float32:
                embedding = embedding.astype(np.float32)

            if len(embedding) != self._embedding_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self._embedding_dimension}, "
                    f"got {len(embedding)}"
                )

            return embedding

        except Exception as e:
            self._logger.error(f"Custom embedding function failed: {e}")
            # Return zero vector as fallback
            return np.zeros(self._embedding_dimension, dtype=np.float32)

    def embed_sequences_batch(self, sequences: List[str]) -> np.ndarray:
        """Generate embeddings for multiple sequences."""
        if not self.is_loaded():
            self.load_model()

        # Use batch function if provided
        if self._batch_function:
            try:
                embeddings = self._batch_function(sequences)
                if embeddings.dtype != np.float32:
                    embeddings = embeddings.astype(np.float32)
                return embeddings
            except Exception as e:
                self._logger.warning(
                    f"Batch function failed, falling back to individual: {e}"
                )

        # Fallback to individual processing
        embeddings = []
        for sequence in sequences:
            embedding = self.embed_sequence(sequence)
            embeddings.append(embedding)

        return np.stack(embeddings)

    def unload_model(self) -> None:
        """Unload the custom model."""
        self._is_loaded = False
        self._logger.info(f"Unloaded custom model: {self._model_name}")

