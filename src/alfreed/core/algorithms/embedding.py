"""Core embedding generation algorithms."""

from typing import List, Optional

import numpy as np

from ..entities.embedding import Embedding
from ..entities.sequence import Sequence
from ..interfaces.models import EmbeddingModelInterface


class EmbeddingGenerator:
    """Core algorithm for generating sequence embeddings."""

    def __init__(self, model: Optional[EmbeddingModelInterface] = None):
        self._model = model
        self._fallback_dimension = 768  # Default dimension for fallback

    def set_model(self, model: EmbeddingModelInterface) -> None:
        """Set the embedding model to use."""
        self._model = model

    def generate_embeddings_batch(
        self, sequences: List[Sequence], normalize: bool = True
    ) -> List[Embedding]:
        """Generate embeddings for a batch of sequences."""
        if not self._model:
            raise ValueError("Model must be set before generating embeddings")

        # Load model if not already loaded
        if not self._model.is_loaded():
            self._model.load_model()

        embeddings = []

        # Extract sequence strings for batch processing
        sequence_strings = [seq.sequence for seq in sequences]

        try:
            # Use model's batch processing if available
            embedding_matrix = self._model.embed_sequences_batch(sequence_strings)

            # Convert to individual Embedding objects
            for i, sequence in enumerate(sequences):
                embedding_vector = embedding_matrix[i]

                embedding = Embedding(
                    sequence_id=sequence.id,
                    vector=embedding_vector,
                    model_name=self._model.model_name,
                    embedding_dimension=len(embedding_vector),
                    metadata={
                        "sequence_length": sequence.length,
                        "sequence_type": sequence.sequence_type.value,
                        "model_max_length": self._model.max_sequence_length,
                        "normalized": normalize,
                    },
                )
                embeddings.append(embedding)

        except Exception as e:
            # Fallback: process sequences individually
            for sequence in sequences:
                try:
                    embedding_vector = self._model.embed_sequence(sequence.sequence)

                    embedding = Embedding(
                        sequence_id=sequence.id,
                        vector=embedding_vector,
                        model_name=self._model.model_name,
                        embedding_dimension=len(embedding_vector),
                        metadata={
                            "sequence_length": sequence.length,
                            "sequence_type": sequence.sequence_type.value,
                            "normalized": normalize,
                        },
                    )
                    embeddings.append(embedding)

                except Exception as seq_error:
                    # Create zero vector as fallback
                    zero_vector = np.zeros(
                        self._get_model_dimension(), dtype=np.float32
                    )
                    embedding = Embedding(
                        sequence_id=sequence.id,
                        vector=zero_vector,
                        model_name=self._model.model_name,
                        embedding_dimension=len(zero_vector),
                        metadata={"error": str(seq_error), "fallback": True},
                    )
                    embeddings.append(embedding)

        return embeddings

    def _generate_single_embedding(self, sequence: str) -> np.ndarray:
        """Generate embedding for a single sequence."""
        if not self._model:
            raise ValueError("Model must be set before generating embeddings")

        return self._model.embed_sequence(sequence)

    def _get_model_dimension(self) -> int:
        """Get the embedding dimension from the model."""
        if self._model:
            return self._model.embedding_dimension
        return self._fallback_dimension

    def validate_sequence_for_embedding(self, sequence: Sequence) -> bool:
        """Validate if a sequence can be embedded with this model."""
        if not self._model:
            return False

        return self._model.validate_sequence(sequence.sequence)
