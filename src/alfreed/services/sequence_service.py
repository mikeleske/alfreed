"""Sequence service implementation."""

import logging
from pathlib import Path
from typing import List, Optional

from ..core.entities.sequence import Sequence, SequenceType
from ..core.interfaces.repositories import SequenceRepositoryInterface
from ..core.interfaces.services import SequenceServiceInterface


class SequenceService(SequenceServiceInterface):
    """Service for sequence processing business logic."""

    def __init__(self, sequence_repository: SequenceRepositoryInterface):
        self._sequence_repository = sequence_repository
        self._logger = logging.getLogger(__name__)

    def load_sequences_from_fasta(self, file_path: Path) -> List[Sequence]:
        """Load and validate sequences from FASTA file."""
        self._logger.info(f"Loading sequences from FASTA file: {file_path}")

        try:
            sequences = self._sequence_repository.load_from_fasta(file_path)
            self._logger.info(f"Loaded {len(sequences)} sequences from {file_path}")

            # Validate sequences
            validated_sequences = self.validate_sequences(sequences)

            if len(validated_sequences) != len(sequences):
                self._logger.warning(
                    f"Filtered out {len(sequences) - len(validated_sequences)} invalid sequences"
                )

            return validated_sequences

        except Exception as e:
            self._logger.error(f"Failed to load sequences from {file_path}: {e}")
            raise

    def validate_sequences(self, sequences: List[Sequence]) -> List[Sequence]:
        """Validate sequence data and filter invalid sequences."""
        validated_sequences = []

        for sequence in sequences:
            if self._is_valid_sequence(sequence):
                validated_sequences.append(sequence)
            else:
                self._logger.warning(f"Skipping invalid sequence: {sequence.id}")

        return validated_sequences

    def _is_valid_sequence(self, sequence: Sequence) -> bool:
        """Check if a sequence is valid."""
        # Basic validation rules
        if not sequence.id or not sequence.sequence:
            return False

        if len(sequence.sequence) < 10:  # Minimum sequence length
            return False

        if len(sequence.sequence) > 50000:  # Maximum sequence length
            return False

        # Check for valid characters based on sequence type
        if sequence.sequence_type == SequenceType.DNA:
            valid_chars = sequence.get_valid_dna_characters()
            return set(sequence.sequence.upper()) <= valid_chars

        return True

    def preprocess_sequences(
        self,
        sequences: List[Sequence],
        max_length: Optional[int] = None,
        remove_duplicates: bool = False,
        min_length: int = 10,
    ) -> List[Sequence]:
        """Preprocess sequences (trimming, filtering, etc.)."""
        self._logger.info(f"Preprocessing {len(sequences)} sequences")

        processed_sequences = []
        seen_sequences = set()

        for sequence in sequences:
            # Apply length filtering
            if len(sequence.sequence) < min_length:
                continue

            # Trim if necessary
            trimmed_sequence = sequence
            if max_length and len(sequence.sequence) > max_length:
                trimmed_sequence = Sequence(
                    id=sequence.id,
                    sequence=sequence.sequence[:max_length],
                    description=f"{sequence.description} (trimmed to {max_length} bp)",
                    sequence_type=sequence.sequence_type,
                )

            # Remove duplicates if requested
            if remove_duplicates:
                if trimmed_sequence.sequence in seen_sequences:
                    continue
                seen_sequences.add(trimmed_sequence.sequence)

            processed_sequences.append(trimmed_sequence)

        self._logger.info(f"Preprocessed to {len(processed_sequences)} sequences")
        return processed_sequences
