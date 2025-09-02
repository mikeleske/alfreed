"""Sequence repository implementation."""

import gzip
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from Bio import SeqIO

from ..core.entities.sequence import Sequence, SequenceType
from ..core.interfaces.repositories import SequenceRepositoryInterface


class SequenceRepository(SequenceRepositoryInterface):
    """Implementation of sequence repository for file-based storage."""

    def __init__(self):
        self._sequence_cache: Dict[str, Sequence] = {}

    def load_from_fasta(self, file_path: Path) -> List[Sequence]:
        """Load sequences from a FASTA file."""
        if not file_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {file_path}")

        valid_extensions = {".fasta", ".fa", ".fasta.gz", ".fa.gz"}
        suffix = "".join(file_path.suffixes).lower()

        if suffix not in valid_extensions:
            raise ValueError(
                f"Unsupported file extension '{suffix}'. "
                f"Expected one of: {', '.join(valid_extensions)}"
            )

        sequences = []

        try:
            if suffix.endswith(".gz"):
                with gzip.open(file_path, "rt") as handle:
                    sequences = self._parse_fasta_records(handle)
            else:
                with open(file_path, "r") as handle:
                    sequences = self._parse_fasta_records(handle)
        except Exception as e:
            raise RuntimeError(f"Failed to parse FASTA file: {e}")

        if not sequences:
            raise ValueError(f"No valid sequences found in FASTA file: {file_path}")

        # Cache sequences for quick retrieval
        for seq in sequences:
            self._sequence_cache[seq.id] = seq

        return sequences

    def _parse_fasta_records(self, handle) -> List[Sequence]:
        """Parse FASTA records from a file handle."""
        sequences = []

        for record in SeqIO.parse(handle, "fasta"):
            if len(record.seq) == 0:
                continue  # Skip empty sequences

            sequence = Sequence(
                id=record.id,
                sequence=str(record.seq).upper(),
                description=record.description,
                sequence_type=SequenceType.DNA,  # Assuming DNA sequences
                length=len(record.seq),
            )
            sequences.append(sequence)

        return sequences

    def load_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Load sequence metadata from a Parquet file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {file_path}")

        try:
            if file_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(file_path)
            elif file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            else:
                raise ValueError(
                    f"Unsupported metadata file format: {file_path.suffix}"
                )

            return df.to_dict("records")

        except Exception as e:
            raise RuntimeError(f"Failed to load metadata: {e}")

    def save_sequences(self, sequences: List[Sequence], file_path: Path) -> None:
        """Save sequences to a FASTA file."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w") as handle:
                for sequence in sequences:
                    handle.write(f">{sequence.id}")
                    if sequence.description and sequence.description != sequence.id:
                        handle.write(f" {sequence.description}")
                    handle.write(f"\n{sequence.sequence}\n")

        except Exception as e:
            raise RuntimeError(f"Failed to save sequences: {e}")

    def get_sequence_by_id(self, sequence_id: str) -> Optional[Sequence]:
        """Retrieve a sequence by its ID from cache or storage."""
        return self._sequence_cache.get(sequence_id)

    def validate_sequence_format(
        self, sequence_str: str, sequence_type: SequenceType
    ) -> bool:
        """Validate sequence format based on type."""
        if sequence_type == SequenceType.DNA:
            # Include IUPAC ambiguous nucleotide codes
            valid_chars = set("ATCGNRYSWKMBDHV-")
            return set(sequence_str.upper()) <= valid_chars

        return False

    def get_sequences_by_ids(self, sequence_ids: List[str]) -> List[Sequence]:
        """Get multiple sequences by their IDs."""
        sequences = []
        for seq_id in sequence_ids:
            seq = self.get_sequence_by_id(seq_id)
            if seq:
                sequences.append(seq)
        return sequences

    def clear_cache(self) -> None:
        """Clear the sequence cache."""
        self._sequence_cache.clear()
