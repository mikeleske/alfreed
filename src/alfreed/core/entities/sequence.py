"""Sequence domain entity."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SequenceType(Enum):
    """Supported sequence types."""

    DNA = "dna"


@dataclass(frozen=True)
class Sequence:
    """Represents a biological sequence with metadata."""

    id: str
    sequence: str
    description: Optional[str] = None
    sequence_type: SequenceType = SequenceType.DNA
    length: Optional[int] = None

    def __post_init__(self):
        """Validate sequence data after initialization."""
        if not self.sequence:
            raise ValueError("Sequence cannot be empty")

        if not self.id:
            raise ValueError("Sequence ID cannot be empty")

        # Auto-calculate length if not provided
        if self.length is None:
            object.__setattr__(self, "length", len(self.sequence))

        # Validate sequence characters based on type
        if self.sequence_type == SequenceType.DNA:
            # Include IUPAC ambiguous nucleotide codes
            valid_chars = set(
                "ATCGNRYSWKMBDHV-"
            )  # Standard bases + IUPAC ambiguous codes
            invalid_chars = set(self.sequence.upper()) - valid_chars
            if invalid_chars:
                raise ValueError(
                    f"Invalid DNA sequence characters found: {sorted(invalid_chars)}. "
                    f"Valid characters: A, T, C, G, N (unknown), IUPAC ambiguous codes (R, Y, S, W, K, M, B, D, H, V), and gaps (-)"
                )

    def to_standard_bases(self, ambiguous_to: str = "N") -> "Sequence":
        """Convert IUPAC ambiguous codes to standard bases or N."""
        if self.sequence_type != SequenceType.DNA:
            return self

        # IUPAC code mapping (convert ambiguous to most common or to N)
        iupac_to_standard = {
            "R": "G",  # A or G -> G (most common in many contexts)
            "Y": "C",  # C or T -> C
            "S": "G",  # G or C -> G
            "W": "A",  # A or T -> A
            "K": "G",  # G or T -> G
            "M": "A",  # A or C -> A
            "B": "G",  # C, G, T -> G
            "D": "G",  # A, G, T -> G
            "H": "A",  # A, C, T -> A
            "V": "G",  # A, C, G -> G
        }

        # If user prefers N for all ambiguous codes
        if ambiguous_to == "N":
            iupac_to_standard = {code: "N" for code in iupac_to_standard.keys()}

        standardized_sequence = self.sequence.upper()
        for iupac_code, standard_base in iupac_to_standard.items():
            standardized_sequence = standardized_sequence.replace(
                iupac_code, standard_base
            )

        return Sequence(
            id=self.id,
            sequence=standardized_sequence,
            description=f"{self.description} (IUPAC converted)",
            sequence_type=self.sequence_type,
        )

    @staticmethod
    def get_valid_dna_characters() -> set:
        """Get the set of valid DNA characters including IUPAC codes."""
        return set("ATCGNRYSWKMBDHV-")
