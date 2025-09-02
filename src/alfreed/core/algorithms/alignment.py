"""Core sequence alignment algorithms."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..entities.search_result import SearchResult
from ..entities.sequence import Sequence


@dataclass
class AlignmentResult:
    """Represents the result of a pairwise sequence alignment."""

    query_aligned: str
    target_aligned: str
    alignment_score: float
    identity_percent: float
    gaps: int
    mismatches: int
    alignment_length: int


class SequenceAligner:
    """Core algorithm for sequence alignment."""

    def __init__(
        self,
        alignment_type: str = "local",
        gap_open_penalty: int = 10,
        gap_extend_penalty: int = 1,
        match_score: int = 2,
        mismatch_penalty: int = -1,
    ):
        """Initialize alignment parameters."""
        self.alignment_type = alignment_type
        self.gap_open_penalty = gap_open_penalty
        self.gap_extend_penalty = gap_extend_penalty
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self._alignment_engine = None

    def set_alignment_engine(self, engine: Any) -> None:
        """Inject alignment engine dependency (e.g., parasail)."""
        self._alignment_engine = engine

    def align_sequences(
        self, query_sequence: Sequence, target_sequence: Sequence
    ) -> AlignmentResult:
        """Align two sequences and return alignment result."""
        if not self._alignment_engine:
            raise ValueError("Alignment engine must be set before aligning sequences")

        # Perform the actual alignment using injected engine
        aligned_query, aligned_target, score = self._perform_alignment(
            query_sequence.sequence, target_sequence.sequence
        )

        # Calculate alignment statistics
        identity_percent = self._calculate_identity_percent(
            aligned_query, aligned_target
        )
        gaps = self._count_gaps(aligned_query, aligned_target)
        mismatches = self._count_mismatches(aligned_query, aligned_target)
        alignment_length = len(aligned_query)

        return AlignmentResult(
            query_aligned=aligned_query,
            target_aligned=aligned_target,
            alignment_score=score,
            identity_percent=identity_percent,
            gaps=gaps,
            mismatches=mismatches,
            alignment_length=alignment_length,
        )

    def add_alignment_to_search_results(
        self,
        results: List[SearchResult],
        query_sequences: Dict[str, Sequence],
        target_sequences: Dict[str, Sequence],
    ) -> List[SearchResult]:
        """Add alignment information to search results."""
        updated_results = []

        for result in results:
            query_seq = query_sequences.get(result.query_sequence_id)
            target_seq = target_sequences.get(result.matched_sequence_id)

            if query_seq and target_seq:
                try:
                    alignment_result = self.align_sequences(query_seq, target_seq)

                    # Update result with alignment information
                    result.alignment_score = alignment_result.alignment_score
                    result.identity_percent = alignment_result.identity_percent

                    if result.metadata is None:
                        result.metadata = {}

                    result.metadata.update(
                        {
                            "alignment_length": alignment_result.alignment_length,
                            "gaps": alignment_result.gaps,
                            "mismatches": alignment_result.mismatches,
                            "query_aligned": alignment_result.query_aligned,
                            "target_aligned": alignment_result.target_aligned,
                        }
                    )

                except Exception as e:
                    # Add error information to metadata
                    if result.metadata is None:
                        result.metadata = {}
                    result.metadata["alignment_error"] = str(e)

            updated_results.append(result)

        return updated_results

    def _perform_alignment(
        self, query_seq: str, target_seq: str
    ) -> Tuple[str, str, float]:
        """Perform the actual sequence alignment."""
        # This is where the actual alignment algorithm runs
        # Implementation depends on the injected alignment engine
        raise NotImplementedError(
            "Alignment logic must be implemented by infrastructure layer"
        )

    def _calculate_identity_percent(
        self, aligned_seq1: str, aligned_seq2: str
    ) -> float:
        """Calculate percent identity for aligned sequences."""
        if len(aligned_seq1) != len(aligned_seq2):
            raise ValueError("Aligned sequences must have the same length")

        matches = sum(
            1
            for a, b in zip(aligned_seq1, aligned_seq2)
            if a == b and a != "-" and b != "-"
        )

        aligned_length = sum(
            1 for a, b in zip(aligned_seq1, aligned_seq2) if a != "-" and b != "-"
        )

        return (matches / aligned_length * 100) if aligned_length > 0 else 0.0

    def _count_gaps(self, aligned_seq1: str, aligned_seq2: str) -> int:
        """Count total gaps in the alignment."""
        return aligned_seq1.count("-") + aligned_seq2.count("-")

    def _count_mismatches(self, aligned_seq1: str, aligned_seq2: str) -> int:
        """Count mismatches in the alignment (excluding gaps)."""
        return sum(
            1
            for a, b in zip(aligned_seq1, aligned_seq2)
            if a != b and a != "-" and b != "-"
        )

    def batch_align_sequences(
        self, sequence_pairs: List[Tuple[Sequence, Sequence]]
    ) -> List[AlignmentResult]:
        """Perform batch alignment of sequence pairs."""
        results = []

        for query_seq, target_seq in sequence_pairs:
            try:
                alignment_result = self.align_sequences(query_seq, target_seq)
                results.append(alignment_result)
            except Exception as e:
                # Create error result
                error_result = AlignmentResult(
                    query_aligned="",
                    target_aligned="",
                    alignment_score=0.0,
                    identity_percent=0.0,
                    gaps=0,
                    mismatches=0,
                    alignment_length=0,
                )
                results.append(error_result)

        return results
