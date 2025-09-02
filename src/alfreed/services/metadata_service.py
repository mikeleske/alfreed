"""Service for handling sequence metadata including taxonomy information."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..core.entities.taxonomy import parse_taxon_string


class MetadataService:
    """Service for managing sequence metadata including taxonomy."""

    def __init__(self):
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._taxonomy_cache: Dict[str, Dict[str, str]] = {}
        self._sequence_ids: List[str] = []
        self._taxonomy_df: Optional[pd.DataFrame] = None  # Store original DataFrame
        self._id_column: Optional[str] = None  # Store the ID column name
        self._logger = logging.getLogger(__name__)
        # Cache for taxonomy indices searches - significantly improves performance
        # when this method is called repeatedly with the same values
        self._taxonomy_indices_cache: Dict[str, List[int]] = {}

    def load_metadata_file(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """
        Load metadata from a file (CSV, Parquet, or JSON format).

        Args:
            file_path: Path to the metadata file

        Returns:
            Dictionary mapping sequence IDs to their metadata
        """
        self._logger.debug(f"Loading metadata from {file_path}")

        id_column = "ID"

        if not file_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {file_path}")

        try:
            if file_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(file_path)
            elif file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() == ".json":
                with open(file_path, "r") as f:
                    data = json.load(f)
                    df = pd.DataFrame(data)
            else:
                raise ValueError(
                    f"Unsupported metadata file format: {file_path.suffix}"
                )

            # Ensure 'ID' and 'Taxon' columns exist
            if "ID" not in df.columns or "Taxon" not in df.columns:
                raise ValueError(
                    "Metadata file must contain an 'ID' and 'Taxon' columns"
                )

            # Store the raw DataFrame and ID column for filtering operations
            self._taxonomy_df = df[["ID", "Taxon"]].copy()

            # Clear the taxonomy indices cache when loading new data
            self._taxonomy_indices_cache.clear()

            self._sequence_ids = df[id_column].tolist()
            self._taxons = df["Taxon"].tolist()
            self._parsed_taxons = [parse_taxon_string(taxon) for taxon in self._taxons]

            # Convert to dictionary mapping sequence ID to metadata
            # Use 'ID' column values as keys for each dictionary entry
            self._metadata_cache = df.set_index("ID").to_dict("index")
            self._taxonomy_cache = {
                seq_id: {"taxon": taxon, "taxonomy": parsed_taxon}
                for seq_id, taxon, parsed_taxon in zip(
                    self._sequence_ids, self._taxons, self._parsed_taxons
                )
            }

            self._logger.debug(
                f"Loaded metadata for {len(self._metadata_cache)} sequences"
            )
            return self._metadata_cache

        except Exception as e:
            self._logger.error(f"Failed to load metadata: {e}")
            raise RuntimeError(f"Failed to load metadata from {file_path}: {e}")

    def get_sequence_metadata(self, sequence_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific sequence ID."""
        return self._metadata_cache.get(sequence_id)

    def get_sequence_taxon(self, sequence_id: str) -> Optional[str]:
        """Get the taxonomic lineage string for a sequence."""
        return self._taxonomy_cache.get(sequence_id).get("taxon")

    def get_sequence_taxonomy(self, sequence_id: str) -> Optional[Dict[str, str]]:
        """Get parsed taxonomy dictionary for a sequence."""
        return self._taxonomy_cache.get(sequence_id).get("taxonomy")

    def get_available_sequence_ids(self) -> List[str]:
        """Get list of sequence IDs with available metadata."""
        return list(self._metadata_cache.keys())

    def get_taxonomy_indices(self, value: str) -> List[int]:
        """
        Get the indices of sequences that match the given taxonomy level and value.

        This method uses caching to significantly improve performance when called
        repeatedly with the same taxonomy values, which is common in batch operations.
        """
        if self._taxonomy_df is None:
            return []

        # Check if result is already cached
        if value in self._taxonomy_indices_cache:
            return self._taxonomy_indices_cache[value]

        # Perform the search with improved error handling
        try:
            matching_indices = self._taxonomy_df[
                self._taxonomy_df["Taxon"].str.contains(value, na=False, regex=False)
            ].index.tolist()
        except Exception as e:
            self._logger.warning(f"Error in taxonomy search for value '{value}': {e}")
            matching_indices = []

        # Cache the result
        self._taxonomy_indices_cache[value] = matching_indices
        return matching_indices

    def get_sequence_ids(self) -> List[str]:
        return self._sequence_ids

    def get_taxons(self) -> List[str]:
        return self._taxons

    def get_parsed_taxons(self) -> List[Dict[str, str]]:
        return self._parsed_taxons

    def clear_taxonomy_indices_cache(self) -> None:
        """Clear the taxonomy indices cache. Useful for testing or memory management."""
        self._taxonomy_indices_cache.clear()
        self._logger.debug("Cleared taxonomy indices cache")
