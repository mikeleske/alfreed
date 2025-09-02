"""Embedding service implementation."""

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from ..core.algorithms.embedding import EmbeddingGenerator
from ..core.entities.embedding import Embedding
from ..core.entities.sequence import Sequence
from ..core.interfaces.repositories import EmbeddingRepositoryInterface
from ..core.interfaces.services import EmbeddingServiceInterface
from ..infrastructure.models.dnabert_model import DNABERTModel
from alfreed.core.entities import embedding



class PersistentEmbeddingCache:
    """Thread-safe persistent disk cache for embedding generation results."""

    def __init__(
        self,
        cache_dir: str = "cache/embeddings",
        max_size_gb: float = 1.0,
        max_age_days: int = 30,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(
            max_size_gb * 1024 * 1024 * 1024
        )  # Convert GB to bytes
        self.max_age = timedelta(days=max_age_days)
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata file
        self.metadata_file = self.cache_dir / "metadata.json"
        self._load_metadata()

        # Clean up old/oversized cache on startup
        self._cleanup_cache()

    def _load_metadata(self):
        """Load cache metadata from disk."""
        with self.lock:
            if self.metadata_file.exists():
                try:
                    with open(self.metadata_file, "r") as f:
                        data = json.load(f)
                        self.hits = data.get("hits", 0)
                        self.misses = data.get("misses", 0)
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        f"Failed to load cache metadata: {e}"
                    )
                    self.hits = self.misses = 0
            else:
                self.hits = self.misses = 0

    def _save_metadata(self):
        """Save cache metadata to disk."""
        with self.lock:
            try:
                metadata = {
                    "hits": self.hits,
                    "misses": self.misses,
                    "last_updated": datetime.now().isoformat(),
                }
                with open(self.metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Failed to save cache metadata: {e}"
                )

    def _make_key(
        self,
        sequences: Tuple[Sequence, ...],
        model_name: str,
        batch_size: int,
        max_length: int,
        normalize: bool,
    ) -> str:
        """Create a deterministic cache key from the parameters."""
        import hashlib

        # Create a deterministic representation of sequences
        # NOTE: We exclude descriptions from the cache key to ensure that
        # sequences with the same ID and sequence content but different
        # descriptions will still cache together
        seq_parts = []
        for seq in sequences:
            seq_key = f"{seq.id}:{seq.sequence}:{seq.sequence_type.value}"
            # Intentionally NOT including description for better cache hits
            seq_parts.append(seq_key)

        sequences_str = "|".join(seq_parts)

        # Combine all parameters into a single string
        key_parts = [
            sequences_str,
            model_name,
            str(batch_size),
            str(max_length),
            str(normalize),
        ]
        key_string = "||".join(key_parts)

        # Return MD5 hash for consistent key length
        return hashlib.md5(key_string.encode("utf-8")).hexdigest()

    def _get_cache_file_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        # Use subdirectories to avoid too many files in one directory
        subdir = key[:2]  # Use first 2 chars of hash for subdirectory
        return self.cache_dir / subdir / f"{key}.pkl"

    def _get_metadata_path(self, key: str) -> Path:
        """Get the metadata file path for a cache key."""
        subdir = key[:2]
        return self.cache_dir / subdir / f"{key}.meta.json"

    def get(
        self,
        sequences: Tuple[Sequence, ...],
        model_name: str,
        batch_size: int,
        max_length: int,
        normalize: bool,
    ) -> Optional[List[Embedding]]:
        """Get cached result if it exists."""
        key = self._make_key(sequences, model_name, batch_size, max_length, normalize)
        cache_file = self._get_cache_file_path(key)
        metadata_file = self._get_metadata_path(key)

        with self.lock:
            if cache_file.exists():
                try:
                    # Check if cache entry is expired
                    if metadata_file.exists():
                        with open(metadata_file, "r") as f:
                            meta = json.load(f)
                            created_time = datetime.fromisoformat(meta["created"])
                            if datetime.now() - created_time > self.max_age:
                                # Cache entry is expired, remove it
                                cache_file.unlink(missing_ok=True)
                                metadata_file.unlink(missing_ok=True)
                                self.misses += 1
                                self._save_metadata()
                                return None

                    # Load cached result
                    result = joblib.load(cache_file)

                    # Update access time
                    if metadata_file.exists():
                        with open(metadata_file, "r") as f:
                            meta = json.load(f)
                        meta["last_accessed"] = datetime.now().isoformat()
                        meta["access_count"] = meta.get("access_count", 0) + 1
                        with open(metadata_file, "w") as f:
                            json.dump(meta, f)

                    self.hits += 1
                    self._save_metadata()
                    return result

                except Exception as e:
                    logging.getLogger(__name__).warning(
                        f"Failed to load cache file {cache_file}: {e}"
                    )
                    # Clean up corrupted cache file
                    cache_file.unlink(missing_ok=True)
                    metadata_file.unlink(missing_ok=True)

            self.misses += 1
            self._save_metadata()
            return None

    def put(
        self,
        sequences: Tuple[Sequence, ...],
        model_name: str,
        batch_size: int,
        max_length: int,
        normalize: bool,
        result: List[Embedding],
    ) -> None:
        """Store result in cache."""
        key = self._make_key(sequences, model_name, batch_size, max_length, normalize)
        cache_file = self._get_cache_file_path(key)
        metadata_file = self._get_metadata_path(key)

        with self.lock:
            try:
                # Create subdirectory if needed
                cache_file.parent.mkdir(parents=True, exist_ok=True)

                # Save the embedding result
                joblib.dump(result, cache_file)

                # Save metadata
                metadata = {
                    "created": datetime.now().isoformat(),
                    "last_accessed": datetime.now().isoformat(),
                    "access_count": 0,
                    "model_name": model_name,
                    "num_sequences": len(sequences),
                    "batch_size": batch_size,
                    "max_length": max_length,
                    "normalize": normalize,
                    "file_size": cache_file.stat().st_size,
                }

                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

                # Check if we need to cleanup cache due to size limits
                self._cleanup_cache_if_needed()

            except Exception as e:
                logging.getLogger(__name__).error(
                    f"Failed to save cache file {cache_file}: {e}"
                )

    def _cleanup_cache_if_needed(self):
        """Clean up cache if it exceeds size limits."""
        total_size = self._get_cache_size()
        if total_size > self.max_size_bytes:
            self._cleanup_cache()

    def _get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        total_size = 0
        try:
            for cache_file in self.cache_dir.rglob("*.pkl"):
                if cache_file.exists():
                    total_size += cache_file.stat().st_size
        except Exception as e:
            logging.getLogger(__name__).warning(f"Error calculating cache size: {e}")
        return total_size

    def _cleanup_cache(self):
        """Clean up expired entries and enforce size limits."""
        logger = logging.getLogger(__name__)

        try:
            now = datetime.now()
            cache_entries = []

            # Collect all cache entries with their metadata
            for cache_file in self.cache_dir.rglob("*.pkl"):
                if cache_file.exists():
                    metadata_file = cache_file.with_suffix(".meta.json")
                    try:
                        if metadata_file.exists():
                            with open(metadata_file, "r") as f:
                                meta = json.load(f)
                            created_time = datetime.fromisoformat(meta["created"])
                            last_accessed = datetime.fromisoformat(
                                meta.get("last_accessed", meta["created"])
                            )
                        else:
                            # Use file modification time as fallback
                            created_time = datetime.fromtimestamp(
                                cache_file.stat().st_mtime
                            )
                            last_accessed = created_time

                        # Remove expired entries
                        if now - created_time > self.max_age:
                            cache_file.unlink(missing_ok=True)
                            metadata_file.unlink(missing_ok=True)
                            continue

                        cache_entries.append(
                            {
                                "file": cache_file,
                                "metadata_file": metadata_file,
                                "last_accessed": last_accessed,
                                "size": cache_file.stat().st_size,
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Error processing cache file {cache_file}: {e}")
                        # Remove corrupted files
                        cache_file.unlink(missing_ok=True)
                        if metadata_file.exists():
                            metadata_file.unlink(missing_ok=True)

            # Enforce size limits by removing least recently accessed files
            cache_entries.sort(key=lambda x: x["last_accessed"])
            total_size = sum(entry["size"] for entry in cache_entries)

            while total_size > self.max_size_bytes and cache_entries:
                entry = cache_entries.pop(0)
                try:
                    entry["file"].unlink(missing_ok=True)
                    entry["metadata_file"].unlink(missing_ok=True)
                    total_size -= entry["size"]
                except Exception as e:
                    logger.warning(f"Error removing cache file: {e}")

            logger.info(
                f"Cache cleanup completed. {len(cache_entries)} entries remaining, {total_size / 1024 / 1024:.1f} MB used"
            )

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

    def clear(self) -> None:
        """Clear the entire cache."""
        with self.lock:
            try:
                # Remove all cache files
                for cache_file in self.cache_dir.rglob("*.pkl"):
                    cache_file.unlink(missing_ok=True)
                for meta_file in self.cache_dir.rglob("*.meta.json"):
                    meta_file.unlink(missing_ok=True)

                # Reset statistics
                self.hits = 0
                self.misses = 0
                self._save_metadata()

                logging.getLogger(__name__).info("Cache cleared successfully")
            except Exception as e:
                logging.getLogger(__name__).error(f"Error clearing cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

            # Count cache files
            num_entries = len(list(self.cache_dir.rglob("*.pkl")))
            cache_size = self._get_cache_size()

            return {
                "type": "persistent",
                "cache_dir": str(self.cache_dir),
                "num_entries": num_entries,
                "cache_size_mb": round(cache_size / 1024 / 1024, 2),
                "max_size_gb": round(self.max_size_bytes / 1024 / 1024 / 1024, 2),
                "max_age_days": self.max_age.days,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.1f}%",
            }


# Module-level cache instance (persistent)
_embedding_cache = PersistentEmbeddingCache(
    cache_dir="cache/embeddings",
    max_size_gb=1.0,  # 1GB cache limit
    max_age_days=30,  # Cache entries expire after 30 days
)


class EmbeddingService(EmbeddingServiceInterface):
    """Service for embedding generation business logic."""

    def __init__(
        self,
        embedding_repository: EmbeddingRepositoryInterface,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        cache_config: Optional[Dict[str, Any]] = None,
    ):
        self._embedding_repository = embedding_repository
        self._embedding_generator = embedding_generator or EmbeddingGenerator()
        self._logger = logging.getLogger(__name__)

        # Configure cache if custom settings provided
        if cache_config:
            self._configure_cache(cache_config)

    def _configure_cache(self, cache_config: Dict[str, Any]) -> None:
        """Configure the persistent cache with custom settings."""
        global _embedding_cache
        try:
            cache_dir = cache_config.get("cache_dir", "cache/embeddings")
            max_size_gb = cache_config.get("max_size_gb", 1.0)
            max_age_days = cache_config.get("max_age_days", 30)

            _embedding_cache = PersistentEmbeddingCache(
                cache_dir=cache_dir, max_size_gb=max_size_gb, max_age_days=max_age_days
            )

            self._logger.info(
                f"Configured persistent cache: {cache_dir}, max_size={max_size_gb}GB, max_age={max_age_days}days"
            )
        except Exception as e:
            self._logger.error(f"Failed to configure cache: {e}")
            # Keep using default cache

    def _instantiate_embedding_model(self, model_name: str):
        """Set the embedding generator."""
        embedding_model = DNABERTModel(model_name=model_name)
        self._embedding_generator.set_model(embedding_model)

    def generate_embeddings(
        self,
        sequences: List[Sequence],
        model_name: str,
        batch_size: int = 32,
        max_length: int = 512,  # This is token length, not nucleotide length
        normalize: bool = True,
    ) -> List[Embedding]:
        """
        Generate embeddings for sequences (with LRU caching).

        Args:
            sequences: List of DNA sequences to embed
            model_name: Name/ID of the embedding model to use
            batch_size: Number of sequences to process at once
            max_length: Maximum number of tokens (not nucleotides) for model input
            normalize: Whether to normalize the embeddings

        Returns:
            List of embedding objects with vector data and metadata
        """
        sequences_tuple = tuple(sequences)

        # Check cache first
        cached_result = _embedding_cache.get(
            sequences_tuple, model_name, batch_size, max_length, normalize
        )
        if cached_result is not None:
            self._logger.info(
                f"Persistent cache HIT: Returning cached embeddings for {len(sequences)} sequences with model {model_name}"
            )
            return cached_result

        self._logger.info(
            f"Persistent cache MISS: Will generate embeddings for {len(sequences)} sequences with model {model_name}"
        )

        # Instantiate the embedding model when cache is missed
        self._instantiate_embedding_model(model_name)

        # Generate embeddings if not in cache
        result = self._generate_embeddings_impl(
            sequences_tuple, model_name, batch_size, max_length, normalize
        )

        # Store result in cache
        _embedding_cache.put(
            sequences_tuple, model_name, batch_size, max_length, normalize, result
        )

        return result

    def _generate_embeddings_impl(
        self,
        sequences: Tuple[Sequence, ...],
        model_name: str,
        batch_size: int,
        max_length: int,
        normalize: bool,
    ) -> List[Embedding]:
        """
        Generate embeddings for sequences (actual implementation).

        Args:
            sequences: Tuple of DNA sequences to embed
            model_name: Name/ID of the embedding model to use
            batch_size: Number of sequences to process at once
            max_length: Maximum number of tokens (not nucleotides) for model input
            normalize: Whether to normalize the embeddings

        Returns:
            List of embedding objects with vector data and metadata
        """
        self._logger.info(
            f"Generating embeddings for {len(sequences)} sequences "
            f"using model {model_name}"
        )

        start_time = time.time()

        try:
            # Create the embedding model
            self._logger.info(f"Creating model: {model_name}")

            # Process sequences in batches
            all_embeddings = []

            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i:i + batch_size]
                self._logger.info(
                    f"Processing batch {i//batch_size + 1}/{(len(sequences) + batch_size - 1)//batch_size}"
                )

                batch_embeddings = self._embedding_generator.generate_embeddings_batch(
                    sequences=batch_sequences, normalize=normalize
                )

                all_embeddings.extend(batch_embeddings)

            elapsed_time = time.time() - start_time
            self._logger.info(
                f"Generated {len(all_embeddings)} embeddings in {elapsed_time:.2f} seconds"
            )

            return all_embeddings

        except Exception as e:
            self._logger.error(f"Failed to generate embeddings: {e}")
            raise

    def load_precomputed_embeddings(
        self,
        file_path: Path
    ) -> np.ndarray:
        """Load pre-computed embeddings from file."""
        self._logger.info(f"Loading precomputed embeddings from {file_path}")

        try:
            embedding_matrix = self._embedding_repository.load_embeddings(file_path)
            self._logger.info(
                f"Loaded {embedding_matrix.shape[0]} precomputed embeddings"
            )
            return embedding_matrix

        except Exception as e:
            self._logger.error(f"Failed to load precomputed embeddings: {e}")
            raise

    def save_embeddings(
        self,
        embeddings: List[Embedding],
        file_path: Path,
    ) -> None:
        """Save embeddings to file."""
        self._logger.info(f"Saving {len(embeddings)} embeddings to {file_path}")

        try:
            import numpy as np

            embedding_matrix = np.vstack([emb.vector for emb in embeddings])
            self._embedding_repository.save_embeddings(embedding_matrix, file_path)
            self._logger.info(f"Successfully saved embeddings to {file_path}")

        except Exception as e:
            self._logger.error(f"Failed to save embeddings: {e}")
            raise

    def get_embeddings_by_sequence_ids(self, sequence_ids: List[str]) -> np.ndarray:
        """Get embeddings for specific sequence IDs."""
        return self._embedding_repository.get_embeddings_for_sequences(sequence_ids)

    def clear_cache(self) -> None:
        """Clear the persistent embedding generation cache."""
        _embedding_cache.clear()
        self._logger.info("Persistent embedding generation cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding generation cache statistics."""
        return _embedding_cache.get_stats()
