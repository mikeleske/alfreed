"""Vector store repository implementation using FAISS."""

from typing import Any, Dict, Optional, Tuple

import faiss
import numpy as np

from ..core.interfaces.repositories import VectorStoreRepositoryInterface


class VectorStoreRepository(VectorStoreRepositoryInterface):
    """Implementation of vector store repository using FAISS."""

    def __init__(self):
        self._index: Optional[faiss.Index] = None
        self._index_metadata: Dict[str, Any] = {}

    def create_index(
        self, embeddings: np.ndarray, metric: str = "cosine"
    ) -> faiss.Index:
        """Create a FAISS index from embeddings."""
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings array, got shape {embeddings.shape}"
            )

        if embeddings.shape[0] == 0:
            raise ValueError("Cannot create index from empty embeddings array")

        # Ensure float32 format for FAISS
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]

        try:
            # Create index based on metric
            if metric == "cosine":
                # Normalize vectors for cosine similarity
                faiss.normalize_L2(embeddings)
                index = faiss.IndexFlatIP(dim)
            elif metric == "l2":
                index = faiss.IndexFlatL2(dim)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            # Add vectors to index
            index.add(embeddings)

            self._index = index
            self._index_metadata = {
                "dimension": dim,
                "total_vectors": embeddings.shape[0],
                "metric": metric,
            }

            return index

        except Exception as e:
            raise RuntimeError(f"Failed to create FAISS index: {e}")

    def search_similar_vectors(
        self, index: faiss.Index, query_vectors: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors in the index."""
        if query_vectors.ndim != 2:
            raise ValueError(
                f"Expected 2D query array, got shape {query_vectors.shape}"
            )

        if query_vectors.shape[1] != index.d:
            raise ValueError(
                f"Query dimension ({query_vectors.shape[1]}) "
                f"doesn't match index dimension ({index.d})"
            )

        if k <= 0:
            raise ValueError("k must be positive")

        if k > index.ntotal:
            k = index.ntotal  # Limit k to available vectors

        try:
            # Ensure float32 format
            if query_vectors.dtype != np.float32:
                query_vectors = query_vectors.astype(np.float32)

            # Normalize for inner product indices
            if isinstance(index, faiss.IndexFlatIP):
                faiss.normalize_L2(query_vectors)

            # Perform search
            distances, indices = index.search(query_vectors, k)

            return distances, indices

        except Exception as e:
            raise RuntimeError(f"Failed to search vectors: {e}")

    def add_vectors_to_index(self, index: faiss.Index, vectors: np.ndarray) -> None:
        """Add new vectors to an existing index."""
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D vectors array, got shape {vectors.shape}")

        if vectors.shape[1] != index.d:
            raise ValueError(
                f"Vector dimension ({vectors.shape[1]}) "
                f"doesn't match index dimension ({index.d})"
            )

        try:
            # Ensure float32 format
            if vectors.dtype != np.float32:
                vectors = vectors.astype(np.float32)

            # Normalize for inner product indices
            if isinstance(index, faiss.IndexFlatIP):
                faiss.normalize_L2(vectors)

            # Add vectors
            index.add(vectors)

            # Update metadata if this is our current index
            if index is self._current_index:
                self._index_metadata["total_vectors"] = index.ntotal

        except Exception as e:
            raise RuntimeError(f"Failed to add vectors to index: {e}")

    def create_ivf_index(
        self, embeddings: np.ndarray, n_clusters: int = 100, metric: str = "cosine"
    ) -> faiss.Index:
        """Create an IVF (Inverted File) index for larger datasets."""
        if embeddings.shape[0] < n_clusters:
            raise ValueError(f"Need at least {n_clusters} vectors for IVF index")

        # Ensure float32 format
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]

        try:
            # Create quantizer
            if metric == "cosine":
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(
                    quantizer, dim, n_clusters, faiss.METRIC_COSINE
                )
                faiss.normalize_L2(embeddings)
            else:
                quantizer = faiss.IndexFlatL2(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_L2)

            # Train the index
            index.train(embeddings)

            # Add vectors
            index.add(embeddings)

            self._index = index
            self._index_metadata = {
                "dimension": dim,
                "total_vectors": embeddings.shape[0],
                "metric": metric,
                "index_type": "IVF",
                "n_clusters": n_clusters,
            }

            return index

        except Exception as e:
            raise RuntimeError(f"Failed to create IVF index: {e}")

    def get_index_info(self, index: faiss.Index) -> Dict[str, Any]:
        """Get information about a FAISS index."""
        return {
            "dimension": index.d,
            "total_vectors": index.ntotal,
            "is_trained": index.is_trained,
            "index_type": type(index).__name__,
            "metric_type": getattr(index, "metric_type", "unknown"),
        }

    def get_index(self) -> Optional[faiss.Index]:
        """Get the current active index."""
        return self._index

    def get_index_metadata(self) -> Dict[str, Any]:
        """Get metadata for the current index."""
        return self._index_metadata.copy()
