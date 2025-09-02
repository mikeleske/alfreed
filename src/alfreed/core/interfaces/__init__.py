"""Abstract interfaces defining contracts for external dependencies."""

from .models import EmbeddingModelInterface
from .repositories import (
    EmbeddingRepositoryInterface,
    SequenceRepositoryInterface,
    VectorStoreRepositoryInterface,
)
from .services import (
    EmbeddingServiceInterface,
    SearchServiceInterface,
    SequenceServiceInterface,
)

__all__ = [
    "SequenceRepositoryInterface",
    "EmbeddingRepositoryInterface",
    "VectorStoreRepositoryInterface",
    "EmbeddingServiceInterface",
    "SearchServiceInterface",
    "SequenceServiceInterface",
    "EmbeddingModelInterface",
    #"ModelRegistry",
    #"model_registry",
]
