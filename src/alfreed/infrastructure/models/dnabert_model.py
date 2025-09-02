"""DNABERT-2 model implementation."""

import logging
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import logging as hf_logging

from ...core.interfaces.models import EmbeddingModelInterface
from ..config.settings import get_settings

# Suppress HuggingFace warnings
hf_logging.set_verbosity_error()


class DNABERTModel(EmbeddingModelInterface):
    """DNABERT-2 embedding model implementation."""

    def __init__(
        self,
        model_name: str = "zhihan1996/DNABERT-2-117M",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self._model_name = model_name
        self._device = device or self._get_device()
        self._cache_dir = cache_dir or get_settings().model.cache_dir
        self._model = None
        self._tokenizer = None
        self._logger = logging.getLogger(__name__)

        # DNABERT-2 specific settings
        self._embedding_dimension = 768
        self._max_token_length = (
            get_settings().model.max_token_length
        )  # Maximum number of tokens the model can handle
        # DNABERT-2 uses BPE tokenization, so sequence length limit depends on BPE compression
        # Let the tokenizer handle truncation automatically

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    @property
    def max_sequence_length(self) -> int:
        """Get the maximum DNA sequence length (approximate, depends on BPE compression)."""
        # DNABERT-2 uses BPE tokenization, so this is an approximation
        # The actual limit depends on how well BPE compresses the sequence
        # Conservative estimate: most sequences will fit within this range
        return self._max_token_length * 4.5  # Conservative estimate for BPE compression

    def _get_device(self) -> str:
        """Determine the best device to use."""
        settings = get_settings()
        device_setting = settings.model.device.lower()

        if device_setting == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        else:
            return device_setting

    def load_model(self) -> None:
        """Load DNABERT-2 model and tokenizer."""
        if self.is_loaded():
            return

        self._logger.info(
            f"Loading DNABERT-2 model '{self._model_name}' on device '{self._device}'"
        )

        try:
            # Load tokenizer
            self._logger.info("Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=True, cache_dir=self._cache_dir
            )

            # Load model
            self._logger.info("Loading model...")
            self._model = AutoModel.from_pretrained(
                self._model_name, trust_remote_code=True, cache_dir=self._cache_dir
            ).to(self._device)

            # Set to evaluation mode
            self._model.eval()

            # Update actual embedding dimension from model
            if hasattr(self._model.config, "hidden_size"):
                self._embedding_dimension = self._model.config.hidden_size

            self._logger.info(
                f"Successfully loaded DNABERT-2 model with dimension {self._embedding_dimension}"
            )
            self._logger.info(
                f"Token limit: {self._max_token_length} (BPE tokenization)"
            )

        except Exception as e:
            self._logger.error(f"Failed to load DNABERT-2 model: {e}")
            raise RuntimeError(f"Error loading DNABERT-2 model: {e}")

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None and self._tokenizer is not None

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Generate embedding for a single sequence using DNABERT-2."""
        if not self.is_loaded():
            self.load_model()

        try:
            # Preprocess sequence
            processed_seq = self.preprocess_sequence(sequence)

            # Tokenize (use token limit, not sequence limit)
            inputs = self._tokenizer(
                processed_seq,
                return_tensors="pt",
                max_length=self._max_token_length,
                truncation=True,
                padding=True,
            )["input_ids"].to(self._device)

            # Generate embedding
            with torch.no_grad():
                hidden_states = self._model(inputs)

            # Mean pooling (DNABERT-2 specific)
            embedding = hidden_states[0].mean(dim=1).squeeze().cpu().numpy()

            return embedding.astype(np.float32)

        except Exception as e:
            self._logger.warning(
                f"Failed to embed sequence: {sequence[:50]}... Error: {e}"
            )
            # Return zero vector as fallback
            return np.zeros(self._embedding_dimension, dtype=np.float32)

    def embed_sequences_batch(self, sequences: List[str]) -> np.ndarray:
        """Generate embeddings for multiple sequences."""
        if not self.is_loaded():
            self.load_model()

        embeddings = []

        # Process sequences individually for now
        # TODO: Implement true batch processing for better efficiency
        for sequence in sequences:
            embedding = self.embed_sequence(sequence)
            embeddings.append(embedding)

        return np.stack(embeddings)

    def validate_sequence(self, sequence: str) -> bool:
        """Validate sequence for DNABERT-2."""
        # Basic validation
        if not super().validate_sequence(sequence):
            return False

        # DNABERT-2 specific validation (DNA sequences only)
        valid_chars = set("ATCGNRYSWKMBDHV-")
        return set(sequence.upper()) <= valid_chars

    def preprocess_sequence(self, sequence: str) -> str:
        """Preprocess sequence for DNABERT-2."""
        return sequence.upper()

