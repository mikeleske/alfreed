"""Configuration settings management."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    """Configuration for ML models."""

    default_embedding_model: str = "zhihan1996/DNABERT-2-117M"
    max_token_length: int = 512  # Maximum number of tokens (not nucleotides)
    batch_size: int = 32
    device: str = "auto"  # "auto", "cpu", "cuda"
    cache_dir: Optional[str] = None


@dataclass
class SearchConfig:
    """Configuration for search operations."""

    default_k: int = 10
    default_metric: str = "cosine"
    similarity_threshold: float = 0.7
    max_results_per_query: int = 100
    enable_alignment: bool = False
    exclude_self_matches: bool = True


@dataclass
class AlignmentConfig:
    """Configuration for sequence alignment."""

    alignment_type: str = "local"  # "local" or "global"
    gap_open_penalty: int = 10
    gap_extend_penalty: int = 1
    match_score: int = 2
    mismatch_penalty: int = -1


@dataclass
class StorageConfig:
    """Configuration for data storage."""

    data_dir: Path = field(default_factory=lambda: Path("./data"))
    cache_dir: Path = field(default_factory=lambda: Path("./cache"))
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    temp_dir: Path = field(default_factory=lambda: Path("./tmp"))

    def __post_init__(self):
        # Convert strings to Path objects
        self.data_dir = Path(self.data_dir)
        self.cache_dir = Path(self.cache_dir)
        self.output_dir = Path(self.output_dir)
        self.temp_dir = Path(self.temp_dir)


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = False
    log_file: Optional[Path] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class Settings:
    """Main application settings."""

    # Core settings
    app_name: str = "alfreed"
    version: str = "0.2.0"
    debug: bool = False

    # Component configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Environment-specific overrides
    environment: str = "production"

    @classmethod
    def from_file(cls, config_path: Path) -> "Settings":
        """Load settings from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        return cls.from_dict(config_data)

    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> "Settings":
        """Create settings from dictionary."""
        settings = cls()

        # Update core settings
        for key, value in config_data.items():
            if hasattr(settings, key) and not isinstance(
                getattr(settings, key),
                (
                    ModelConfig,
                    SearchConfig,
                    AlignmentConfig,
                    StorageConfig,
                    LoggingConfig,
                ),
            ):
                setattr(settings, key, value)

        # Update component configurations
        if "model" in config_data:
            settings.model = ModelConfig(**config_data["model"])

        if "search" in config_data:
            settings.search = SearchConfig(**config_data["search"])

        if "alignment" in config_data:
            settings.alignment = AlignmentConfig(**config_data["alignment"])

        if "storage" in config_data:
            settings.storage = StorageConfig(**config_data["storage"])

        if "logging" in config_data:
            settings.logging = LoggingConfig(**config_data["logging"])

        return settings

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        settings = cls()

        # Core settings from env
        settings.debug = os.getenv("ALFREED_DEBUG", "false").lower() == "true"
        settings.environment = os.getenv("ALFREED_ENV", "production")

        # Model settings from env
        if os.getenv("ALFREED_MODEL_NAME"):
            settings.model.default_embedding_model = os.getenv("ALFREED_MODEL_NAME")

        if os.getenv("ALFREED_BATCH_SIZE"):
            settings.model.batch_size = int(os.getenv("ALFREED_BATCH_SIZE"))

        if os.getenv("ALFREED_DEVICE"):
            settings.model.device = os.getenv("ALFREED_DEVICE")

        # Storage settings from env
        if os.getenv("ALFREED_DATA_DIR"):
            settings.storage.data_dir = Path(os.getenv("ALFREED_DATA_DIR"))

        if os.getenv("ALFREED_OUTPUT_DIR"):
            settings.storage.output_dir = Path(os.getenv("ALFREED_OUTPUT_DIR"))

        # Logging settings from env
        if os.getenv("ALFREED_LOG_LEVEL"):
            settings.logging.level = os.getenv("ALFREED_LOG_LEVEL").upper()

        return settings

    def create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.storage.data_dir,
            self.storage.cache_dir,
            self.storage.output_dir,
            self.storage.temp_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def validate(self) -> List[str]:
        """Validate settings and return list of errors."""
        errors = []

        # Validate model settings
        if self.model.batch_size <= 0:
            errors.append("Model batch size must be positive")

        if self.model.max_token_length <= 0:
            errors.append("Model max token length must be positive")

        # Validate search settings
        if self.search.default_k <= 0:
            errors.append("Search k must be positive")

        if not 0 <= self.search.similarity_threshold <= 1:
            errors.append("Similarity threshold must be between 0 and 1")

        # Validate alignment settings
        if self.alignment.alignment_type not in ["local", "global"]:
            errors.append("Alignment type must be 'local' or 'global'")

        return errors


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings

    if _settings is None:
        # Try to load from config file first
        config_path = Path("config") / "production.yaml"

        if config_path.exists():
            _settings = Settings.from_file(config_path)

        # Fallback to environment variables
        if _settings is None:
            _settings = Settings.from_env()

        # Validate settings
        errors = _settings.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

        # Create necessary directories
        _settings.create_directories()

    return _settings


def reload_settings() -> None:
    """Reload settings from configuration sources."""
    global _settings
    _settings = None
    get_settings()
