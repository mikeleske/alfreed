"""Logging configuration setup."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from .settings import LoggingConfig


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """Setup logging configuration."""
    if config is None:
        from .settings import get_settings

        config = get_settings().logging

    # Create formatter
    formatter = logging.Formatter(config.format)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if requested
    if config.log_to_file and config.log_file:
        log_file_path = Path(config.log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set specific logger levels for external libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)

    # Create alfreed-specific logger
    alfreed_logger = logging.getLogger("alfreed")
    alfreed_logger.setLevel(getattr(logging, config.level.upper()))


class LoggerMixin:
    """Mixin class to add logging capability to other classes."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return logging.getLogger(f"alfreed.{self.__class__.__name__}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the alfreed namespace."""
    return logging.getLogger(f"alfreed.{name}")
