"""CLI interface components."""

from .commands import EmbedCommand, IndexCommand, SearchCommand
from .main import main

__all__ = ["main", "SearchCommand", "EmbedCommand", "IndexCommand"]
