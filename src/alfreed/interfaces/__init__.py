"""Interface layer for external APIs and CLI."""

from .cli.main import main as cli_main

__all__ = ["cli_main"]
