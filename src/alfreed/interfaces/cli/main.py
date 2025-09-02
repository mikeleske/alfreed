"""Main CLI entry point."""

import argparse
import sys
import os
import warnings
from pathlib import Path
from typing import List, Optional

# Suppress HuggingFace messages at CLI entry point
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['HF_HUB_VERBOSITY'] = 'error'

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from ...infrastructure.config.logging import setup_logging
    from ...infrastructure.config.settings import get_settings
    from .commands import EmbedCommand, IndexCommand, SearchCommand
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(
        "Make sure all dependencies are installed: pip install -r requirements/base.txt"
    )
    sys.exit(1)


class AlfreedCLI:
    """Main CLI application."""

    def __init__(self):
        self.settings = get_settings()
        setup_logging()

        self.commands = {
            "search": SearchCommand(),
            "embed": EmbedCommand(),
            "index": IndexCommand(),
        }

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser."""
        parser = argparse.ArgumentParser(
            prog="alfreed",
            description="DNA sequence similarity search using DNABERT-2 and FAISS",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        parser.add_argument(
            "--version", action="version", version=f"%(prog)s {self.settings.version}"
        )

        parser.add_argument("--config", type=Path, help="Path to configuration file")

        parser.add_argument("--debug", action="store_true", help="Enable debug logging")

        # Create subparsers for commands
        subparsers = parser.add_subparsers(
            dest="command", help="Available commands", metavar="COMMAND"
        )

        # Add command parsers
        for command_name, command in self.commands.items():
            command_parser = subparsers.add_parser(
                command_name,
                help=command.help,
                description=command.description,
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
            command.add_arguments(command_parser)

        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI application."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)

        # Handle no command provided
        if not parsed_args.command:
            parser.print_help()
            return 1

        # Update settings if config file provided
        if parsed_args.config:
            from ...infrastructure.config.settings import Settings

            self.settings = Settings.from_file(parsed_args.config)

        # Update debug mode
        if parsed_args.debug:
            self.settings.debug = True
            self.settings.logging.level = "DEBUG"
            setup_logging()

        # Execute the command
        try:
            command = self.commands[parsed_args.command]
            return command.execute(parsed_args, self.settings)

        except KeyboardInterrupt:
            print("\n⚠️ Operation cancelled by user")
            return 130

        except Exception as e:
            if self.settings.debug:
                import traceback

                traceback.print_exc()
            else:
                print(f"❌ Error: {e}")
            return 1


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    cli = AlfreedCLI()
    return cli.run(args)


if __name__ == "__main__":
    sys.exit(main())
