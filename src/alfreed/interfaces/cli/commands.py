"""CLI command implementations."""

import argparse
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

# Suppress verbose HuggingFace messages globally
import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['PYTHONWARNINGS'] = 'ignore'

from ...infrastructure.config.settings import Settings, get_settings
from ...repositories.consensus_repository import ConsensusRepository
from ...repositories.embedding_repository import EmbeddingRepository
from ...repositories.result_repository import ResultRepository
from ...repositories.sequence_repository import SequenceRepository
from ...repositories.vector_store_repository import VectorStoreRepository
from ...services.consensus_service import ConsensusService
from ...services.embedding_service import EmbeddingService
from ...services.metadata_service import MetadataService
from ...services.search_service import SearchService
from ...services.sequence_service import SequenceService


class BaseCommand(ABC):
    """Base class for CLI commands."""

    def __init__(self):
        self.logger = logging.getLogger(f"alfreed.cli.{self.__class__.__name__}")

    @property
    @abstractmethod
    def help(self) -> str:
        """Short help text for the command."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Detailed description of the command."""
        pass

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments to the parser."""
        pass

    @abstractmethod
    def execute(self, args: argparse.Namespace, settings: Settings) -> int:
        """Execute the command."""
        pass


class SearchCommand(BaseCommand):
    """Command for performing similarity search."""

    @property
    def help(self) -> str:
        return "Perform DNA sequence similarity search"

    @property
    def description(self) -> str:
        return """
Perform DNA sequence similarity search using DNABERT-2 embeddings and FAISS.

This command can work with:
- Pre-computed embeddings and database index
- FASTA files (will generate embeddings on-the-fly)
- Mixed input types

Examples:
  # Search with pre-computed embeddings
  alfreed search --database-embeddings db.npy --database-metadata db.parquet \\
                 --query-embeddings query.npy --k 10

  # Search FASTA against pre-computed database
  alfreed search --database-embeddings db.npy --database-metadata db.parquet \\
                 --query-fasta queries.fasta --k 10 --embed-model DNABERT-2

  # Full FASTA-to-FASTA search
  alfreed search --database-fasta database.fasta --query-fasta queries.fasta \\
                 --k 10 --embed-model DNABERT-2
        """

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # Database arguments
        db_group = parser.add_argument_group("Database Options")
        db_input = db_group.add_mutually_exclusive_group(required=True)
        db_input.add_argument(
            "--database-embeddings",
            type=Path,
            help="Path to pre-computed database embeddings (.npy)",
        )
        db_input.add_argument(
            "--database-fasta", type=Path, help="Path to database FASTA file"
        )

        db_group.add_argument(
            "--database-metadata",
            type=Path,
            help="Path to database metadata file (.parquet or .csv)",
        )

        db_group.add_argument(
            "--database-index", type=Path, help="Path to pre-built FAISS index file"
        )

        # Query arguments
        query_group = parser.add_argument_group("Query Options")
        query_input = query_group.add_mutually_exclusive_group(required=True)
        query_input.add_argument(
            "--query-embeddings",
            type=Path,
            help="Path to pre-computed query embeddings (.npy)",
        )
        query_input.add_argument(
            "--query-fasta", type=Path, help="Path to query FASTA file"
        )

        query_group.add_argument(
            "--query-metadata",
            type=Path,
            help="Path to query metadata file (.parquet or .csv)",
        )

        # Search parameters
        search_group = parser.add_argument_group("Search Parameters")
        search_group.add_argument(
            "--k",
            type=int,
            default=10,
            help="Number of nearest neighbors to find (default: 10)",
        )

        search_group.add_argument(
            "--similarity-threshold",
            type=float,
            help="Minimum similarity threshold for results",
        )

        search_group.add_argument(
            "--exclude-self-matches",
            action="store_true",
            default=None,
            help="Exclude self-matches from results (default: true)",
        )

        search_group.add_argument(
            "--metric",
            choices=["l2", "cosine"],
            default="cosine",
            help="Distance metric to use (default: cosine)",
        )

        search_group.add_argument(
            "--max-results-per-query",
            type=int,
            help="Maximum results to return per query",
        )

        # Embedding parameters
        embed_group = parser.add_argument_group("Embedding Options")
        embed_group.add_argument(
            "--embed-model",
            default="zhihan1996/DNABERT-2-117M",
            help="HuggingFace model for embedding sequences",
        )

        embed_group.add_argument(
            "--batch-size", type=int, help="Batch size for embedding generation"
        )

        embed_group.add_argument(
            "--max-length", type=int, help="Maximum sequence length for embedding"
        )

        # Alignment options
        align_group = parser.add_argument_group("Alignment Options")
        align_group.add_argument(
            "--enable-alignment",
            action="store_true",
            help="Enable sequence alignment for results",
        )

        align_group.add_argument(
            "--alignment-type",
            choices=["local", "global"],
            default="local",
            help="Type of sequence alignment (default: local)",
        )

        # Output options
        output_group = parser.add_argument_group("Output Options")
        output_group.add_argument(
            "--output",
            type=Path,
            default=Path("search_results.json"),
            help="Output file for search results (default: search_results.json)",
        )

        output_group.add_argument(
            "--output-format",
            choices=["csv", "json"],
            default="json",
            help="Output format (default: json)",
        )

        output_group.add_argument(
            "--alignment-output", type=Path, help="Output file for alignment details"
        )

        output_group.add_argument(
            "--consensus-strategy",
            choices=["basic", "majority", "majority_70"],
            default="basic",
            help="Consensus calculation strategy (default: basic)",
        )

        # Performance options
        perf_group = parser.add_argument_group("Performance Options")
        perf_group.add_argument(
            "--index-type",
            choices=["flat", "ivf"],
            default="flat",
            help="Type of search index to build (default: flat)",
        )

        perf_group.add_argument(
            "--save-index", type=Path, help="Save the built index to file for reuse"
        )

    def execute(self, args: argparse.Namespace, settings: Settings) -> int:
        """Execute the search command."""
        start_time = time.time()

        self.logger.info("Starting DNA sequence similarity search")

        try:
            # Initialize repositories and services
            sequence_repo = SequenceRepository()
            embedding_repo = EmbeddingRepository()
            vector_store_repo = VectorStoreRepository()
            result_repo = ResultRepository()
            consensus_repo = ConsensusRepository()

            metadata_service = MetadataService()
            sequence_service = SequenceService(sequence_repo)
            embedding_service = EmbeddingService(embedding_repo)
            consensus_service = ConsensusService(
                consensus_repository=consensus_repo,
                default_strategy=args.consensus_strategy,
            )
            search_service = SearchService(
                vector_store_repo,
                result_repository=result_repo,
                consensus_service=consensus_service,
                metadata_service=metadata_service,
            )

            # Load metadata if provided
            if args.database_metadata:
                search_service.load_metadata(args.database_metadata)

            # Load or generate database embeddings
            if args.database_embeddings:
                db_embeddings = embedding_service.load_precomputed_embeddings(
                    args.database_embeddings,
                )
            else:
                # Load database FASTA and generate embeddings
                db_sequences = sequence_service.load_sequences_from_fasta(
                    args.database_fasta
                )
                db_embeddings = embedding_service.generate_embeddings(
                    db_sequences,
                    args.embed_model,
                    batch_size=args.batch_size or settings.model.batch_size,
                )

            self.logger.info(f"[SUCCESS] Loaded {db_embeddings.shape[0]} database embeddings")

            # Load or generate query embeddings
            if args.query_embeddings:
                query_embeddings = embedding_service.load_precomputed_embeddings(
                    args.query_embeddings,
                )
            else:
                # Load query FASTA and generate embeddings
                query_sequences = sequence_service.load_sequences_from_fasta(
                    args.query_fasta
                )
                query_embeddings = embedding_service.generate_embeddings(
                    query_sequences,
                    args.embed_model,
                    batch_size=args.batch_size or settings.model.batch_size,
                )

            self.logger.info(f"[SUCCESS] Loaded {len(query_embeddings)} query embeddings")

            # Build search index
            self.logger.info("Building search index...")
            _ = search_service.build_search_index(
                db_embeddings, index_type=args.index_type, metric=args.metric
            )

            # Determine exclude_self_matches setting
            if args.exclude_self_matches is not None:
                exclude_self_matches = args.exclude_self_matches
            else:
                exclude_self_matches = settings.search.exclude_self_matches

            # Perform search
            results = search_service.search_similar_sequences(
                query_embeddings,
                db_embeddings,
                k=args.k,
                similarity_threshold=args.similarity_threshold,
                exclude_self_matches=exclude_self_matches,
            )

            # Apply max results per query filter if specified
            if args.max_results_per_query:
                self.logger.info(
                    f"Applying max results per query limit: {args.max_results_per_query}"
                )
                results = search_service.get_top_hits_per_query(
                    results, args.max_results_per_query
                )
            else:
                # Use config default if not specified
                settings = get_settings()
                if (
                    hasattr(settings.search, "max_results_per_query")
                    and settings.search.max_results_per_query
                ):
                    self.logger.info(
                        f"Applying config max results per query limit: {settings.search.max_results_per_query}"
                    )
                    results = search_service.get_top_hits_per_query(
                        results, settings.search.max_results_per_query
                    )

            search_service.calculate_consensus(results)
            # search_service.get_candidate_embeddings()

            # Add alignment if requested
            if args.enable_alignment:
                # This would need the actual sequences, not just embeddings
                # Implementation depends on having access to original sequences
                pass

            # Export results
            self.logger.info(
                f"Saving {results.result_count} results to {args.output}"
            )
            search_service.export_results(
                results,
                args.output,
                format_type=args.output_format,
                include_alignments=args.enable_alignment,
                alignment_output_path=args.alignment_output,
            )

            # Print summary
            elapsed_time = time.time() - start_time
            stats = search_service.calculate_search_statistics(results)
            consensus_stats = search_service.get_concensus_stats()

            print("\n🎯 Search completed successfully!")
            print(f"   • Results: {results.result_count}")
            print(f"   • Queries: {len(query_embeddings)}")
            print(f"   • Database: {len(db_embeddings)} sequences")
            print(f"   • Time: {elapsed_time:.2f} seconds")
            print(f"   • Output: {args.output}")

            if stats.get("mean_similarity_score"):
                print(f"   • Mean similarity: {stats['mean_similarity_score']:.3f}")

            if consensus_stats.get("total_queries", 0) > 0:
                print("\n📊 Consensus Analysis:")
                print(
                    f"   • Total queries analyzed: {consensus_stats['total_queries']}"
                )
                print(
                    f"   • Queries with consensus: {consensus_stats['queries_with_consensus']} ({consensus_stats['consensus_coverage']:.1f}%)"
                )
                print(
                    f"   • Average consensus depth: {consensus_stats['average_consensus_depth']:.1f}"
                )

                # Display level distribution
                print("   • Consensus level distribution:")
                level_dist = consensus_stats["consensus_level_distribution"]
                level_pct = consensus_stats["consensus_level_percentages"]

                # Order levels from broad to specific
                ordered_levels = [
                    "domain",
                    "phylum",
                    "class",
                    "order",
                    "family",
                    "genus",
                    "species",
                    "no_consensus",
                ]

                for level in ordered_levels:
                    count = level_dist.get(level, 0)
                    percentage = level_pct.get(level, 0)
                    if count > 0:
                        level_name = level.replace("_", " ").title()
                        print(
                            f"     - {level_name}: {count} queries ({percentage:.1f}%)"
                        )

            return 0

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return 1


class EmbedCommand(BaseCommand):
    """Command for generating sequence embeddings."""

    @property
    def help(self) -> str:
        return "Generate embeddings for DNA sequences"

    @property
    def description(self) -> str:
        return """
Generate embeddings for DNA sequences using DNABERT-2 or other models.

This command takes FASTA files as input and produces embedding files
that can be used for similarity search.

Examples:
  # Generate embeddings for a FASTA file
  alfreed embed --input sequences.fasta --output embeddings.npy

  # Generate with custom model and parameters
  alfreed embed --input sequences.fasta --output embeddings.npy \\
                --model custom-dnabert --batch-size 64 --max-length 1024
        """

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # Input/Output
        parser.add_argument("--input", type=Path, help="Input FASTA file")

        parser.add_argument("--output", type=Path, help="Output embeddings file (.npy)")

        parser.add_argument(
            "--metadata-output", type=Path, help="Output metadata file (.parquet)"
        )

        # Model parameters
        parser.add_argument(
            "--model",
            default="dnabert2",
            help="Embedding model to use. Options: dnabert2, custom models, or HuggingFace model name",
        )

        parser.add_argument("--batch-size", type=int, help="Batch size for processing")

        parser.add_argument("--max-length", type=int, help="Maximum sequence length")

        parser.add_argument(
            "--device",
            choices=["auto", "cpu", "cuda"],
            help="Device to use for computation",
        )

        # Processing options
        parser.add_argument(
            "--normalize",
            action="store_true",
            default=True,
            help="L2 normalize embeddings (default: true)",
        )

        parser.add_argument(
            "--validate",
            action="store_true",
            help="Validate sequences before embedding",
        )

        parser.add_argument(
            "--convert-iupac",
            choices=["keep", "to-n", "to-standard"],
            default="keep",
            help="How to handle IUPAC ambiguous codes: keep (default), convert to N, or convert to standard bases",
        )

    def execute(self, args: argparse.Namespace, settings: Settings) -> int:
        """Execute the embed command."""

        # Check required arguments for actual embedding generation
        if not args.input:
            print(
                "❌ Error: --input is required"
            )
            return 1

        if not args.output:
            print(
                "❌ Error: --output is required"
            )
            return 1

        self.logger.info(f"Generating embeddings for {args.input}")

        try:
            # Initialize services
            sequence_repo = SequenceRepository()
            embedding_repo = EmbeddingRepository()

            sequence_service = SequenceService(sequence_repo)
            embedding_service = EmbeddingService(embedding_repo)

            # Load sequences
            sequences = sequence_service.load_sequences_from_fasta(args.input)
            self.logger.info(f"[SUCCESS] Loaded {len(sequences)} sequences")

            # Validate if requested
            if args.validate:
                sequences = sequence_service.validate_sequences(sequences)
                self.logger.info(f"[SUCCESS] Validated {len(sequences)} sequences")

            # Handle IUPAC codes if requested
            if args.convert_iupac != "keep":
                self.logger.info(f"Converting IUPAC codes: {args.convert_iupac}")
                converted_sequences = []
                for seq in sequences:
                    if args.convert_iupac == "to-n":
                        converted_seq = seq.to_standard_bases(ambiguous_to="N")
                    else:  # to-standard
                        converted_seq = seq.to_standard_bases(ambiguous_to="standard")
                    converted_sequences.append(converted_seq)
                sequences = converted_sequences
                self.logger.info(
                    f"[SUCCESS] Converted IUPAC codes in {len(sequences)} sequences"
                )

            # Generate embeddings
            embeddings = embedding_service.generate_embeddings(
                sequences,
                args.model,
                batch_size=args.batch_size or settings.model.batch_size,
                max_length=args.max_length or settings.model.max_token_length,
                normalize=args.normalize,
            )

            # Save embeddings
            embedding_service.save_embeddings(
                embeddings,
                args.output,
            )

            print("\n✅ Embeddings generated successfully!")
            print(f"   • Sequences: {len(embeddings)}")
            print(f"   • Dimension: {embeddings[0].embedding_dimension}")
            print(f"   • Model: {args.model}")
            print(f"   • Output: {args.output}")
            if args.metadata_output:
                print(f"   • Metadata: {args.metadata_output}")

            return 0

        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return 1



class IndexCommand(BaseCommand):
    """Command for building and managing search indices."""

    @property
    def help(self) -> str:
        return "Build and manage FAISS search indices"

    @property
    def description(self) -> str:
        return """
Build FAISS search indices from embeddings for faster similarity search.

Pre-building indices can significantly speed up repeated searches
against the same database.

Examples:
  # Build a flat index
  alfreed index --embeddings database.npy --output database.index

  # Build an IVF index for large databases
  alfreed index --embeddings database.npy --output database.index \\
                --type ivf --clusters 1000
        """

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--embeddings",
            type=Path,
            required=True,
            help="Input embeddings file (.npy)",
        )

        parser.add_argument(
            "--output", type=Path, required=True, help="Output index file"
        )

        parser.add_argument(
            "--type",
            choices=["flat", "ivf"],
            default="flat",
            help="Type of index to build (default: flat)",
        )

        parser.add_argument(
            "--metric",
            choices=["cosine", "l2"],
            default="cosine",
            help="Distance metric (default: cosine)",
        )

        parser.add_argument(
            "--clusters",
            type=int,
            default=100,
            help="Number of clusters for IVF index (default: 100)",
        )

    def execute(self, args: argparse.Namespace, settings: Settings) -> int:
        """Execute the index command."""
        self.logger.info(f"Building {args.type} index from {args.embeddings}")

        try:
            # Initialize repositories
            embedding_repo = EmbeddingRepository()
            vector_store_repo = VectorStoreRepository()

            # Load embeddings
            embedding_matrix = embedding_repo.load_embeddings(args.embeddings)
            self.logger.info(f"[SUCCESS] Loaded {embedding_matrix.shape[0]} embeddings")

            # Build index
            if args.type == "flat":
                index = vector_store_repo.create_index(
                    embedding_matrix, metric=args.metric
                )
            else:  # ivf
                index = vector_store_repo.create_ivf_index(
                    embedding_matrix, n_clusters=args.clusters, metric=args.metric
                )

            # Save index
            vector_store_repo.save_index(index, args.output)

            # Print summary
            index_info = vector_store_repo.get_index_info(index)

            print("\n🎯 Index built successfully!")
            print(f"   • Type: {args.type}")
            print(f"   • Vectors: {index_info['total_vectors']}")
            print(f"   • Dimension: {index_info['dimension']}")
            print(f"   • Metric: {args.metric}")
            if args.type == "ivf":
                print(f"   • Clusters: {args.clusters}")
            print(f"   • Output: {args.output}")

            return 0

        except Exception as e:
            self.logger.error(f"Index building failed: {e}")
            return 1
