# 🧬 Alfreed v0.2 - DNA Sequence Similarity Search

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker/)

A high-performance DNA sequence similarity search system using Genomic Language Model (gLM) embeddings and FAISS vector search.

This implementation follows the principles defined in our paper:

**Alignment-free Bacterial Taxonomy Classification with Genomic Language Models**
*Mike Leske, Jamie A. FitzGerald, Keith Coughlan, Francesca Bottacini, Haithem Afli, Bruno Gabriel Nascimento Andrade*
bioRxiv 2025.06.27.662019; doi: https://doi.org/10.1101/2025.06.27.662019
[![Link](https://www.biorxiv.org/content/10.1101/2025.06.27.662019v1)](Paper)

## 🏛️ Architecture

Alfreed follows clean architecture principles with clear separation of concerns:

```
┌─────────────────────────────────────────┐
│             Interface Layer             │
│                  (CLI)                  │
├─────────────────────────────────────────┤
│             Service Layer               │
│      (Business Logic Orchestration)     │
├─────────────────────────────────────────┤
│           Repository Layer              │
│         (Data Access Abstraction)       │
├─────────────────────────────────────────┤
│              Core Layer                 │
│     (Domain Entities & Algorithms)      │
├─────────────────────────────────────────┤
│          Infrastructure Layer           │
│    (Configuration, Models, Storage)     │
└─────────────────────────────────────────┘
```

### 🎯 Layer Responsibilities

- **Core Layer**: Business entities (Sequence, Embedding, SearchResult) and algorithms
- **Repository Layer**: Data access for sequences, embeddings, metadata, and vector stores
- **Service Layer**: Workflow orchestration and use case implementation
- **Interface Layer**: CLI commands
- **Infrastructure Layer**: Configuration management, logging, and external dependencies

## 🚀 Quick Start

### Local Installation

```bash
# Clone and install
git clone <repository-url>
cd alfreed
pip install -e .
```

### Download starter data

The provided script will load required files and precomputed embeddings for the Greengenes2 2024.09 Full 16S dataset.

```bash
cd data
sh download.sh
```

### Using Docker

> 📖 **For comprehensive Docker setup instructions, see [DOCKER_USAGE.md](DOCKER_USAGE.md)**

```bash
# Clone the repository
git clone <repository-url>
cd alfreed

# Quick Build Commands (build only)
docker build -f docker/Dockerfile -t alfreed:latest .
```

## 📖 Usage

### CLI Commands

#### 1. Search Similar Sequences

```bash
# Search with pre-computed embeddings
alfreed search \
  --database-embeddings database.npy \
  --database-metadata database.parquet \
  --query-fasta queries.fasta \
  --k 10 \
  --output results.json

# Full FASTA-to-FASTA search
alfreed search \
  --database-fasta database.fasta \
  --query-fasta queries.fasta \
  --k 10 \
  --embed-model zhihan1996/DNABERT-2-117M \
  --enable-alignment \
  --output results.json
```

#### 2. Generate Embeddings

```bash
# Generate embeddings for sequences
alfreed embed \
  --input sequences.fasta \
  --output embeddings.npy \
  --metadata-output metadata.parquet \
  --model zhihan1996/DNABERT-2-117M
```

#### 3. Build Search Index

```bash
# Build FAISS index for faster searches
alfreed index \
  --embeddings database.npy \
  --output database.index \
  --type flat \
  --metric cosine
```

### Configuration

Create configuration files in the `config/` directory:

```yaml
# config/production.yaml
model:
  default_embedding_model: "zhihan1996/DNABERT-2-117M"
  batch_size: 64
  device: "auto"

search:
  default_k: 10
  similarity_threshold: 0.8
  enable_alignment: true

storage:
  data_dir: "/app/data"
  output_dir: "/app/output"
```

Or use environment variables:

```bash
export ALFREED_ENV=production
export ALFREED_DEVICE=cuda
export ALFREED_BATCH_SIZE=8
```

### Simple Docker Build Commands

#### Standard Build (Recommended)
```bash
# Universal build command (works on all platforms)
docker build -t alfreed:latest -f docker/Dockerfile .

# Clean build (no cache)
docker build --no-cache -t alfreed:latest -f docker/Dockerfile .
```

### Direct Container Usage

```bash
# Linux/macOS/Windows Git Bash
docker run --rm -it \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/output:/app/output \
  alfreed:latest search --help

docker run --gpus all --rm -it \
  -v ${PWD}/data:/app/data:ro \
  -v ${PWD}/output:/app/output \
  alfreed:latest search \
  --database-embeddings /app/data/embeddings.npy \
  --database-metadata /app/data/gg2_full_1400_1600.parquet \
  --query-fasta /app/data/query_sequences_1000.fasta \
  --output /app/output/results.json \
  --max-results-per-query 5 \
  --k 10 \
  --embed-model zhihan1996/DNABERT-2-117M
```

## 🔧 Code Quality

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing framework

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Run tests
pytest tests/
```

### Project Structure

```
src/alfreed/
├── core/                   # Domain layer
│   ├── entities/          # Business entities
│   ├── algorithms/        # Core algorithms
│   └── interfaces/        # Abstract interfaces
├── repositories/          # Data access layer
├── services/              # Business logic layer
├── infrastructure/        # External dependencies
│   ├── config/           # Configuration management
│   ├── models/           # ML model clients
│   └── storage/          # Storage abstractions
└── interfaces/            # API layer
    ├── cli/              # Command-line interface
    └── web/              # Future web interface
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
