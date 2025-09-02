# Alfreed Docker Setup Guide

This guide explains how to use the Docker setup for Alfreed, which is GPU-optimized for maximum performance.

## Overview

Alfreed now provides a single GPU-accelerated Docker variant:

- **`alfreed`**: GPU-accelerated variant optimized for performance (requires NVIDIA GPU and Docker GPU support)

## Quick Start

### Building Images

#### Using Build Scripts (Recommended)

**Linux/macOS:**
```bash
# Build GPU image
./scripts/build-docker.sh

# Build with custom tag
./scripts/build-docker.sh -t v1.0.0

# Build without cache
./scripts/build-docker.sh --no-cache
```

**Windows (PowerShell):**
```powershell
# Build GPU image
.\scripts\build-docker.ps1

# Build with custom tag
.\scripts\build-docker.ps1 -Tag v1.0.0

# Build without cache
.\scripts\build-docker.ps1 -NoCache
```

#### Using Docker Compose

**Build GPU image:**
```bash
docker-compose -f docker/docker-compose.yml build alfreed
```

or simply:

```bash
docker-compose -f docker/docker-compose.yml build
```

#### Direct Docker Build

**Build GPU image:**
```bash
docker build -f docker/Dockerfile -t alfreed:latest .
```

### Running Alfreed

#### Using Docker Compose

**Run Alfreed:**
```bash
docker-compose -f docker/docker-compose.yml up
```

#### Using Windows Scripts

**Windows PowerShell:**
```powershell
# Run Alfreed
.\scripts\run-windows.ps1 run

# Run specific commands
.\scripts\run-windows.ps1 search --database-embeddings data\embeddings.npy --query-fasta data\queries.fasta --k 10
```

#### Direct Container Execution

```bash
docker run --rm -it --gpus all \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/output:/app/output \
  alfreed:latest --help
```

## System Requirements

- NVIDIA GPU with CUDA Compute Capability 6.0+
- NVIDIA Docker runtime installed
- Docker Engine 20.10+ with GPU support
- 8GB+ GPU memory recommended
- Docker Engine 20.10+ or Docker Desktop
- 4GB+ RAM recommended
- Linux, macOS, or Windows with WSL2

#### Installing GPU Support

**Linux:**
```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Windows:**
- Install Docker Desktop with WSL2 backend
- Install NVIDIA drivers for WSL2
- Enable GPU support in Docker Desktop settings

## Configuration Files

### Docker Compose Files

- **`docker/docker-compose.yml`**: Main configuration for Linux/macOS
- **`docker/docker-compose.windows.yml`**: Windows-optimized configuration
- **`docker/docker-compose.override.yml`**: Local development overrides

### Dockerfile

The main `docker/Dockerfile` is optimized for GPU performance and contains:

- System dependencies for CUDA and Python
- GPU-optimized dependencies (PyTorch CUDA)
- Production-ready environment setup

### Requirements Files

- **`requirements.txt`**: GPU-optimized dependencies (PyTorch CUDA, FAISS-GPU)  
- **`requirements-dev.txt`**: Development dependencies (testing, linting, documentation)

## Environment Variables

### Common Variables

| Variable | Description | Default |
|----------|-------------|---------|  
| `ALFREED_ENV` | Environment (production) | `production` |
| `ALFREED_DEVICE` | Device type (always cuda) | `cuda` |
| `ALFREED_LOG_LEVEL` | Logging level | `INFO` |
| `PYTHONPATH` | Python path | `/app/src` |

### GPU-Specific Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU devices to use | `0` |
| `HUGGINGFACE_HUB_CACHE` | HuggingFace model cache | `/app/cache/huggingface` |
| `TRANSFORMERS_CACHE` | Transformers cache | `/app/cache/transformers` |

## Volume Mounts

### Standard Mounts

- **`/app/data`**: Input data (read-only)
- **`/app/output`**: Output files (read-write)  
- **`/app/cache`**: Model and embedding cache
- **`/app/logs`**: Application logs

### Windows Considerations

Windows uses explicit bind mounts for better performance:

```yaml
volumes:
  - type: bind
    source: ../data
    target: /app/data
    read_only: true
```

## Troubleshooting

### Common Issues

**Docker build fails:**
- Check that you're building from the project root directory
- Ensure you have sufficient disk space for CUDA base image

**GPU not detected:**
- Verify NVIDIA Docker runtime is installed
- Check `docker run --rm --gpus all nvidia/cuda:12.6.1-runtime-ubuntu22.04 nvidia-smi`
- Ensure Docker daemon has GPU access

**Permission errors on Linux:**
- The container runs as user `alfreed` (UID 1000)
- Ensure your data directory is readable by UID 1000
- Use `chown -R 1000:1000 data output` if needed

**Windows-specific issues:**
- Ensure WSL2 backend is enabled in Docker Desktop
- Check that path separators use forward slashes in Docker commands
- For GPU support, verify WSL2 has NVIDIA driver support

### Health Checks

The container includes health checks:

- Validates CUDA availability
- Ensures GPU can be accessed by PyTorch

### Logs and Debugging

**View logs:**
```bash
docker-compose -f docker/docker-compose.yml logs -f
```

**Debug container:**
```bash
docker-compose -f docker/docker-compose.yml run --rm alfreed bash
```

**Check GPU in container:**
```bash
docker-compose -f docker/docker-compose.yml run --rm alfreed python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Performance Considerations

### GPU Performance
- Significantly faster for large datasets
- Requires NVIDIA GPU with CUDA support
- Higher memory requirements (both system and GPU)
- Optimized for production workloads

### Memory Usage

| Component | GPU Requirements |
|-----------|------------------|
| Base Memory | 4-6GB system RAM |
| Model Loading | +2-4GB system RAM |
| Processing | GPU memory dependent (4-8GB+ GPU RAM) |

## Examples

### Basic Search Operation

```bash
docker-compose -f docker/docker-compose.yml run --rm alfreed \
  search \
  --database-embeddings /app/data/embeddings.npy \
  --query-fasta /app/data/queries.fasta \
  --output /app/output/results.json \
  --k 10
```

### Embedding Generation

```bash
docker-compose -f docker/docker-compose.yml run --rm alfreed \
  embed \
  --input /app/data/sequences.fasta \
  --output /app/output/embeddings.npy \
  --model-name dnabert2 \
  --batch-size 32
```
