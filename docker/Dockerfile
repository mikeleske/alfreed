# GPU-enabled Docker build for Alfreed DNA Sequence Search
FROM python:3.12-slim

# Build environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
        build-essential \
        git \
        curl \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Setup pip (Python 3.12 already available in base image)
RUN python -m pip install --upgrade pip setuptools wheel

# Create application user and directories  
RUN groupadd -r alfreed && useradd -r -g alfreed -d /app -s /bin/bash alfreed \
    && mkdir -p /app/{data,output,cache,logs} /tmp/alfreed

WORKDIR /app

# Install PyTorch with CUDA 12.4 support
RUN python -m pip install --no-cache-dir \
    torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124 \
    && python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed')"

# Install other ML/CUDA dependencies 
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y triton || true

# Copy application code and install core package
COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml LICENSE README.md ./

# Install application (only core deps, ML deps already installed above)
RUN pip install -e . \
    && python -c "import alfreed; print('Alfreed installed successfully')" \
    && chown -R alfreed:alfreed /app /tmp/alfreed

# Copy and setup entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown alfreed:alfreed /entrypoint.sh

# Switch to non-root user
USER alfreed

# Runtime environment variables
ENV ALFREED_ENV=production \
    ALFREED_DEVICE=cuda \
    HOME=/app \
    CUDA_VISIBLE_DEVICES=0

# GPU health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"

# Container startup
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]