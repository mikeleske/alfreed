#!/bin/bash
set -e

# Set HuggingFace suppression variables IMMEDIATELY
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_VERBOSITY=error
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS=ignore
export HF_HUB_VERBOSITY=error
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Banner
echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║    █████╗ ██║     ███████╗██████╗ ███████╗███████╗██████╗  ║"
echo "║   ██╔══██╗██║     ██╔════╝██╔══██╗██╔════╝██╔════╝██╔══██╗ ║"
echo "║   ███████║██║     █████╗  ██████╔╝█████╗  █████╗  ██║  ██║ ║"
echo "║   ██╔══██║██║     ██╔══╝  ██╔══██╗██╔══╝  ██╔══╝  ██║  ██║ ║"
echo "║   ██║  ██║███████╗██║     ██║  ██║███████╗███████╗██████╔╝ ║"
echo "║   ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝╚═════╝  ║"
echo "║                                                            ║"
echo "║    🧬 Alignent-free DNA Sequence Similarity Search v0.2.0  ║"
echo "║    🚀 CUDA-Accelerated                                     ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Environment info
log "Environment: ${ALFREED_ENV:-production}"
log "Device: ${ALFREED_DEVICE:-cuda}"

# GPU check (non-blocking)
if command -v nvidia-smi &> /dev/null; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
        log "✅ CUDA available with $GPU_COUNT GPU(s)"
    else
        warn "CUDA not available - running in CPU mode"
        export ALFREED_DEVICE=cpu
    fi
else
    warn "No GPU runtime detected - running in CPU mode"
    export ALFREED_DEVICE=cpu
fi

# Package check with fallback
if python -c "import alfreed; print(f'✅ Alfreed {alfreed.__version__} ready')" 2>/dev/null; then
    : # Package works
elif python -c "import sys; sys.path.insert(0, '/app/src'); import alfreed; print(f'Alfreed {alfreed.__version__} ready')" 2>/dev/null; then
    export PYTHONPATH="/app/src:$PYTHONPATH"
else
    error "Alfreed package not found"
    exit 1
fi

# Configuration check
python -c "
import sys, os
pythonpath = os.environ.get('PYTHONPATH', '')
if pythonpath:
    for path in pythonpath.split(':'):
        if path and path not in sys.path:
            sys.path.insert(0, path)
from alfreed.infrastructure.config.settings import get_settings
get_settings()
print('✅ Configuration valid')
" || (error "Configuration failed" && exit 1)

log "🚀 Ready for DNA sequence processing"

# Command handling
if [ $# -eq 0 ]; then
    PYTHONWARNINGS=ignore python -m alfreed.interfaces.cli.main --help
elif [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
    exec /bin/bash
elif [ "$1" = "python" ]; then
    exec "$@"
else
    PYTHONWARNINGS=ignore python -m alfreed.interfaces.cli.main "$@"
fi