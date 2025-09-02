#!/bin/bash

# Build script for Alfreed Docker images
# GPU-enabled build only

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display usage
show_usage() {
    echo -e "${BLUE}🧬 Alfreed Docker Build Script (GPU-Only)${NC}"
    echo "=========================================="
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "OPTIONS:"
    echo "  -h, --help     Show this help message"
    echo "  -t, --tag      Tag for the image (default: latest)"
    echo "  --no-cache     Build without cache"
    echo "  --push         Push image to registry after build"
    echo
    echo "Examples:"
    echo "  $0                        # Build GPU variant"
    echo "  $0 -t v1.0.0             # Build with specific tag"
    echo "  $0 --no-cache            # Build without cache"
}

# Function to check Docker availability
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed or not in PATH${NC}"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo -e "${RED}❌ Docker is not running${NC}"
        exit 1
    fi
}

# Function to build GPU variant
build_gpu() {
    local tag=${1:-latest}
    local cache_option=${2:-}
    
    echo -e "${YELLOW}🔨 Building Alfreed GPU image...${NC}"
    
    cd "$(dirname "$0")/.."
    
    docker build \
        ${cache_option} \
        -f docker/Dockerfile \
        -t "alfreed:${tag}" \
        .
    
    echo -e "${GREEN}✅ GPU image built successfully: alfreed:${tag}${NC}"
}

# Function to push image
push_image() {
    local tag=${1:-latest}
    
    echo -e "${YELLOW}📤 Pushing image to registry...${NC}"
    
    docker push "alfreed:${tag}"
    
    echo -e "${GREEN}✅ Image pushed successfully${NC}"
}

# Main execution
main() {
    # Default values
    tag="latest"
    no_cache=""
    push=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -t|--tag)
                tag="$2"
                shift 2
                ;;
            --no-cache)
                no_cache="--no-cache"
                shift
                ;;
            --push)
                push=true
                shift
                ;;
            *)
                echo -e "${RED}❌ Unknown option: $1${NC}"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Check Docker
    check_docker
    
    echo -e "${BLUE}🧬 Alfreed Docker Build (GPU)${NC}"
    echo "============================="
    echo "Tag: $tag"
    echo "No cache: ${no_cache:-false}"
    echo "Push: $push"
    echo
    
    # Build GPU variant
    build_gpu "$tag" "$no_cache"
    
    # Push if requested
    if [[ "$push" == true ]]; then
        push_image "$tag"
    fi
    
    echo
    echo -e "${GREEN}🎉 Build complete!${NC}"
    
    # Show built image
    echo
    echo "Built image:"
    docker images | grep "alfreed" | grep "$tag"
}

# Run main function with all arguments
main "$@"
