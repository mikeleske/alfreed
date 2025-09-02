# PowerShell script to build Alfreed Docker images
# GPU-enabled build only

param(
    [Parameter()]
    [string]$Tag = "latest",
    
    [Parameter()]
    [switch]$NoCache,
    
    [Parameter()]
    [switch]$Push,
    
    [Parameter()]
    [switch]$Help
)

function Write-ColorText {
    param(
        [string]$Text,
        [ConsoleColor]$Color = "White"
    )
    Write-Host $Text -ForegroundColor $Color
}

function Show-Usage {
    Write-ColorText "🧬 Alfreed Docker Build Script (GPU-Only)" Blue
    Write-Host "==========================================="
    Write-Host ""
    Write-Host "Usage: .\build-docker.ps1 [OPTIONS]"
    Write-Host ""
    Write-ColorText "OPTIONS:" Green
    Write-Host "  -Tag       Tag for the image (default: latest)"
    Write-Host "  -NoCache   Build without cache"
    Write-Host "  -Push      Push image to registry after build"
    Write-Host "  -Help      Show this help message"
    Write-Host ""
    Write-ColorText "Examples:" Cyan
    Write-Host "  .\build-docker.ps1                       # Build GPU image"
    Write-Host "  .\build-docker.ps1 -Tag v1.0.0          # Build with specific tag"
    Write-Host "  .\build-docker.ps1 -NoCache             # Build without cache"
}

function Test-DockerAvailable {
    try {
        docker info | Out-Null
        return $true
    }
    catch {
        Write-ColorText "❌ Docker is not running or not available" Red
        return $false
    }
}

function Build-GpuImage {
    param(
        [string]$ImageTag,
        [bool]$UseNoCache
    )
    
    Write-ColorText "🔨 Building Alfreed GPU image..." Yellow
    
    # Change to project root
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $projectRoot = Split-Path -Parent $scriptDir
    Push-Location $projectRoot
    
    try {
        $buildArgs = @(
            "build"
            "-f", "docker/Dockerfile"
            "-t", "alfreed:$ImageTag"
        )
        
        if ($UseNoCache) {
            $buildArgs += "--no-cache"
        }
        
        $buildArgs += "."
        
        & docker @buildArgs
        
        if ($LASTEXITCODE -ne 0) {
            throw "Docker build failed"
        }
        
        Write-ColorText "✅ GPU image built successfully: alfreed:$ImageTag" Green
    }
    finally {
        Pop-Location
    }
}

function Push-Image {
    param(
        [string]$ImageTag
    )
    
    Write-ColorText "📤 Pushing image to registry..." Yellow
    
    docker push "alfreed:$ImageTag"
    
    Write-ColorText "✅ Image pushed successfully" Green
}

# Main execution
try {
    if ($Help) {
        Show-Usage
        exit 0
    }
    
    # Check Docker
    if (-not (Test-DockerAvailable)) {
        exit 1
    }
    
    Write-ColorText "🧬 Alfreed Docker Build (GPU)" Blue
    Write-Host "=============================="
    Write-Host "Tag: $Tag"
    Write-Host "No cache: $NoCache"
    Write-Host "Push: $Push"
    Write-Host ""
    
    # Build GPU image
    Build-GpuImage $Tag $NoCache.IsPresent
    
    # Push if requested
    if ($Push) {
        Push-Image $Tag
    }
    
    Write-Host ""
    Write-ColorText "🎉 Build complete!" Green
    
    # Show built image
    Write-Host ""
    Write-Host "Built image:"
    docker images | Select-String "alfreed" | Select-String $Tag
}
catch {
    Write-ColorText "❌ Error: $_" Red
    exit 1
}
