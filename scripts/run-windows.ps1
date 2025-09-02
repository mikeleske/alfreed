# PowerShell script to run Alfreed with Docker on Windows

param(
    [Parameter(Position=0)]
    [string]$Command,
    
    [Parameter(Position=1, ValueFromRemainingArguments=$true)]
    [string[]]$RemainingArgs,
    
    [Parameter()]
    [ValidateSet("cpu", "gpu")]
    [string]$Variant = "cpu"
)

function Write-Banner {
    Write-Host "🧬 Alfreed - Windows Docker Runner" -ForegroundColor Blue
    Write-Host "================================" -ForegroundColor Blue
    Write-Host ""
}

function Test-DockerRunning {
    try {
        docker info | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Show-Usage {
    Write-Host "Usage: .\run-windows.ps1 [command] [options] [-Variant cpu|gpu]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Green
    Write-Host "  build-cpu - Build CPU-only Docker image"
    Write-Host "  build-gpu - Build GPU Docker image"  
    Write-Host "  build-all - Build both CPU and GPU images"
    Write-Host "  run       - Run Alfreed with specified variant (default: cpu)"
    Write-Host "  search    - Run search command"
    Write-Host "  embed     - Run embed command"
    Write-Host "  index     - Run index command"
    Write-Host "  shell     - Start interactive shell in container"
    Write-Host "  help      - Show help"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Green
    Write-Host "  -Variant  - Choose cpu or gpu variant (default: cpu)"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\run-windows.ps1 build-cpu"
    Write-Host "  .\run-windows.ps1 build-gpu"
    Write-Host "  .\run-windows.ps1 run -Variant cpu"
    Write-Host "  .\run-windows.ps1 search -Variant gpu --database-embeddings data\embeddings.npy --query-fasta data\queries.fasta --k 10"
    Write-Host "  .\run-windows.ps1 embed --input data\sequences.fasta --output embeddings.npy"
}

Write-Banner

# Check if Docker is running
if (-not (Test-DockerRunning)) {
    Write-Host "❌ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Change to project root directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

try {
    switch ($Command) {
        "" { 
            Show-Usage
            break
        }
        
        "build" {
            Write-Host "🔨 Building Alfreed GPU Docker image..." -ForegroundColor Yellow
            Write-Host "⚠️ Note: GPU support on Windows requires WSL2 and Docker Desktop with GPU support" -ForegroundColor Yellow
            docker-compose -f docker/docker-compose.windows.yml build alfreed
            break
        }
        
        "run" {
            Write-Host "🚀 Starting Alfreed..." -ForegroundColor Green
            docker-compose -f docker/docker-compose.windows.yml up -d
            docker-compose -f docker/docker-compose.windows.yml logs -f
            break
        }
        
        "search" {
            Write-Host "🔍 Running search command..." -ForegroundColor Cyan
            $AllArgs = @("search") + $RemainingArgs
            docker-compose -f docker/docker-compose.windows.yml run --rm alfreed @AllArgs
            break
        }
        
        "embed" {
            Write-Host "🧬 Running embed command..." -ForegroundColor Cyan
            $AllArgs = @("embed") + $RemainingArgs
            docker-compose -f docker/docker-compose.windows.yml run --rm alfreed @AllArgs
            break
        }
        
        "index" {
            Write-Host "🏗️ Running index command..." -ForegroundColor Cyan
            $AllArgs = @("index") + $RemainingArgs
            docker-compose -f docker/docker-compose.windows.yml run --rm alfreed @AllArgs
            break
        }
        
        "shell" {
            Write-Host "🐚 Starting interactive shell..." -ForegroundColor Magenta
            docker-compose -f docker/docker-compose.windows.yml run --rm alfreed bash
            break
        }
        
        "help" {
            Write-Host "📖 Showing Alfreed help..." -ForegroundColor Blue
            docker-compose -f docker/docker-compose.windows.yml run --rm alfreed --help
            break
        }
        
        default {
            Write-Host "🧬 Running Alfreed command: $Command $($RemainingArgs -join ' ')" -ForegroundColor Green
            $AllArgs = @($Command) + $RemainingArgs
            docker-compose -f docker/docker-compose.windows.yml run --rm alfreed @AllArgs
            break
        }
    }
}
catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
    exit 1
}
finally {
    # Return to original directory
    Set-Location $ScriptDir
}
