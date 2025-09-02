@echo off
REM Windows batch script to run Alfreed with Docker

echo 🧬 Alfreed - Windows Docker Runner
echo ================================

REM Check if Docker is running
docker info >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Docker is not running. Please start Docker Desktop.
    exit /b 1
)

REM Get current directory
set CURRENT_DIR=%CD%
cd /d "%~dp0.."

REM Check command type
if "%1"=="" (
    echo Usage: run-windows.bat [command] [options]
    echo.
    echo Commands:
    echo   build    - Build the Docker image
    echo   prod     - Run in production mode
    echo   search   - Run search command
    echo   embed    - Run embed command
    echo   index    - Run index command
    echo   help     - Show help
    echo.
    exit /b 0
)

if "%1"=="build" (
    echo 🔨 Building Alfreed Docker image...
    docker-compose -f docker/docker-compose.windows.yml build
    goto end
)


if "%1"=="prod" (
    echo 🚀 Starting Alfreed in production mode...
    docker-compose -f docker/docker-compose.windows.yml --profile prod up
    goto end
)

if "%1"=="search" (
    echo 🔍 Running search command...
    shift
    docker-compose -f docker/docker-compose.windows.yml --profile run run --rm alfreed-run search %*
    goto end
)

if "%1"=="embed" (
    echo 🧬 Running embed command...
    shift
    docker-compose -f docker/docker-compose.windows.yml --profile run run --rm alfreed-run embed %*
    goto end
)

if "%1"=="index" (
    echo 🏗️ Running index command...
    shift
    docker-compose -f docker/docker-compose.windows.yml --profile run run --rm alfreed-run index %*
    goto end
)

if "%1"=="help" (
    echo 📖 Showing Alfreed help...
    docker-compose -f docker/docker-compose.windows.yml --profile run run --rm alfreed-run --help
    goto end
)

REM Default: pass all arguments to alfreed
echo 🧬 Running Alfreed command: %*
docker-compose -f docker/docker-compose.windows.yml --profile run run --rm alfreed-run %*

:end
cd /d "%CURRENT_DIR%"
