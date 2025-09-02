#!/bin/bash

# Download script for ALFREED data files from Google Drive
# Usage: ./download.sh

set -e  # Exit on any error

# Data directory
DATA_DIR="$(dirname "$0")"
cd "$DATA_DIR"

echo "Downloading data files from Google Drive..."

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Error: gdown is not installed. Installing it now..."
    pip install gdown
    if [ $? -ne 0 ]; then
        echo "Failed to install gdown. Please install it manually with: pip install gdown"
        exit 1
    fi
fi

# Google Drive file IDs (replace these with your actual file IDs)
# To get the file ID from a Google Drive sharing link:
# https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing
declare -A FILES=(
    ["embeddings.npy"]="https://drive.google.com/file/d/1rkxPOiRvNGc1oR4aLJ-XRd6dDKA_KXPS/view?usp=sharing"
    ["gg2_full_1400_1600.parquet"]="https://drive.google.com/file/d/1lLbS6ZvZYZ7zn-PSgJILNjUc5bpA6-0L/view?usp=sharing"
    ["query_sequences_100.fasta"]="https://drive.google.com/file/d/1szOTuf62weRlybO8hM5fq6OvWtnzj5uu/view?usp=sharing"
    ["query_sequences_1000.fasta"]="https://drive.google.com/file/d/1l8rlrTjLPzWVwTwrFnIfMb4XlviCL4cb/view?usp=sharing"
)

# Download each file
download_count=0
for filename in "${!FILES[@]}"; do
    file_id="${FILES[$filename]}"
    echo "Downloading $filename..."
    
    # Check if file already exists
    if [ -f "$filename" ]; then
        echo "File $filename already exists. Skipping download."
        echo "Delete the file if you want to re-download it."
        continue
    fi
    
    # Download from Google Drive
    if gdown --id "$file_id" --output "$filename"; then
        echo "✓ Successfully downloaded $filename"
        ((download_count++))
    else
        echo "✗ Failed to download $filename"
        echo "Please check that:"
        echo "  - The file ID is correct"
        echo "  - The file is publicly accessible or you have permission"
        echo "  - Your internet connection is stable"
        exit 1
    fi
    
    echo ""
done

echo "Download completed!"
echo "Files downloaded: $download_count"
echo ""
echo "Downloaded files:"
for filename in "${!FILES[@]}"; do
    if [ -f "$filename" ]; then
        file_size=$(du -sh "$filename" | cut -f1)
        echo "  ✓ $filename ($file_size)"
    fi
done

echo ""
echo "All data files are ready for use!"