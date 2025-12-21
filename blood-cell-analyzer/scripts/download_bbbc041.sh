#!/bin/bash
# Download BBBC041 Malaria Dataset from Broad Institute
# Dataset: 1,364 images, ~80,000 cells with bounding box annotations

set -e  # Exit on error

DATASET_URL="https://data.broadinstitute.org/bbbc/BBBC041/malaria.zip"
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
TARGET_DIR="$DATA_DIR/bbbc041_malaria"
ZIP_FILE="$DATA_DIR/malaria.zip"

echo "=========================================="
echo "BBBC041 Malaria Dataset Download"
echo "=========================================="
echo ""
echo "Target directory: $TARGET_DIR"
echo "Dataset URL: $DATASET_URL"
echo "Dataset size: ~2.26 GB"
echo ""

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Check if dataset already exists
if [ -d "$TARGET_DIR" ] && [ "$(ls -A $TARGET_DIR)" ]; then
    echo "⚠️  Dataset directory already exists and is not empty: $TARGET_DIR"
    read -p "Do you want to re-download and overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        exit 0
    fi
    echo "Removing existing directory..."
    rm -rf "$TARGET_DIR"
fi

# Download dataset
echo "📥 Downloading BBBC041 dataset (2.26 GB)..."
echo "This may take several minutes depending on your connection..."
if command -v wget &> /dev/null; then
    wget --progress=bar:force "$DATASET_URL" -O "$ZIP_FILE"
elif command -v curl &> /dev/null; then
    curl -L --progress-bar "$DATASET_URL" -o "$ZIP_FILE"
else
    echo "❌ Error: Neither wget nor curl found. Please install one of them."
    exit 1
fi

# Verify download
if [ ! -f "$ZIP_FILE" ]; then
    echo "❌ Error: Download failed. File not found: $ZIP_FILE"
    exit 1
fi

FILE_SIZE=$(stat -f%z "$ZIP_FILE" 2>/dev/null || stat -c%s "$ZIP_FILE" 2>/dev/null)
if [ $FILE_SIZE -lt 1000000000 ]; then  # Less than 1GB suggests incomplete download
    echo "⚠️  Warning: Downloaded file seems small ($FILE_SIZE bytes). Expected ~2.26 GB."
    read -p "Continue with extraction anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        rm "$ZIP_FILE"
        echo "Download cancelled. Please try again."
        exit 1
    fi
fi

echo "✅ Download complete!"
echo ""

# Extract dataset
echo "📦 Extracting dataset..."
mkdir -p "$TARGET_DIR"
unzip -q "$ZIP_FILE" -d "$TARGET_DIR"

# Clean up zip file
echo "🗑️  Cleaning up..."
rm "$ZIP_FILE"

# Verify extraction
echo ""
echo "🔍 Verifying dataset structure..."

# Count images in subdirectories
TOTAL_IMAGES=0
for subdir in "$TARGET_DIR"/*; do
    if [ -d "$subdir" ]; then
        COUNT=$(find "$subdir" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
        SUBDIR_NAME=$(basename "$subdir")
        echo "  - $SUBDIR_NAME: $COUNT images"
        TOTAL_IMAGES=$((TOTAL_IMAGES + COUNT))
    fi
done

echo ""
echo "Total images found: $TOTAL_IMAGES"

if [ $TOTAL_IMAGES -ge 1300 ] && [ $TOTAL_IMAGES -le 1400 ]; then
    echo "✅ Dataset verification passed! Expected ~1,364 images."
else
    echo "⚠️  Warning: Expected ~1,364 images, found $TOTAL_IMAGES"
fi

# Check for annotation files
ANNOTATION_FILES=$(find "$TARGET_DIR" -type f \( -name "*.json" -o -name "*.csv" -o -name "*.txt" \) | wc -l)
echo "Annotation files found: $ANNOTATION_FILES"

echo ""
echo "=========================================="
echo "✅ BBBC041 Dataset Downloaded Successfully"
echo "=========================================="
echo ""
echo "Dataset location: $TARGET_DIR"
echo ""
echo "Next steps:"
echo "  1. Run: python scripts/parse_bbbc041_annotations.py"
echo "  2. Run: python scripts/prepare_bbbc041_dataset.py"
echo ""
echo "Dataset information:"
echo "  - Source: Broad Bioimage Benchmark Collection"
echo "  - Images: ~1,364 blood smear microscopy images"
echo "  - Cells: ~80,000 annotated cells"
echo "  - Classes: 6 (uninfected RBCs, leukocytes, ring, trophozoite, schizont, gametocyte)"
echo "  - License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0"
echo "  - Citation: Ljosa et al., Nature Methods, 2012"
echo ""
