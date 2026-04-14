#!/bin/bash
set -e

echo "===== RUN START ====="

# Install project
echo "Installing project..."
pip install -e .

# Ensure dataset exists
if [ ! -f "FLAIR-HUB_FULL.zip" ]; then
    echo "ERROR: FLAIR-HUB_FULL.zip not found!"
    exit 1
fi

# Extract dataset
echo "Extracting dataset..."
mkdir -p csv
unzip -o FLAIR-HUB_FULL.zip -d csv/

# Fix CSV only once
if [ ! -f "csv/.processed" ]; then
    echo "Processing CSV files..."
    sed -i 's/;/,/g' csv/FLAIR-HUB_*.csv
    touch csv/.processed
fi

# Download HuggingFace data
echo "Downloading HuggingFace data..."
python flair-hub-HF-dl.py

# Unzip downloaded files
echo "Unzipping downloaded files..."
for f in FLAIR-HUB_download/data/*.zip; do
    unzip -o "$f" -d FLAIR-HUB_download/data/ && rm "$f";
done

DATA_PATH=$(realpath --relative-to="$(pwd)" FLAIR-HUB_download/data)
sed -i "s|\.\./|${DATA_PATH}/|g" csv/FLAIR-HUB_*.csv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# TRAINING
echo "===== TRAINING START ====="

# CLI ARCHITECTURE SELECTION:
# --use_LPR_decoder  : Enable Local Patch Refiner decoder (default: disabled)
# --no_LPR_decoder   : Disable Local Patch Refiner decoder (use UNet)
# --use_ViT          : Enable Vision Transformer encoder (default: enabled)
# --no_ViT           : Disable Vision Transformer encoder (use Swin)
#
# ARCHITECTURE COMMANDS:
# [1] Swin + UNet   : flairhub --config configs/train/ --name NAME --no_LPR_decoder --no_ViT
# [2] Swin + LPR    : flairhub --config configs/train/ --name NAME --no_ViT
# [3] ViT + UNet    : flairhub --config configs/train/ --name NAME --no_LPR_decoder
# [4] ViT + LPR     : flairhub --config configs/train/ --name NAME

echo "Running Swin + UNet..."
flairhub --config configs/train/ --name SwinUNet --no_LPR_decoder --no_ViT

echo "Running Swin + LPR..."
flairhub --config configs/train/ --name SwinLPR --no_ViT

echo "Running ViT + UNet..."
flairhub --config configs/train/ --name ViTUNet --no_LPR_decoder

echo "Running ViT + LPR..."
flairhub --config configs/train/ --name ViTLPR

echo "===== ALL DONE ====="
