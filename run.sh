#!/bin/bash
set -e

echo "===== RUN START ====="

# Install project
echo "Installing project..."
pip install -e .

# CUDA safety test
echo "===== CUDA TEST ====="
python -c "import torch, torch.nn as nn; x=torch.randn(2,3,224,224).cuda(); m=nn.Conv2d(3,16,3).cuda(); y=m(x); print('cuDNN OK')" || exit 1

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

export CUDA_VISIBLE_DEVICES=0

# Download HuggingFace data
DOWNLOAD_FLAG="FLAIR-HUB_download/.downloaded"
UNZIP_FLAG="FLAIR-HUB_download/.unzipped"
DATA_DIR="FLAIR-HUB_download/data"

if [ -f "$DOWNLOAD_FLAG" ]; then
    echo "Download already completed, skipping download."
else
    echo "Downloading HuggingFace data..."
    python flair-hub-HF-dl.py
    touch "$DOWNLOAD_FLAG"
fi

if [ -f "$UNZIP_FLAG" ]; then
    echo "Unzip already completed, skipping unzip."
else
    echo "Unzipping downloaded files..."
    mkdir -p "$DATA_DIR"
    shopt -s nullglob
    zip_files=("$DATA_DIR"/*.zip)
    shopt -u nullglob

    if [ ${#zip_files[@]} -eq 0 ]; then
        echo "ERROR: No ZIP files found in $DATA_DIR"
        exit 1
    fi

    for f in "${zip_files[@]}"; do
        unzip -o "$f" -d "$DATA_DIR" && rm "$f"
    done

    touch "$UNZIP_FLAG"
fi

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
