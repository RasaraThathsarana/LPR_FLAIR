#!/bin/bash
set -e

echo "===== RUN START ====="

# Activate env
eval "$(conda shell.bash hook)"
conda activate FLAIRHUB

# Install project
pip install -e .

# CUDA safety test (VERY IMPORTANT)
echo "===== CUDA TEST ====="
python -c "import torch, torch.nn as nn; x=torch.randn(2,3,224,224).cuda(); m=nn.Conv2d(3,16,3).cuda(); y=m(x); print('cuDNN OK')" || exit 1

# Ensure dataset exists
if [ ! -f "FLAIR-HUB_FULL.zip" ]; then
    echo "ERROR: FLAIR-HUB_FULL.zip not found!"
    exit 1
fi

# Extract dataset
mkdir -p csv
unzip -o FLAIR-HUB_FULL.zip -d csv/

# Fix CSV once
if [ ! -f "csv/.processed" ]; then
    sed -i 's/;/,/g' csv/FLAIR-HUB_*.csv
    touch csv/.processed
fi

# Environment safety
export CUDA_VISIBLE_DEVICES=0
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
