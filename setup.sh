#!/bin/bash
set -e

echo "===== SETUP START ====="

# Install system dependencies
sudo apt update
sudo apt install -y unzip wget curl git

# Install Miniconda if not installed
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b
    ~/miniconda3/bin/conda init bash
fi

export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"

# Recreate env (avoid broken states)
conda env remove -n FLAIRHUB -y || true
conda env create -f environment.yml

conda activate FLAIRHUB

echo "===== DETECTING GPU ====="
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "GPU: $GPU_NAME"

# Decide CUDA version
if [[ "$GPU_NAME" == *"H100"* ]] || [[ "$GPU_NAME" == *"RTX 40"* ]]; then
    echo "Using CUDA 12.1"
    pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Using CUDA 11.8"
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
fi

echo "===== VERIFYING CUDA ====="
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" || exit 1

echo "===== SETUP DONE ====="
exec bash --init-file <(echo "source ~/.bashrc; conda activate FLAIRHUB")
