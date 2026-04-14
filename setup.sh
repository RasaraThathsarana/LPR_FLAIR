#!/bin/bash
set -e

echo "===== SETUP START ====="

# Install Miniconda if not installed
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b
    
    # Permanently initialize conda in  ~/.bashrc
    ~/miniconda3/bin/conda init bash
fi

# Temporarily add conda to PATH so the next commands work inside this script
export PATH="$HOME/miniconda3/bin:$PATH"

# Enable conda commands for the remainder of this script
eval "$(conda shell.bash hook)"

# Create env
conda env create -f environment.yml || echo "Environment already exists"

echo "===== SETUP DONE ====="
exec bash --init-file <(echo "source ~/.bashrc; conda activate FLAIRHUB")
