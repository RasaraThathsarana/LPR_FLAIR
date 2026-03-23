#!/bin/bash
set -e

echo "===== SETUP START ====="

# Install Miniconda if not installed
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b
    export PATH="$HOME/miniconda3/bin:$PATH"
fi

# Enable conda
eval "$(conda shell.bash hook)"

# Create env (safe if already exists)
conda env create -f environment.yml || echo "Environment already exists"

echo "===== SETUP DONE ====="
