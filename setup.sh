#!/bin/bash
set -e

# Install Miniconda if not exists
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b
    export PATH="$HOME/miniconda3/bin:$PATH"
fi

eval "$(conda shell.bash hook)"

# Create env
conda env create -f environment.yml || echo "Env already exists"

conda activate flairhub

echo "Setup complete!"
