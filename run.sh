#!/bin/bash

# Proper conda initialization (correct path for your VM)
source /opt/conda/etc/profile.d/conda.sh

# Create env only if not exists
conda env list | grep -q FLAIRHUB || conda create -n FLAIRHUB python=3.10 -y

# Activate env
conda activate FLAIRHUB || exit 1

# Install required packages (IMPORTANT)
pip install --upgrade pip
pip install huggingface_hub
# If you have requirements file:
# pip install -r requirements.txt

# Install flairhub if needed
pip install flairhub  # or correct package name if different

# Run your script
python flair-hub-HF-dl.py

# Safe unzip
for f in *.zip; do
  [ -e "$f" ] || continue
  unzip "$f"
done

# Run training
for i in {1..5}; do
    flairhub --config configs/train/ --out_model_name Test_$i
done
