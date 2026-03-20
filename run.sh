#!/bin/bash

# Initialize conda
source ~/anaconda3/etc/profile.d/conda.sh

# Create environment only if it doesn't exist
conda env list | grep -q FLAIRHUB || conda create -n FLAIRHUB python=3.10 -y

# Activate environment
conda activate FLAIRHUB

# Run your Python script
python flair-hub-HF-dl.py

# Unzip files safely
for f in *.zip; do
  [ -e "$f" ] || continue
  unzip "$f"
done

# Run training loop
for i in {1..5}; do
    flairhub --config configs/train/ --out_model_name Test_$i
done