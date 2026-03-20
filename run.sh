#!/bin/bash

# Initialize conda
source /opt/conda/etc/profile.d/conda.sh

# Create env if not exists
conda env list | grep -q FLAIRHUB || conda create -n FLAIRHUB python=3.10 -y

# Activate env
conda activate FLAIRHUB || exit 1

# 🔽 Unzip your specific file
unzip -o FLAIR-HUB_FULL.zip -d csv/

# Run your script
python flair-hub-HF-dl.py

# Unzip any additional zip files (optional)
for f in *.zip; do
  [ -e "$f" ] || continue
  unzip "$f"
done

# Training loop
for i in {1..5}; do
    flairhub --config configs/train/ --out_model_name Test_$i
done
