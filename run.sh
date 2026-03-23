#!/bin/bash
set -e

echo "===== RUN START ====="

# Install project (safe repeat)
echo "Installing project..."
pip install -e .

# Ensure dataset exists
if [ ! -f "FLAIR-HUB_FULL.zip" ]; then
    echo "ERROR: FLAIR-HUB_FULL.zip not found!"
    exit 1
fi

# Extract dataset (safe)
echo "Extracting dataset..."
mkdir -p csv
unzip -o FLAIR-HUB_FULL.zip -d csv/

# Fix CSV only once
if [ ! -f "csv/.processed" ]; then
    echo "Processing CSV files..."
    sed -i 's/;/,/g' csv/FLAIR-HUB_*.csv
    touch csv/.processed
fi

# Download HuggingFace data (VISIBLE LOGS)
echo "Downloading HuggingFace data..."
python flair-hub-HF-dl.py

# Unzip downloaded files
echo "Unzipping downloaded files..."
for f in FLAIR-HUB_download/data/*.zip; do
    unzip -o "$f" -d FLAIR-HUB_download/data/
done

# TRAINING (IMPORTANT: SHOW LIVE LOGS)
echo "===== TRAINING START ====="

for i in {1..5}; do
    echo "----- RUN $i -----"
    flairhub --config configs/train/ --name Test_$i
done

echo "===== ALL DONE ====="
