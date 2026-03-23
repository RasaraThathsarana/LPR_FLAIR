#!/bin/bash
set -e  # stop on error

LOG_FILE="train.log"

echo "Starting script..." | tee -a $LOG_FILE

# Activate conda safely
eval "$(conda shell.bash hook)"
conda activate flairhub

# Install project (safe repeat)
pip install -e .

# Install extra deps
pip install huggingface_hub

# Check dataset
if [ ! -f "FLAIR-HUB_FULL.zip" ]; then
    echo "Dataset zip not found!" | tee -a $LOG_FILE
    exit 1
fi

# Extract dataset safely
mkdir -p csv
unzip -o FLAIR-HUB_FULL.zip -d csv/ >> $LOG_FILE 2>&1

# Fix CSV format (only once)
if [ ! -f "csv/.processed" ]; then
    sed -i 's/;/,/g' csv/FLAIR-HUB_*.csv
    touch csv/.processed
fi

# Download HF data
python flair-hub-HF-dl.py >> $LOG_FILE 2>&1

# Unzip downloaded files
for f in FLAIR-HUB_download/data/*.zip; do
    unzip -o "$f" -d FLAIR-HUB_download/data/
done

# Train
for i in {1..5}; do
    echo "Starting run $i" | tee -a $LOG_FILE
    flairhub --config configs/train/ --name Test_$i >> $LOG_FILE 2>&1
done

echo "Training completed!" | tee -a $LOG_FILE
