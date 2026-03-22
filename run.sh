#!/bin/bash

source ~/.bashrc

# Unzip your specific file
unzip -o FLAIR-HUB_FULL.zip -d csv/

sed -i 's/;/,/g' csv/FLAIR-HUB_TRAIN.csv csv/FLAIR-HUB_VALID.csv csv/FLAIR-HUB_TEST.csv
DATA_PATH=$(realpath --relative-to="$(pwd)" FLAIR-HUB_download/data)
sed -i "s|\.\./|${DATA_PATH}/|g" csv/FLAIR-HUB_*.csv

pip install huggingface_hub
export HF_TOKEN="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"

pip install -e . 

python flair-hub-HF-dl.py

for f in FLAIR-HUB_download/data/*.zip; do unzip "$f" -d FLAIR-HUB_download/data/ && rm "$f"; done

for i in {1..5}; do
    flairhub --config configs/train/ --name Test_$i
done
