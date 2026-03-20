#!/bin/bash

source ~/.bashrc

# Unzip your specific file
unzip -o FLAIR-HUB_FULL.zip -d csv/

pip install huggingface_hub

python flair-hub-HF-dl.py

for f in *.zip; do unzip "$f"; done

for i in {1..5}; do
    flairhub --config configs/train/ --out_model_name Test_$i
done
