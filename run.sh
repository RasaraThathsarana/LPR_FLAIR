#!/bin/bash

if ! command -v tmux &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y tmux
fi

if [ -z "$TMUX" ]; then
    tmux new-session -d -s flair_training "bash $0; exec bash"
    exit 0
fi

source ~/.bashrc

conda activate FLAIRHUB

unzip -o FLAIR-HUB_FULL.zip -d csv/

sed -i 's/;/,/g' csv/FLAIR-HUB_TRAIN.csv csv/FLAIR-HUB_VALID.csv csv/FLAIR-HUB_TEST.csv

pip install huggingface_hub

pip install -e . 

python flair-hub-HF-dl.py

for f in FLAIR-HUB_download/data/*.zip; do unzip "$f" -d FLAIR-HUB_download/data/ && rm "$f"; done

DATA_PATH=$(realpath --relative-to="$(pwd)" FLAIR-HUB_download/data)
sed -i "s|\.\./|${DATA_PATH}/|g" csv/FLAIR-HUB_*.csv

for i in {1..5}; do
    flairhub --config configs/train/ --name Test_$i
done
