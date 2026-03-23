#!/bin/bash

SESSION_NAME="flair_training"

if ! command -v tmux &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y tmux
fi

tmux new-session -d -s $SESSION_NAME "bash run.sh"

echo "Started tmux session: $SESSION_NAME"
echo "Attach using: tmux attach -t $SESSION_NAME"
