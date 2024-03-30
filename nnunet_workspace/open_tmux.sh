#!/bin/bash

tmux attach || {
tmux new-session -A -d -n monitor "watch nvidia-smi" # create new session and detach
tmux split-window -h "htop"
tmux new-window
tmux a
}
