#!/bin/bash

# Load default Python virtual environment used for preprocessing, evaluation and for my other scripts.

# set visible GPU
export CUDA_VISIBLE_DEVICES=1

# deactivate old venv
deactivate
ml purge

# activate venv
ml tensorboard/2.15.1-foss-2023a torchvision/0.16.2-foss-2023a-CUDA-12.1.1 PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
. venv/bin/activate
