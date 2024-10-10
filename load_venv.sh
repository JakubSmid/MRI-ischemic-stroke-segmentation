#!/bin/bash

# Load default Python virtual environment used for preprocessing, evaluation and for my other scripts.

# set visible GPU
export CUDA_VISIBLE_DEVICES=1

# deactivate old venv
deactivate
ml purge

# activate venv
ml PyTorch/2.2.1-foss-2023b-CUDA-12.4.0
. venv/bin/activate
