#!/bin/bash

# set visible GPU
export CUDA_VISIBLE_DEVICES=1

# deactivate old venv
deactivate
ml purge

# activate venv
ml PyTorch/2.3.0-foss-2023b-CUDA-12.4.0
. "./venv_nnunet/bin/activate"

# set environment variables for nnUNet
export PATH="~/.local/bin:$PATH"
export nnUNet_raw="/datagrid/Medical/TEMPORARY/MRI/smidjak3/nnunet_workspace/nnUNet_raw"
export nnUNet_preprocessed="/datagrid/Medical/TEMPORARY/MRI/smidjak3/nnunet_workspace/nnUNet_preprocessed"
export nnUNet_results="/datagrid/Medical/TEMPORARY/MRI/smidjak3/nnunet_workspace/nnUNet_results"

# graphviz is required for nnUNet visualisation
export PATH="$HOME/graphviz/bin:$PATH"
