#!/bin/bash

# set visible GPU
export CUDA_VISIBLE_DEVICES=1

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# activate venv
cd "$DIR/../venv_nnunet/bin"
. "activate"

ml Seaborn tqdm scikit-learn scikit-image Pillow IPython PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
export PATH="~/.local/bin:$PATH"

export nnUNet_raw="/datagrid/Medical/TEMPORARY/MRI/smidjak3/nnunet_workspace/nnUNet_raw"
export nnUNet_preprocessed="/datagrid/Medical/TEMPORARY/MRI/smidjak3/nnunet_workspace/nnUNet_preprocessed"
export nnUNet_results="/datagrid/Medical/TEMPORARY/MRI/smidjak3/nnunet_workspace/nnUNet_results"

# graphviz is required for nnUNet visualisation
export PATH="$HOME/graphviz/bin:$PATH"
