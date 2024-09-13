#!/bin/bash

# set visible GPU
export CUDA_VISIBLE_DEVICES=2

# deactivate old venv
deactivate
ml purge

# activate venv
ml TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1
. "./venv_deepmedic/bin/activate"

# run DeepMedic
./deepmedic/deepMedicRun -model "deepmedic_workspace/debug_model/modelConfig_debug1.cfg"    -train "deepmedic_workspace/debug_model/trainConfig_debug1.cfg" -dev cuda
./deepmedic/deepMedicRun -model "deepmedic_workspace/debug_model/modelConfig_debug2.cfg"	-train "deepmedic_workspace/debug_model/trainConfig_debug2.cfg" -dev cuda
